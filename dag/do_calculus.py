from __future__ import annotations

from itertools import product
from typing import Any, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.inference import CausalInference
from pgmpy.models import BayesianNetwork

from dag.scm import CRISPRCausalModel

# child -> parents
DEFAULT_DAG: dict[str, list[str]] = {
    "pam_score": ["node_A_pam"],
    "mismatch_rate": [
        "node_B_proximal",
        "node_C_seed_extension",
        "node_D_non_seed",
        "mean_energy_penalty",
    ],
    "label": ["pam_score", "node_B_proximal", "node_C_seed_extension", "node_D_non_seed"],
}


def _ensure_dag(dag: Mapping[str, list[str]] | None) -> dict[str, list[str]]:
    if dag is None:
        return {k: list(v) for k, v in DEFAULT_DAG.items()}
    return {str(k): [str(parent) for parent in v] for k, v in dag.items()}


def _build_nx_graph(dag: Mapping[str, list[str]]) -> nx.DiGraph:
    """Costruisce un DiGraph di NetworkX dal dizionario DAG."""
    G = nx.DiGraph()
    for child, parents in dag.items():
        for parent in parents:
            G.add_edge(parent, child)
    return G


def _normalize_intervention(intervention: Mapping[str, Any]) -> dict[str, Any]:
    if not intervention:
        raise ValueError("intervention cannot be empty")
    out = {str(k): v for k, v in intervention.items()}
    if any(v is None for v in out.values()):
        raise ValueError("intervention values cannot be None")
    return out


def _observational_mask(df: pd.DataFrame, intervention: Mapping[str, Any]) -> pd.Series:
    mask = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    for key, value in intervention.items():
        if key not in df.columns:
            raise ValueError(f"Intervention key not in dataframe: {key}")
        if isinstance(value, (float, int, np.floating, np.integer)):
            mask &= np.isclose(df[key].to_numpy(dtype=float), float(value), atol=1e-8)
        else:
            mask &= df[key] == value
    return mask


def _propagate_scm_under_intervention(
    scm: CRISPRCausalModel,
    df: pd.DataFrame,
    intervention: Mapping[str, Any],
    dag: Mapping[str, list[str]] | None = None,
) -> pd.DataFrame:
    out = df.copy()

    for key, value in intervention.items():
        if key not in out.columns:
            raise ValueError(f"Intervention key not in dataframe: {key}")
        out[key] = value

    # Genera l'ordinamento topologico dinamicamente (Fix Architetturale)
    graph = _build_nx_graph(_ensure_dag(dag))
    try:
        topological_order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("Il DAG contiene cicli e non può essere ordinato topologicamente.")

    # Mappiamo i target alle equazioni disponibili nell'SCM
    scm_equations = {
        eq.target: eq
        for eq in [scm.pam_equation, scm.mismatch_equation, scm.activity_equation]
    }

    # Propaghiamo seguendo l'ordine del DAG
    for target in topological_order:
        if target in intervention or target not in scm_equations:
            continue
        
        equation = scm_equations[target]
        for parent in equation.parents:
            if parent not in out.columns:
                raise ValueError(f"Missing parent column '{parent}' required by equation '{target}'")
        
        predicted = equation.predict(out)
        
        if target == "label":
            out["activity_probability"] = predicted
            out["label"] = (predicted >= 0.5).astype(int)
        else:
            out[target] = predicted

    # Fallback nel caso estremo di do(label=x)
    if "activity_probability" not in out.columns:
        if "label" in intervention:
            out["activity_probability"] = np.clip(out["label"].to_numpy(dtype=float), 0.0, 1.0)
        else:
            out["activity_probability"] = scm.activity_equation.predict(out)
            
    return out


def _expand_interventions(intervention: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = list(intervention.keys())
    values_lists: list[list[Any]] = []
    for key in keys:
        value = intervention[key]
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            as_list = list(value)
            if len(as_list) == 0:
                raise ValueError(f"Intervention list for '{key}' cannot be empty")
            values_lists.append(as_list)
        else:
            values_lists.append([value])
    return [dict(zip(keys, combo)) for combo in product(*values_lists)]


def backdoor_adjustment(
    dag: Mapping[str, list[str]] | None,
    treatment: str,
    outcome: str,
) -> set[str]:
    """Return a valid backdoor adjustment set using pgmpy.

    This ensures full d-separation and prevents collider bias, unlike
    simple common-ancestor heuristics.
    """
    graph_dict = _ensure_dag(dag)
    edges = []
    for child, parents in graph_dict.items():
        for parent in parents:
            edges.append((parent, child))

    model = BayesianNetwork(edges)
    inference = CausalInference(model)
    
    # pgmpy estrae formalmente i set validi per il backdoor criterion
    backdoor_sets = inference.get_all_backdoor_adjustment_sets(treatment, outcome)
    
    if not backdoor_sets:
        return set()
        
    # Restituisce il set più piccolo tra quelli validi
    return set(min(backdoor_sets, key=len))


def do_query(
    scm: CRISPRCausalModel,
    df: pd.DataFrame,
    intervention: dict[str, Any],
) -> dict[str, Any]:
    """Estimate P(Y|do(X=x)) using SCM-based g-computation."""
    if not scm.fitted:
        raise RuntimeError("SCM must be fitted before calling do_query")
    if len(df) == 0:
        raise ValueError("df cannot be empty")

    intervention_clean = _normalize_intervention(intervention)
    df_do = _propagate_scm_under_intervention(scm, df, intervention_clean)
    p_do = float(df_do["activity_probability"].mean())

    mask_obs = _observational_mask(df, intervention_clean)
    n_obs = int(mask_obs.sum())
    if n_obs > 0:
        p_obs = float(df.loc[mask_obs, "label"].to_numpy(dtype=float).mean())
    else:
        p_obs = float("nan")

    delta = float(p_do - p_obs) if np.isfinite(p_obs) else float("nan")

    return {
        "intervention": dict(intervention_clean),
        "p_do": p_do,
        "p_observational": p_obs,
        "delta_do_minus_obs": delta,
        "n_observational": n_obs,
    }


def compare_observational_vs_interventional(
    scm: CRISPRCausalModel,
    df: pd.DataFrame,
    intervention: dict[str, Any],
) -> pd.DataFrame:
    """Compare associative and interventional responses for one or many interventions."""
    expanded = _expand_interventions(_normalize_intervention(intervention))
    rows: list[dict[str, Any]] = []
    for spec in expanded:
        result = do_query(scm=scm, df=df, intervention=spec)
        row = {
            "treatment": ",".join(sorted(spec.keys())),
            "value": ";".join(f"{k}={v}" for k, v in sorted(spec.items())),
            "p_observational": result["p_observational"],
            "p_do": result["p_do"],
            "delta_do_minus_obs": result["delta_do_minus_obs"],
            "n_observational": result["n_observational"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_intervention_dataset(
    df: pd.DataFrame,
    interventions: list[dict[str, Any]],
) -> pd.DataFrame:
    """Generate a synthetic intervention dataset from observed rows."""
    if len(df) == 0:
        raise ValueError("df cannot be empty")
    if not interventions:
        raise ValueError("interventions cannot be empty")

    blocks: list[pd.DataFrame] = []
    base = df.reset_index(drop=False).rename(columns={"index": "source_index"})

    for idx, raw_spec in enumerate(interventions):
        spec = _normalize_intervention(raw_spec)
        expanded_specs = _expand_interventions(spec)
        for jdx, expanded in enumerate(expanded_specs):
            block = base.copy()
            for key, value in expanded.items():
                if key not in block.columns:
                    raise ValueError(f"Intervention key not in dataframe: {key}")
                block[key] = value
            block["intervention_id"] = f"int_{idx:03d}_{jdx:03d}"
            block["intervention_spec"] = str(dict(sorted(expanded.items())))
            block["is_intervened"] = 1
            blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


__all__ = [
    "DEFAULT_DAG",
    "backdoor_adjustment",
    "build_intervention_dataset",
    "compare_observational_vs_interventional",
    "do_query",
]