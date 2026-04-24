from __future__ import annotations

from itertools import product
from typing import Any, Mapping

import numpy as np
import pandas as pd

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


def _children_map(dag: Mapping[str, list[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for child, parents in dag.items():
        out.setdefault(child, set())
        for parent in parents:
            out.setdefault(parent, set()).add(child)
    return out


def _ancestors_of(node: str, dag: Mapping[str, list[str]]) -> set[str]:
    visited: set[str] = set()
    stack: list[str] = [node]
    while stack:
        current = stack.pop()
        for parent in dag.get(current, []):
            if parent not in visited:
                visited.add(parent)
                stack.append(parent)
    return visited


def _descendants_of(node: str, dag: Mapping[str, list[str]]) -> set[str]:
    children = _children_map(dag)
    visited: set[str] = set()
    stack: list[str] = [node]
    while stack:
        current = stack.pop()
        for child in children.get(current, set()):
            if child not in visited:
                visited.add(child)
                stack.append(child)
    return visited


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
) -> pd.DataFrame:
    out = df.copy()

    for key, value in intervention.items():
        if key not in out.columns:
            raise ValueError(f"Intervention key not in dataframe: {key}")
        out[key] = value

    # Topological order for the current SCM equations.
    equations = [
        ("pam_score", scm.pam_equation),
        ("mismatch_rate", scm.mismatch_equation),
        ("label", scm.activity_equation),
    ]

    for target, equation in equations:
        if target in intervention:
            continue
        for parent in equation.parents:
            if parent not in out.columns:
                raise ValueError(f"Missing parent column '{parent}' required by equation '{target}'")
        predicted = equation.predict(out)
        if target == "label":
            out["activity_probability"] = predicted
            out["label"] = (predicted >= 0.5).astype(int)
        else:
            out[target] = predicted

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
    """Return a practical backdoor adjustment set from a known DAG.

    This implementation is intentionally conservative and returns observable nodes
    that are common ancestors of treatment and outcome, excluding descendants of
    treatment and excluding treatment/outcome themselves.
    """
    graph = _ensure_dag(dag)
    treatment = str(treatment)
    outcome = str(outcome)

    anc_t = _ancestors_of(treatment, graph)
    anc_y = _ancestors_of(outcome, graph)
    descendants_t = _descendants_of(treatment, graph)

    candidates = (anc_t & anc_y) - descendants_t - {treatment, outcome}
    return set(sorted(candidates))


def do_query(
    scm: CRISPRCausalModel,
    df: pd.DataFrame,
    intervention: dict[str, Any],
) -> dict[str, Any]:
    """Estimate P(Y|do(X=x)) using SCM-based g-computation.

    The estimate is computed by clamping intervention variables and propagating
    descendants through fitted structural equations while averaging over the
    empirical distribution of non-intervened variables.
    """
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
    """Generate a synthetic intervention dataset from observed rows.

    This function applies clamped interventions to each row and appends metadata
    columns to track source example and intervention specification.
    """
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
