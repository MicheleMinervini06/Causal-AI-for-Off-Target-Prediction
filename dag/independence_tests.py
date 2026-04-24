from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from dag.do_calculus import DEFAULT_DAG


def _ensure_dag(dag: Mapping[str, list[str]] | None) -> dict[str, list[str]]:
    if dag is None:
        return {k: list(v) for k, v in DEFAULT_DAG.items()}
    return {str(k): [str(parent) for parent in v] for k, v in dag.items()}


def _as_conditioning_list(Z: str | Sequence[str] | None) -> list[str]:
    if Z is None:
        return []
    if isinstance(Z, str):
        return [Z]
    return [str(v) for v in Z]


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Family-Wise Error Rate (FWER) step-down correction."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)

    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return adjusted

    p_finite = p[finite_mask]
    m = len(p_finite)
    order = np.argsort(p_finite)
    sorted_p = p_finite[order]

    adj_p = np.empty(m, dtype=float)
    current_max = 0.0
    for i, p_val in enumerate(sorted_p):
        # Formula di Holm: p * (m - i)
        val = p_val * (m - i)
        current_max = max(current_max, val)
        adj_p[i] = min(current_max, 1.0)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    adjusted[finite_mask] = adj_p[inv_order]
    return adjusted


@dataclass(slots=True)
class ConditionalIndependenceSpec:
    X: str
    Y: str
    Z: list[str]
    source: str = "pgmpy_minimal_d_sep"


def _extract_minimal_d_separations(
    dag: Mapping[str, list[str]],
) -> list[ConditionalIndependenceSpec]:
    """
    Estrae le indipendenze condizionate usando pgmpy con deduplicazione rigorosa 
    per proteggere la potenza statistica della correzione FWER.
    """
    edges = []
    for child, parents in dag.items():
        for parent in parents:
            edges.append((parent, child))
            
    if not edges:
        return []

    model = BayesianNetwork(edges)
    independencies = model.get_independencies()
    
    specs: list[ConditionalIndependenceSpec] = []
    seen_signatures = set()
    
    for assertion in independencies.get_assertions():
        for x in assertion.event1:
            for y in assertion.event2:
                # Ordinamento canonico per eliminare simmetrie
                node_a, node_b = sorted([x, y])
                cond_set = tuple(sorted(assertion.event3))
                
                signature = (node_a, node_b, cond_set)
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    specs.append(
                        ConditionalIndependenceSpec(
                            X=node_a,
                            Y=node_b,
                            Z=list(cond_set)
                        )
                    )
                    
    return specs


def _get_dml_residuals(Z_mat: np.ndarray, target_vec: np.ndarray, is_binary: bool) -> np.ndarray:
    """Estrae i residui ortogonali usando Cross-Fitting (Double Machine Learning)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    residuals = np.zeros_like(target_vec, dtype=float)

    for train_idx, test_idx in kf.split(Z_mat):
        Z_train, Z_test = Z_mat[train_idx], Z_mat[test_idx]
        y_train, y_test = target_vec[train_idx], target_vec[test_idx]

        if is_binary:
            if len(np.unique(y_train)) < 2:
                pred = np.full(len(y_test), float(y_train[0]))
            else:
                model = HistGradientBoostingClassifier(
                    max_iter=50, max_depth=5, early_stopping=False, random_state=42
                )
                model.fit(Z_train, y_train)
                pred = model.predict_proba(Z_test)[:, 1] 
        else:
            model = HistGradientBoostingRegressor(
                max_iter=50, max_depth=5, early_stopping=False, random_state=42
            )
            model.fit(Z_train, y_train)
            pred = model.predict(Z_test)

        residuals[test_idx] = y_test - pred

    return residuals


def test_conditional_independence(
    df: pd.DataFrame,
    X: str,
    Y: str,
    Z: str | Sequence[str] | None = None,
    *,
    alpha: float = 0.05,
) -> dict[str, float | int | str | bool]:
    """Test X ⟂ Y | Z utilizzando Double Machine Learning per i residui."""
    
    z_cols = _as_conditioning_list(Z)
    required = [X, Y] + z_cols
    missing = [c for c in required if c not in df.columns]
    
    base_result = {
        "X": X, "Y": Y, "Z": ",".join(z_cols),
        "n": 0, "rho_partial": float("nan"), "p_value": float("nan"),
        "independent": False,
    }

    if missing:
        return {**base_result, "status": f"missing_columns:{','.join(missing)}"}

    work = df[required].dropna().copy()
    n = len(work)
    
    # DML richiede una soglia minima di dati per il K-Fold
    if n < max(20, len(z_cols) * 5): 
        return {**base_result, "n": int(n), "status": "insufficient_data"}

    x_vec = work[X].to_numpy(dtype=float)
    y_vec = work[Y].to_numpy(dtype=float)

    x_is_binary = len(np.unique(x_vec)) == 2
    y_is_binary = len(np.unique(y_vec)) == 2

    if z_cols:
        Z_mat = work[z_cols].to_numpy(dtype=float)
        rx = _get_dml_residuals(Z_mat, x_vec, x_is_binary)
        ry = _get_dml_residuals(Z_mat, y_vec, y_is_binary)
    else:
        rx = x_vec - x_vec.mean()
        ry = y_vec - y_vec.mean()

    sx = float(np.std(rx))
    sy = float(np.std(ry))
    
    if np.isclose(sx, 0.0) or np.isclose(sy, 0.0):
        return {**base_result, "n": int(n), "status": "degenerate_residuals"}

    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho = float(np.clip(rho, -0.999999, 0.999999))

    dof = n - len(z_cols) - 2
    if dof <= 0:
        return {**base_result, "n": int(n), "status": "non_positive_dof"}

    t_stat = rho * np.sqrt(dof / (1.0 - rho**2))
    p_value = float(2.0 * stats.t.sf(np.abs(t_stat), df=dof))

    return {
        **base_result,
        "n": int(n),
        "rho_partial": rho,
        "p_value": p_value,
        "independent": bool(p_value >= alpha),
        "status": "ok",
    }


def validate_dag_implications(
    dag: Mapping[str, list[str]] | None,
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Validate conditional-independence using pgmpy d-separation, DML and FWER."""
    graph = _ensure_dag(dag)
    specs = _extract_minimal_d_separations(graph)

    rows: list[dict[str, float | int | str | bool]] = []
    for spec in specs:
        row = test_conditional_independence(df, spec.X, spec.Y, spec.Z, alpha=alpha)
        row["source"] = spec.source
        rows.append(row)

    report = pd.DataFrame(rows)
    if report.empty:
        return report

    report["p_value_fwer"] = _holm_bonferroni(report["p_value"].to_numpy(dtype=float))
    report["independent_fwer"] = (report["p_value_fwer"] >= alpha).fillna(False)
    report["reject_h0_fwer"] = (report["p_value_fwer"] < alpha).fillna(False)
    report["status_fwer"] = np.where(report["status"] == "ok", "ok", report["status"])

    ordered_cols = [
        "X", "Y", "Z", "source", "n", "rho_partial", "p_value", 
        "p_value_fwer", "independent", "independent_fwer", 
        "reject_h0_fwer", "status", "status_fwer"
    ]
    existing_cols = [c for c in ordered_cols if c in report.columns]
    return report[existing_cols].sort_values(["Y", "X", "Z"]).reset_index(drop=True)


__all__ = [
    "ConditionalIndependenceSpec",
    "test_conditional_independence",
    "validate_dag_implications",
]