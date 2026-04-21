from __future__ import annotations

import numpy as np


def normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    v_min = values.min()
    v_max = values.max()
    if np.isclose(v_min, v_max):
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def tree_shap_importance(model: object, x: np.ndarray) -> np.ndarray:
    """Compute normalized TreeSHAP importances or fallback to model.explain."""
    x = np.asarray(x)

    try:
        import shap

        estimator = getattr(model, "model", model)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(x)

        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1], dtype=float)
        else:
            values = np.asarray(shap_values, dtype=float)

        if values.ndim == 1:
            values = values[:, None]
        if values.ndim == 3:
            values = values[..., 0]

        return normalize_to_unit_interval(np.abs(values))
    except Exception:
        if hasattr(model, "explain"):
            raw = np.asarray(model.explain(x), dtype=float)
            return normalize_to_unit_interval(np.abs(raw))
        raise
