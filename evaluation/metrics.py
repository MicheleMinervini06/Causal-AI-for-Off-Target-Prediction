from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def evaluate_model(model: Any, x: Any, y_true: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Unified classification metrics for any model exposing predict_proba()."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(model.predict_proba(x), dtype=float).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, float] = {
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics
