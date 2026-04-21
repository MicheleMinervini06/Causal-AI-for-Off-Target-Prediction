from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - handled at runtime
    XGBClassifier = None


HYPERPARAMETER_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [100, 300],
    "subsample": [0.8, 1.0],
}


@dataclass
class XGBoostWrapper:
    params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Run: uv sync")

        default = {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
        }
        if self.params:
            default.update(self.params)
        self.model = XGBClassifier(**default)

    @property
    def hyperparameter_grid(self) -> dict[str, list[Any]]:
        return HYPERPARAMETER_GRID

    def fit(self, x: np.ndarray, y: np.ndarray) -> "XGBoostWrapper":
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)

    def explain(self, x: np.ndarray) -> np.ndarray:
        importances = np.asarray(getattr(self.model, "feature_importances_", np.zeros(x.shape[1])))
        return np.tile(importances, (x.shape[0], 1))
