from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - handled at runtime
    CatBoostClassifier = None


HYPERPARAMETER_GRID: dict[str, list[Any]] = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "iterations": [200, 500],
    "l2_leaf_reg": [3, 5, 7],
}


@dataclass
class CatBoostWrapper:
    params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed. Run: uv sync")

        default = {
            "depth": 6,
            "learning_rate": 0.08,
            "iterations": 300,
            "loss_function": "Logloss",
            "verbose": False,
            "random_seed": 42,
        }
        if self.params:
            default.update(self.params)
        self.model = CatBoostClassifier(**default)

    @property
    def hyperparameter_grid(self) -> dict[str, list[Any]]:
        return HYPERPARAMETER_GRID

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CatBoostWrapper":
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)

    def explain(self, x: np.ndarray) -> np.ndarray:
        feature_importance = np.asarray(self.model.get_feature_importance(), dtype=float)
        return np.tile(feature_importance, (x.shape[0], 1))
