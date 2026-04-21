from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def explanation_stability_score(explanations: Sequence[np.ndarray]) -> float:
    """Average pairwise cosine similarity between explanation vectors."""
    vectors = [np.asarray(exp, dtype=float).reshape(-1) for exp in explanations if np.asarray(exp).size > 0]
    if len(vectors) < 2:
        return 1.0

    scores = [_cosine_similarity(a, b) for a, b in combinations(vectors, 2)]
    return float(np.mean(scores))
