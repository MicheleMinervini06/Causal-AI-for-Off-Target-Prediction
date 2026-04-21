from __future__ import annotations

import numpy as np


def cross_assay_explanation_transferability(
    source_explanations: np.ndarray,
    target_explanations: np.ndarray,
) -> float:
    """Cosine similarity between assay-level explanation signatures."""
    source = np.asarray(source_explanations, dtype=float)
    target = np.asarray(target_explanations, dtype=float)

    if source.ndim > 1:
        source = np.mean(np.abs(source), axis=0)
    if target.ndim > 1:
        target = np.mean(np.abs(target), axis=0)

    source = source.reshape(-1)
    target = target.reshape(-1)
    if source.size == 0 or target.size == 0:
        return 0.0

    min_len = min(len(source), len(target))
    source = source[:min_len]
    target = target[:min_len]

    denom = (np.linalg.norm(source) * np.linalg.norm(target)) + 1e-8
    cosine = float(np.dot(source, target) / denom)
    return max(0.0, min(1.0, 0.5 * (cosine + 1.0)))
