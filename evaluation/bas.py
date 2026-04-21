from __future__ import annotations

import numpy as np


def _resize_vector(vector: np.ndarray, target_len: int) -> np.ndarray:
    if len(vector) == target_len:
        return vector
    x_old = np.linspace(0, 1, num=len(vector), endpoint=True)
    x_new = np.linspace(0, 1, num=target_len, endpoint=True)
    return np.interp(x_new, x_old, vector)


def biological_alignment_score(explanations: np.ndarray, bio_prior: np.ndarray) -> float:
    """Correlation-based alignment score between explanation mass and pi_bio."""
    exp = np.asarray(explanations, dtype=float)
    prior = np.asarray(bio_prior, dtype=float).reshape(-1)

    if exp.ndim > 1:
        exp = np.mean(np.abs(exp), axis=0)
    else:
        exp = np.abs(exp.reshape(-1))

    if len(exp) == 0 or len(prior) == 0:
        return 0.0

    exp = _resize_vector(exp, len(prior))

    if np.isclose(exp.std(), 0.0) or np.isclose(prior.std(), 0.0):
        return 0.0

    corr = float(np.corrcoef(exp, prior)[0, 1])
    # Map from [-1, 1] to [0, 1].
    return max(0.0, min(1.0, 0.5 * (corr + 1.0)))
