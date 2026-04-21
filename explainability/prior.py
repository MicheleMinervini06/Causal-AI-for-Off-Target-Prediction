from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_bio_prior(change_seq_df: pd.DataFrame, n_positions: int = 23) -> np.ndarray:
    """Estimate empirical position prior pi_bio from CHANGE-seq-like tables."""
    if change_seq_df.empty:
        return np.full(n_positions, 1.0 / n_positions)

    position_cols = [f"mismatch_pos_{i}" for i in range(n_positions)]
    if set(position_cols).issubset(change_seq_df.columns):
        prior = change_seq_df[position_cols].mean(axis=0).to_numpy(dtype=float)
    elif "mismatch_count" in change_seq_df.columns:
        # Fallback: spread mismatch burden uniformly if per-position data is unavailable.
        mean_mismatch = float(change_seq_df["mismatch_count"].mean())
        prior = np.full(n_positions, mean_mismatch / max(1, n_positions))
    else:
        prior = np.ones(n_positions, dtype=float)

    prior = np.clip(prior, a_min=0.0, a_max=None)
    total = prior.sum()
    if total <= 0:
        return np.full(n_positions, 1.0 / n_positions)
    return prior / total
