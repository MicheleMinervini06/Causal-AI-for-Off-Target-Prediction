from __future__ import annotations

import numpy as np
import pandas as pd


def generate_contrastive_dataset(
    base_df: pd.DataFrame,
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create positive/negative synthetic pairs by target shuffling."""
    if base_df.empty:
        return pd.DataFrame(columns=["guide_seq", "target_seq", "label"])

    required = {"guide_seq", "target_seq"}
    if not required.issubset(base_df.columns):
        raise ValueError("base_df must contain guide_seq and target_seq columns")

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(base_df.index.to_numpy(), size=min(n_samples, len(base_df)), replace=False)
    sampled = base_df.loc[sample_idx, ["guide_seq", "target_seq"]].reset_index(drop=True)

    positives = sampled.copy()
    positives["label"] = 1

    negatives = sampled.copy()
    negatives["target_seq"] = rng.permutation(negatives["target_seq"].to_numpy())
    negatives["label"] = 0

    return pd.concat([positives, negatives], ignore_index=True)
