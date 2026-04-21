import numpy as np

from dag.mismatch import mismatch_energy_penalty, mismatch_type


def positional_energy_profile(guide_seq: str, target_seq: str) -> np.ndarray:
    """Estimate per-position energy penalties for a sequence pair."""
    guide_seq = guide_seq.upper()
    target_seq = target_seq.upper()
    length = min(len(guide_seq), len(target_seq))

    profile = np.zeros(length, dtype=float)
    for i in range(length):
        kind = mismatch_type(guide_seq[i], target_seq[i])
        profile[i] = mismatch_energy_penalty(kind, i)
    return profile
