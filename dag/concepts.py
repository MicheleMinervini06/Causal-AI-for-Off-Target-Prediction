from typing import Any, Mapping

CONCEPT_KEYS = (
    "pam_score",
    "mismatch_count",
    "seed_mismatch_count",
    "gc_guide",
    "gc_target",
    "mean_energy_penalty",
    "total_energy_penalty",
)


def to_concept_dict(feature_row: Mapping[str, Any]) -> dict[str, float]:
    """Map engineered features to a CBM-ready concept vector."""
    concepts: dict[str, float] = {}
    for key in CONCEPT_KEYS:
        concepts[key] = float(feature_row.get(key, 0.0))
    return concepts
