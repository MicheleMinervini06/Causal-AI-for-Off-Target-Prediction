"""Explainability utilities for CRISPR off-target models."""

from explainability.attention import attention_rollout, aggregate_heads
from explainability.ig import integrated_gradients
from explainability.prior import estimate_bio_prior
from explainability.shap_utils import normalize_to_unit_interval, tree_shap_importance

__all__ = [
    "normalize_to_unit_interval",
    "tree_shap_importance",
    "integrated_gradients",
    "aggregate_heads",
    "attention_rollout",
    "estimate_bio_prior",
]
