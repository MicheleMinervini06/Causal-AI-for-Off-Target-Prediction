"""Model layer for baseline and deep CRISPR predictors."""

from models.train import run_baseline_pipeline, run_cbm_pipeline, run_deep_pipeline

__all__ = ["run_baseline_pipeline", "run_deep_pipeline", "run_cbm_pipeline"]
