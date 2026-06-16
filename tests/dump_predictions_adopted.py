"""Standalone: load the adopted NeuralSCM checkpoint and dump per-instance
predictions on the within-CHANGE-seq held-out test split and on GUIDE-seq,
for downstream statistical comparison (DeLong / paired bootstrap) against
other models.

No retraining. Pure forward pass on the saved checkpoint.

The adopted model is currently Exp30 (biological_mismatch encoder with
positional_use_encoder=true); if the adopted configuration changes, update
the four ADOPTED_* constants below.

Output (at OUT_DIR):
    predictions_neuralscm_test.parquet
    predictions_neuralscm_guideseq.parquet

Schema (both files):  sgRNA_seq | off_seq | label | prob
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.deep import NeuralSCM
from models.deep.encoding import BiologicalMismatchEncoder


# ─── Adopted model configuration (update when the adopted model changes) ────
ADOPTED_RUN_DIR = "Exp30_Ablation_EncoderBiologicalMismatch_UseEncoder"
ADOPTED_CFG = "config_exp30_ablation_encoder_biological_mismatch_use_encoder.yaml"

CFG_PATH  = ROOT / "experiments/exp_03_neural_scm" / ADOPTED_CFG
CKPT_PATH = ROOT / "experiments/results" / ADOPTED_RUN_DIR / "neural_scm.pt"
OUT_DIR   = ROOT / "experiments/results" / ADOPTED_RUN_DIR

TEST_PARQUET  = ROOT / "data/processed/splits_merged/test.parquet"
GUIDE_PARQUET = ROOT / "data/processed/features/guideseq_features.parquet"


def _load_config_with_base(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cur = cfg
    while isinstance(cur, dict) and "_base" in cur:
        base_rel = cur.pop("_base")
        base_path = ROOT / base_rel
        base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        merged = dict(base_cfg)
        for k, v in cur.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                tmp = dict(merged[k])
                tmp.update(v)
                merged[k] = tmp
            else:
                merged[k] = v
        cur = merged
    return cur


def _predict(model, df: pd.DataFrame, context_cols: list[str], batch_size: int = 4096) -> np.ndarray:
    sgrnas = df["sgRNA_seq"].astype(str).tolist()
    offs   = df["off_seq"].astype(str).tolist()
    if context_cols:
        ctx = df[context_cols].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        ctx = None

    probs = np.empty(len(sgrnas), dtype=np.float32)
    n = len(sgrnas)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            # .copy() to ensure writable buffer (silences read-only warning)
            ctx_b = torch.from_numpy(ctx[i:j].copy()) if ctx is not None else None
            out = model(sgrnas[i:j], offs[i:j], context_features=ctx_b)
            p = out["activity_probability"].squeeze(-1).cpu().numpy()
            probs[i:j] = p
            print(f"  ... {j}/{n}", flush=True)
    return probs


def main() -> None:
    print(f"Adopted run: {ADOPTED_RUN_DIR}")
    print(f"Loading config: {CFG_PATH}")
    cfg = _load_config_with_base(CFG_PATH)
    model_cfg = cfg.get("model", {})
    context_cols = model_cfg.get("context_cols", [])

    print(f"Building model: arch={model_cfg.get('architecture')} "
          f"pam_mode={model_cfg.get('pam_mode')} context_dim={len(context_cols)} "
          f"positional_use_encoder={model_cfg.get('positional_use_encoder', False)}")
    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(
        architecture=str(model_cfg.get("architecture", "positional_mlp")),
        embed_dim=int(model_cfg.get("embed_dim", 16)),
        hidden_dim=int(model_cfg.get("hidden_dim", 8)),
        encoder=encoder,
        context_dim=len(context_cols),
        variational=bool(model_cfg.get("variational", False)),
        pam_mode=str(model_cfg.get("pam_mode", "additive")),
        positional_use_encoder=bool(model_cfg.get("positional_use_encoder", False)),
    )

    print(f"Loading checkpoint: {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── CHANGE-seq held-out test ───────────────────────────────────────
    print(f"\nPredicting on test set: {TEST_PARQUET}")
    test_df = pd.read_parquet(TEST_PARQUET)
    probs = _predict(model, test_df, context_cols)
    out = test_df[["sgRNA_seq", "off_seq", "label"]].reset_index(drop=True).copy()
    out["prob"] = probs.astype(np.float32)
    out_path = OUT_DIR / "predictions_neuralscm_test.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  -> {out_path} ({len(out)} rows)")

    # ── GUIDE-seq cross-assay ──────────────────────────────────────────
    print(f"\nPredicting on GUIDE-seq: {GUIDE_PARQUET}")
    guide_df = pd.read_parquet(GUIDE_PARQUET)
    probs = _predict(model, guide_df, context_cols)
    out = guide_df[["sgRNA_seq", "off_seq", "label"]].reset_index(drop=True).copy()
    out["prob"] = probs.astype(np.float32)
    out_path = OUT_DIR / "predictions_neuralscm_guideseq.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  -> {out_path} ({len(out)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
