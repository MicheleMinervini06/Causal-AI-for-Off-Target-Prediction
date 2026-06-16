"""Standalone test of the Causal Consistency Score on the adopted model.

The adopted model is currently Exp30 (biological_mismatch encoder with
positional_use_encoder=true, additive PAM, lambda_causal=0.10); if the
adopted configuration changes, update the ADOPTED_* constants below.

The five rules (R1 PAM ablation, R2 PAM hierarchy, R3 seed vs distal,
R4 positional gradient, R5 healing under intervention) are defined in
Section 3.8 of the methodology chapter and implemented in
evaluation/ccs.py:calculate_ccs_neural_v2 (the historical "v2" tag in the
function name is the implementation that targets per-position
interventions; this is the only CCS reported in the thesis).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.ccs import calculate_ccs_neural_v2
from models.deep import NeuralSCM
from models.deep.encoding import BiologicalMismatchEncoder


# ─── Adopted model configuration (update when the adopted model changes) ────
ADOPTED_RUN_DIR = "Exp30_Ablation_EncoderBiologicalMismatch_UseEncoder"
ADOPTED_CFG = "config_exp30_ablation_encoder_biological_mismatch_use_encoder.yaml"

CFG_PATH  = ROOT / "experiments/exp_03_neural_scm" / ADOPTED_CFG
CKPT_PATH = ROOT / "experiments/results" / ADOPTED_RUN_DIR / "neural_scm.pt"
TEST_SPLIT = ROOT / "data/processed/splits_merged/test.parquet"
OUT_JSON   = ROOT / "experiments/results" / ADOPTED_RUN_DIR / "ccs_neural.json"


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


def main() -> None:
    cfg = _load_config_with_base(CFG_PATH)
    model_cfg = cfg.get("model", {})

    architecture = str(model_cfg.get("architecture", "positional_mlp"))
    embed_dim = int(model_cfg.get("embed_dim", 16))
    hidden_dim = int(model_cfg.get("hidden_dim", 8))
    pam_mode = str(model_cfg.get("pam_mode", "additive"))
    positional_use_encoder = bool(model_cfg.get("positional_use_encoder", False))
    context_cols = model_cfg.get("context_cols", [])
    context_dim = len(context_cols)

    print(f"Adopted run:        {ADOPTED_RUN_DIR}")
    print(f"Architecture:       {architecture}")
    print(f"PAM mode:           {pam_mode}")
    print(f"positional_use_enc: {positional_use_encoder}")
    print(f"Encoder:            BiologicalMismatchEncoder")
    print(f"Context dim:        {context_dim}")
    print()

    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(
        architecture=architecture,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        encoder=encoder,
        context_dim=context_dim,
        variational=False,
        pam_mode=pam_mode,
        positional_use_encoder=positional_use_encoder,
    )

    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"WARN missing keys: {missing[:5]}")
    if unexpected:
        print(f"WARN unexpected keys: {unexpected[:5]}")
    model.eval()
    print(f"Loaded model from: {CKPT_PATH}")

    test_df = pd.read_parquet(TEST_SPLIT, columns=["sgRNA_seq"])
    raw_guides = test_df["sgRNA_seq"].astype(str).unique().tolist()
    guides = sorted({g[:20].upper() for g in raw_guides if len(g) >= 20})
    print(f"Unique guides from test split: {len(guides)}")
    print()

    print("=== CCS (five rules on per-position / PAM interventions) ===")
    results = calculate_ccs_neural_v2(model, guides)
    for k, v in results.items():
        if k in ("method", "config"):
            continue
        if isinstance(v, float):
            print(f"  {k:30s} {v:.4f}")
        else:
            print(f"  {k:30s} {v}")
    print()
    print(f"Config: {results['config']}")

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
