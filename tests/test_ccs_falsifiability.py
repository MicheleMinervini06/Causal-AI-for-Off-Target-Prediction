"""Falsifiability check for the Causal Consistency Score (CCS).

Runs the five rules on a panel of post-audit model variants to verify that
the score discriminates between models that respect the encoded biophysical
invariances and those that do not. The adopted model is Exp30; the panel
spans the ablation choices reported in Section 4.3 of the results chapter
(lambda_causal sweep, monotonicity prior, context features, spacer
encoder, architectural backbone).

Output: experiments/results/ccs_falsifiability.json
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
from models.deep.encoding import (
    BaseEncoder,
    BiologicalMismatchEncoder,
    ContextAwareMismatchEncoder,
    PairwiseTokenEncoder,
)


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


def _build_encoder(model_cfg: dict) -> BaseEncoder:
    """Select the encoder class based on the config's `encoder` field."""
    name = str(model_cfg.get("encoder", "biological_mismatch")).lower()
    embed_dim = int(model_cfg.get("embed_dim", 16))
    if name == "biological_mismatch":
        return BiologicalMismatchEncoder()
    if name == "pairwise":
        return PairwiseTokenEncoder(embed_dim=embed_dim)
    if name == "context_aware":
        return ContextAwareMismatchEncoder(embed_dim=8)
    raise ValueError(f"Unknown encoder: {name}")


def _build_model_from_cfg(cfg: dict) -> NeuralSCM:
    model_cfg = cfg.get("model", {})
    encoder = _build_encoder(model_cfg)
    return NeuralSCM(
        architecture=str(model_cfg.get("architecture", "positional_mlp")),
        embed_dim=int(model_cfg.get("embed_dim", 16)),
        hidden_dim=int(model_cfg.get("hidden_dim", 8)),
        encoder=encoder,
        context_dim=len(model_cfg.get("context_cols", [])),
        variational=bool(model_cfg.get("variational", False)),
        pam_mode=str(model_cfg.get("pam_mode", "additive")),
        positional_use_encoder=bool(model_cfg.get("positional_use_encoder", False)),
    )


def _load_model(cfg: dict, ckpt_path: Path) -> NeuralSCM:
    model = _build_model_from_cfg(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  (load: missing={len(missing)} unexpected={len(unexpected)})")
    model.eval()
    return model


def _get_guides(split_path: Path) -> list[str]:
    df = pd.read_parquet(split_path, columns=["sgRNA_seq"])
    raw = df["sgRNA_seq"].astype(str).unique().tolist()
    return sorted({g[:20].upper() for g in raw if len(g) >= 20})


# Reference split used to source the held-out guides for every variant.
REF_SPLIT = "data/processed/splits_merged/test.parquet"

CANDIDATES = [
    # Adopted reference
    {
        "label": "Adopted (Exp30, biological encoder, additive PAM, lambda=0.10)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp30_ablation_encoder_biological_mismatch_use_encoder.yaml",
        "ckpt":  "experiments/results/Exp30_Ablation_EncoderBiologicalMismatch_UseEncoder/neural_scm.pt",
    },
    # lambda_causal sweep (Section 4.3.5)
    {
        "label": "lambda_causal = 0.00 (Exp23a)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp23a_causal_0.yaml",
        "ckpt":  "experiments/results/Exp23a_Causal_0/neural_scm.pt",
        "split": "data/processed/splits/test.parquet",
    },
    {
        "label": "lambda_causal = 0.10 (Exp23b, single-split reference)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp23b_causal_01.yaml",
        "ckpt":  "experiments/results/Exp23b_Causal_0p1/neural_scm.pt",
        "split": "data/processed/splits/test.parquet",
    },
    {
        "label": "lambda_causal = 0.30 (Exp23c)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp23c_causal_03.yaml",
        "ckpt":  "experiments/results/Exp23c_Causal_0p3/neural_scm.pt",
        "split": "data/processed/splits/test.parquet",
    },
    {
        "label": "lambda_causal = 1.00 (Exp23d, collapsed training)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp23d_causal_1.yaml",
        "ckpt":  "experiments/results/Exp23d_Causal_1p0/neural_scm.pt",
        "split": "data/processed/splits/test.parquet",
    },
    # Encoder ablation (Section 4.3.4)
    {
        "label": "Encoder: mismatch type only (Exp24, 4-dim internal)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp24_merged_split.yaml",
        "ckpt":  "experiments/results/Exp24_MergedSplit_Causal_0p1/neural_scm.pt",
    },
    {
        "label": "Encoder: pairwise learned (Exp28, 16-dim)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp28_ablation_encoder_pairwise.yaml",
        "ckpt":  "experiments/results/Exp28_Ablation_EncoderPairwise/neural_scm.pt",
    },
    {
        "label": "Encoder: context-aware (Exp29, 8-dim, failed training)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp29_ablation_encoder_context_aware.yaml",
        "ckpt":  "experiments/results/Exp29_Ablation_EncoderContextAware/neural_scm.pt",
    },
    # Architecture / prior / context features ablations (Sections 4.3.1-4.3.3)
    {
        "label": "Architecture: regional typed MLP (Exp25)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp25_ablation_typed_mlp.yaml",
        "ckpt":  "experiments/results/Exp25_Ablation_TypedMLP/neural_scm.pt",
    },
    {
        "label": "No monotonicity prior (Exp26, unconstrained w_pos)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp26_ablation_no_monotonicity.yaml",
        "ckpt":  "experiments/results/Exp26_Ablation_NoMonotonicity/neural_scm.pt",
    },
    {
        "label": "No context features (Exp27, g_ctx = 0)",
        "cfg":   "experiments/exp_03_neural_scm/config_exp27_ablation_no_context.yaml",
        "ckpt":  "experiments/results/Exp27_Ablation_NoContext/neural_scm.pt",
    },
]


def main() -> None:
    results = []
    for entry in CANDIDATES:
        print(f"\n=== {entry['label']} ===")
        ckpt_path = ROOT / entry["ckpt"]
        split_path = ROOT / entry.get("split", REF_SPLIT)
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found: {ckpt_path}")
            continue
        if not split_path.exists():
            print(f"  SKIP: split not found: {split_path}")
            continue
        if "inline_cfg" in entry:
            cfg = entry["inline_cfg"]
        else:
            cfg_path = ROOT / entry["cfg"]
            if not cfg_path.exists():
                print(f"  SKIP: config not found: {cfg_path}")
                continue
            cfg = _load_config_with_base(cfg_path)
        try:
            model = _load_model(cfg, ckpt_path)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue
        guides = _get_guides(split_path)
        if not guides:
            print(f"  SKIP: no guides")
            continue
        ccs = calculate_ccs_neural_v2(model, guides)
        print(f"  guides:       {len(guides)}")
        print(f"  R1 PAM Abl:   {ccs['R1_PAM_Ablation']:.4f}")
        print(f"  R2 PAM Hier:  {ccs['R2_PAM_Hierarchy']:.4f}")
        print(f"  R3 Seed/Dist: {ccs['R3_Seed_vs_Distal']:.4f}")
        print(f"  R4 PosGrad:   {ccs['R4_Position_Gradient']:.4f}")
        print(f"  R5 Heal:      {ccs['R5_Heal_Mismatch']:.4f}")
        print(f"  CCS Overall:  {ccs['CCS_Overall']:.4f}")
        results.append({"label": entry["label"], **ccs})

    print("\n" + "=" * 95)
    print(f"{'Model':<60} {'R1':>5} {'R2':>5} {'R3':>5} {'R4':>5} {'R5':>5} {'CCS':>6}")
    print("=" * 95)
    for r in results:
        print(f"{r['label']:<60} "
              f"{r['R1_PAM_Ablation']:>5.2f} {r['R2_PAM_Hierarchy']:>5.2f} "
              f"{r['R3_Seed_vs_Distal']:>5.2f} {r['R4_Position_Gradient']:>5.2f} "
              f"{r['R5_Heal_Mismatch']:>5.2f} {r['CCS_Overall']:>6.3f}")

    out = ROOT / "experiments/results/ccs_falsifiability.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
