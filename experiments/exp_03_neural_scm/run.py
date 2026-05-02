from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.ccs import calculate_ccs_neural
from models.deep import NeuralSCM, train
from models.deep.train import evaluate
from models.deep.encoding import PairwiseTokenEncoder, BiologicalMismatchEncoder

from models.utils.tracking import ExperimentTracker

log = logging.getLogger(__name__)


class _CRISPRDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self._sgrnas = df["sgRNA_seq"].astype(str).tolist()
        self._off_targets = df["off_seq"].astype(str).tolist()
        self._labels = df["label"].astype(float).tolist()

    def __len__(self) -> int:
        return len(self._sgrnas)

    def __getitem__(self, idx: int) -> dict:
        return {
            "sgrna": self._sgrnas[idx],
            "off_target": self._off_targets[idx],
            "label": self._labels[idx],
        }


def _collate(batch: list[dict]) -> dict:
    """
    Costruisce il batch a dizionario e genera mutazioni causali on-the-fly
    per addestrare il Neural SCM a rispettare la topologia.
    """
    sgrnas = [b["sgrna"] for b in batch]
    off_targets = [b["off_target"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

    out = {
        "sgrnas": sgrnas,
        "off_targets": off_targets,
        "labels": labels,
    }

    # Causal Representation Learning: Generazione varianti On-The-Fly
    # Inseriamo un rateo del 50% per avere batch misti (efficienza computazionale)
    if random.random() < 0.5:
        mut_offs = []
        masks_prox = []
        masks_seed = []
        masks_nonseed = []
        exp_dirs = []

        for sgrna, off in zip(sgrnas, off_targets):
            # Esempio base: Iniezione di un mismatch nella regione Seed (pos 8-16)
            # Biologicamente, questo DEVE abbassare l'efficienza (expected_direction = -1)
            # E NON deve alterare i moduli Proximal e NonSeed (masks = 1)
            if len(off) >= 20:
                # Alteriamo la base in posizione 10 (pieno seed)
                mut_base = "A" if off[10] != "A" else "C"
                mut_off = off[:10] + mut_base + off[11:]
                
                mut_offs.append(mut_off)
                masks_prox.append(1)    # Intatto
                masks_seed.append(0)    # Alterato (ignorare consistenza qui)
                masks_nonseed.append(1) # Intatto
                exp_dirs.append(-1.0)   # Il mismatch peggiora il taglio
            else:
                # Fallback di sicurezza per sequenze anomale (neutrale)
                mut_offs.append(off)
                masks_prox.append(1); masks_seed.append(1); masks_nonseed.append(1)
                exp_dirs.append(0.0)

        out["sgrnas_mut"] = sgrnas
        out["off_targets_mut"] = mut_offs
        out["unaltered_masks"] = {
            "proximal": torch.tensor(masks_prox, dtype=torch.bool),
            "seed": torch.tensor(masks_seed, dtype=torch.bool),
            "nonseed": torch.tensor(masks_nonseed, dtype=torch.bool),
        }
        out["expected_direction"] = torch.tensor(exp_dirs, dtype=torch.float32)

    return out


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cur = cfg
    while isinstance(cur, dict) and "_base" in cur:
        base_rel = cur.pop("_base")
        base_path = ROOT / base_rel
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        with base_path.open(encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        _deep_merge(base, cur)
        cur = base

    return cur


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _load_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    data_cfg = cfg.get("data", {})
    splits = {
        "train": pd.read_parquet(ROOT / data_cfg["train_split"]),
        "val": pd.read_parquet(ROOT / data_cfg["val_split"]),
        "test": pd.read_parquet(ROOT / data_cfg["test_split"]),
    }
    for name, df in splits.items():
        log.info("Split %s: rows=%d positives=%d", name, len(df), int(df["label"].sum()))
    return splits


def _sample_df(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _safe_float_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in payload.items()
    }


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        _CRISPRDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        drop_last=False,
    )


def main(config_path: Path) -> None:
    cfg = load_config(config_path)

    logging_cfg = cfg.get("logging") or {}
    results_root = logging_cfg.get("results_dir", "experiments/results/")
    results_dir = (
        Path(results_root) if Path(results_root).is_absolute() else ROOT / results_root
    ) / cfg.get("experiment", {}).get("name", "exp_03_neural_scm")
    results_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", results_dir)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load CHANGE-seq splits
    splits = _load_splits(cfg)
    phase3_cfg = cfg.get("phase3", {})
    fit_split = str(phase3_cfg.get("fit_split", "train"))
    eval_split = str(phase3_cfg.get("evaluation_split", "test"))
    
    if fit_split not in splits or eval_split not in splits:
        raise ValueError(f"Invalid split selection fit={fit_split}, eval={eval_split}")

    df_train = _sample_df(splits[fit_split], phase3_cfg.get("max_fit_rows"), seed)
    df_val = splits["val"]
    df_test = _sample_df(splits[eval_split], phase3_cfg.get("max_eval_rows"), seed)

    log.info(
        "Effective rows — train: %d, val: %d, test: %d",
        len(df_train), len(df_val), len(df_test),
    )

    # 2. Initialise Hardware and NeuralSCM from config
    training_cfg = cfg.get("training", {})

    # Calcolo dinamico di pos_weight dal training set (evita sbilanciamento manuale)
    try:
        n_neg = int((df_train["label"] == 0).sum())
        n_pos = int((df_train["label"] == 1).sum())
        if n_pos == 0:
            log.warning("No positive examples in training split; using pos_weight=1.0")
            training_cfg["pos_weight"] = 1.0
        else:
            training_cfg["pos_weight"] = float(n_neg) / float(n_pos)
            log.info("pos_weight: %.1f", training_cfg["pos_weight"])
    except Exception as exc:  # fallback sicuro
        log.warning("Could not compute pos_weight dynamically: %s; defaulting to config or 1.0", exc)
        training_cfg["pos_weight"] = float(training_cfg.get("pos_weight", 1.0))

    device = torch.device(training_cfg.get("device", "cpu"))
    log.info("Using device: %s", device)

    model_cfg = cfg.get("model", {})
    
    # Istanzia l'encoder scelto dal config
    encoder_type = str(model_cfg.get("encoder", "pairwise")).lower()
    if encoder_type == "biological_mismatch":
        encoder = BiologicalMismatchEncoder()
        log.info("Using BiologicalMismatchEncoder (embed_dim=12)")
    else:  # default: pairwise
        encoder = PairwiseTokenEncoder(embed_dim=int(model_cfg.get("embed_dim", 16)))
        log.info("Using PairwiseTokenEncoder (embed_dim=%d)", encoder.embed_dim)
    
    model = NeuralSCM(
        embed_dim=int(model_cfg.get("embed_dim", 16)),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        encoder=encoder,
    ).to(device)

    # 3. Train and save best model
    batch_size = int(training_cfg.get("batch_size", 64))
    
    train_loader = _make_loader(df_train, batch_size, shuffle=True)
    val_loader = _make_loader(df_val, batch_size, shuffle=False)
    
    # Optional: Initialize experiment tracker (e.g., Weights & Biases) se configurato
    use_tracking = cfg.get("tracking", {}).get("enabled", False)
    tracker = ExperimentTracker(config=cfg, enabled=use_tracking)

    trained_model = train(model, train_loader, val_loader, training_cfg, tracker=tracker)

    model_path = results_dir / str(cfg.get("output", {}).get("model_pt", "neural_scm.pt"))
    torch.save(trained_model.state_dict(), model_path)
    log.info("Model saved: %s", model_path)

    # Metriche su train/val/test, usando loader deterministici senza shuffle.
    train_eval_loader = _make_loader(df_train, batch_size, shuffle=False)
    train_metrics = evaluate(trained_model, train_eval_loader, device)
    log.info("CHANGE-seq train: %s", train_metrics)
    _save_json(
        {"split": "changeseq_train", **_safe_float_dict(train_metrics)},
        results_dir / str(cfg.get("output", {}).get("metrics_changeseq_train_json", "metrics_changeseq_train.json")),
    )

    val_loader = _make_loader(df_val, batch_size, shuffle=False)
    val_metrics = evaluate(trained_model, val_loader, device)
    log.info("CHANGE-seq val: %s", val_metrics)
    _save_json(
        {"split": "changeseq_val", **_safe_float_dict(val_metrics)},
        results_dir / str(cfg.get("output", {}).get("metrics_changeseq_val_json", "metrics_changeseq_val.json")),
    )

    # 4. Evaluate within-dataset (CHANGE-seq test)
    test_loader = _make_loader(df_test, batch_size, shuffle=False)
    test_metrics = evaluate(trained_model, test_loader, device)
    log.info("CHANGE-seq test: %s", test_metrics)
    _save_json(
        {"split": "changeseq_test", **_safe_float_dict(test_metrics)},
        results_dir / str(cfg.get("output", {}).get("metrics_changeseq_json", "metrics_changeseq.json")),
    )
    if tracker is not None:
        tracker.log_metrics({f"changeseq_{k}": v for k, v in test_metrics.items()})

    # 5. Evaluate cross-assay (GUIDE-seq)
    guideseq_path = ROOT / cfg.get("data", {}).get("guideseq_features", "")
    out_guideseq = results_dir / str(cfg.get("output", {}).get("metrics_guideseq_json", "metrics_guideseq.json"))
    if guideseq_path.exists():
        df_guide = pd.read_parquet(guideseq_path)
        guide_loader = _make_loader(df_guide, batch_size, shuffle=False)
        guide_metrics = evaluate(trained_model, guide_loader, device)
        log.info("GUIDE-seq cross-assay: %s", guide_metrics)
        _save_json(
            {"split": "guideseq", **_safe_float_dict(guide_metrics)},
            out_guideseq,
        )
        if tracker is not None:
            tracker.log_metrics({f"guideseq_{k}": v for k, v in guide_metrics.items()})
    else:
        log.warning("GUIDE-seq features not found at %s; skipping cross-assay", guideseq_path)
        _save_json({"status": "skipped", "reason": f"not found: {guideseq_path}"}, out_guideseq)

    # 6. CCS with native do() — compare with XGBoost baseline
    ccs_cfg = cfg.get("ccs", {})
    if ccs_cfg.get("enabled", True):
        guide_source = str(ccs_cfg.get("guide_source_split", eval_split))
        if guide_source not in splits:
            raise ValueError(f"Invalid ccs.guide_source_split: {guide_source}")

        guides = sorted(splits[guide_source]["sgRNA_seq"].astype(str).unique().tolist())
        max_guides = int(ccs_cfg.get("max_guides", 0))
        if max_guides > 0 and len(guides) > max_guides:
            guides = guides[:max_guides]

        trained_model.eval()
        ccs_neural = calculate_ccs_neural(
            trained_model, guides, mode=ccs_cfg.get("mode", "6_rules")
        )
        log.info("Neural CCS_Overall: %.4f", ccs_neural.get("CCS_Overall", float("nan")))
        _save_json(
            ccs_neural,
            results_dir / str(cfg.get("output", {}).get("ccs_neural_json", "ccs_neural.json")),
        )

        baseline_ccs_path_str = cfg.get("baseline", {}).get("ccs_json", "")
        baseline_ccs_path = ROOT / baseline_ccs_path_str if baseline_ccs_path_str else Path("")
        out_comparison = results_dir / str(cfg.get("output", {}).get("ccs_comparison_json", "ccs_comparison.json"))
        
        if baseline_ccs_path.exists():
            baseline_payload = json.loads(baseline_ccs_path.read_text(encoding="utf-8"))
            baseline_overall = float(
                baseline_payload.get("metrics", {}).get("CCS_Overall", float("nan"))
            )
            neural_overall = float(ccs_neural.get("CCS_Overall", float("nan")))
            comparison = {
                "neural_ccs": ccs_neural,
                "baseline_ccs_overall": baseline_overall,
                "delta_ccs_overall": round(neural_overall - baseline_overall, 4)
                if np.isfinite(baseline_overall)
                else None,
            }
            _save_json(comparison, out_comparison)
        else:
            log.warning("Baseline CCS JSON not found: %s", baseline_ccs_path)
            _save_json(
                {"neural_ccs": ccs_neural, "baseline_ccs_overall": None, "delta_ccs_overall": None},
                out_comparison,
            )

    log.info("Phase 3 experiment completed: %s", results_dir)


def cli_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli_main()