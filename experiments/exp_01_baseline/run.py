from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dag.validate import validate_dag, empirical_sensitivity_profile
from evaluation.metrics import (
    EvalResult,
    evaluate_model,
    find_optimal_threshold,
    results_to_dataframe,
)
from models.baseline.catboost import CatBoostWrapper
from models.baseline.xgboost import XGBoostWrapper

log = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "xgboost": XGBoostWrapper,
    "catboost": CatBoostWrapper,
}


class BaselineModel(Protocol):
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "BaselineModel":
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def explain(self, X: np.ndarray) -> np.ndarray:
        ...

    def feature_importance(self) -> np.ndarray:
        ...

    def save(self, path: str | Path) -> None:
        ...


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve _base recursively (the chain can be > 1 level deep).
    while "_base" in cfg:
        base_path = ROOT / cfg.pop("_base")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        _deep_merge(base, cfg)
        cfg = base

    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ── Data ──────────────────────────────────────────────────────────────────────

def load_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(ROOT / cfg["data"]["train_split"])
    val   = pd.read_parquet(ROOT / cfg["data"]["val_split"])
    test  = pd.read_parquet(ROOT / cfg["data"]["test_split"])
    log.info("Split: train=%d | val=%d | test=%d", len(train), len(val), len(test))
    return train, val, test


def load_guideseq(cfg: dict) -> pd.DataFrame | None:
    path = ROOT / cfg["data"].get("guideseq_features", "")
    if not path.exists():
        log.warning("GUIDE-seq non trovato in %s — skip cross-assay.", path)
        return None
    df = pd.read_parquet(path)
    log.info("GUIDE-seq: %d righe, %d positivi", len(df), int(df["label"].sum()))
    return df


# ── Features ──────────────────────────────────────────────────────────────────

def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("Feature mancanti (ignorate): %s", missing)
    X = df[available].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    return X, y, available


# ── SHAP ──────────────────────────────────────────────────────────────────────

def save_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    path: Path,
) -> None:
    """Save per-instance predictions for downstream statistical testing.

    Output: parquet with [sgRNA_seq, off_seq, label, prob] for the test rows.
    Required for DeLong / paired bootstrap comparisons across models.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [c for c in ("sgRNA_seq", "off_seq", "label") if c in df.columns]
    out = df[keep_cols].reset_index(drop=True).copy()
    # predict_proba may return 2D (n, 2); keep positive-class probability
    probs_arr = np.asarray(probs)
    if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
        probs_arr = probs_arr[:, 1]
    out["prob"] = probs_arr.astype(np.float32).reshape(-1)
    out.to_parquet(path, index=False)
    log.info("Predictions saved: %s (%d rows)", path, len(out))


def save_shap(
    shap_vals:     np.ndarray,
    df:            pd.DataFrame,
    feature_names: list[str],
    path:          Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("shap_values",  data=shap_vals, compression="gzip")
        f.create_dataset("labels",       data=df["label"].to_numpy(dtype=np.float32))
        f.create_dataset("feature_names", data=np.array(feature_names, dtype="S"))
        if "guide_name" in df.columns:
            f.create_dataset("guide_names",
                             data=np.array(df["guide_name"].to_numpy(dtype="S"), dtype="S"))
        if "sgRNA_seq" in df.columns:
            f.create_dataset("sgrna_seqs",
                             data=np.array(df["sgRNA_seq"].to_numpy(dtype="S"), dtype="S"))
    log.info("SHAP salvati: %s  shape=%s", path, shap_vals.shape)


def _model_label(model_type: str) -> str:
    normalized = model_type.lower()
    if normalized == "xgboost":
        return "XGBoost"
    if normalized == "catboost":
        return "CatBoost"
    return model_type


def build_model(model_cfg: dict, feature_names: list[str]) -> tuple[str, type[Any]]:
    model_type = model_cfg["type"].lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_cfg['type']}")

    model_cls = MODEL_REGISTRY[model_type]
    return _model_label(model_type), model_cls


# ── Ablation ──────────────────────────────────────────────────────────────────

def run_ablation(
    model_name: str,
    model_cls: type[Any],
    cfg:         dict,
    model_cfg:   dict,
    train:       pd.DataFrame,
    val:         pd.DataFrame,
    test:        pd.DataFrame,
    results_dir: Path,
) -> list[dict]:
    if not cfg.get("ablation", {}).get("run", False):
        return []

    all_cols  = cfg["features"]["feature_cols"]
    rows: list[dict] = []

    for variant in cfg["ablation"]["variants"]:
        name = variant["name"]
        log.info("Ablation: %s", name)

        if variant.get("feature_cols") is not None:
            cols = variant["feature_cols"]
        elif variant.get("exclude_cols"):
            cols = [c for c in all_cols if c not in variant["exclude_cols"]]
        else:
            cols = all_cols

        X_tr, y_tr, used = prepare_xy(train, cols)
        X_v,  y_v,  _    = prepare_xy(val,   used)
        X_te, y_te, _    = prepare_xy(test,  used)

        m = model_cls(
            params=model_cfg.get("params"),
            early_stopping_rounds=model_cfg.get("early_stopping_rounds", 30),
            feature_names=used,
        )
        m.fit(X_tr, y_tr, X_v, y_v, feature_names=used)

        thr = find_optimal_threshold(
            y_v, m.predict_proba(X_v),
            metric=cfg["evaluation"]["threshold_metric"],
        )
        res = evaluate_model(
            f"{model_name}-{name}", y_te, m.predict_proba(X_te),
            split="within_test", threshold=thr,
            store_curves=False,
        )
        rows.append({**res.to_dict(), "variant": name})

    return rows


def run_model_pipeline(
    model_cfg: dict,
    cfg: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    guide_df: pd.DataFrame | None,
    feature_cols: list[str],
    results_dir: Path,
) -> tuple[list[EvalResult], list[dict]]:
    model_name, model_cls = build_model(model_cfg, feature_cols)
    log.info("Training model: %s", model_name)

    X_train, y_train, used_cols = prepare_xy(train, feature_cols)
    X_val, y_val, _ = prepare_xy(val, used_cols)
    X_test, y_test, _ = prepare_xy(test, used_cols)

    model: BaselineModel = model_cls(
        params=model_cfg.get("params"),
        early_stopping_rounds=model_cfg.get("early_stopping_rounds", 30),
        feature_names=used_cols,
    )

    model.fit(X_train, y_train, X_val, y_val, feature_names=used_cols)

    if cfg["logging"]["save_model"]:
        model.save(results_dir / f"{model_name.lower()}_model.pkl")

    optimal_thr = find_optimal_threshold(
        y_val,
        model.predict_proba(X_val),
        metric=cfg["evaluation"]["threshold_metric"],
    )

    p_test = model.predict_proba(X_test)
    result_within = evaluate_model(
        model_name,
        y_test,
        p_test,
        split="within_test",
        threshold=optimal_thr,
        store_curves=cfg["evaluation"]["store_curves"],
    )
    save_predictions(test, p_test, results_dir / f"predictions_{model_name.lower()}_test.parquet")

    all_results: list[EvalResult] = [result_within]
    X_guide: np.ndarray | None = None
    y_guide: np.ndarray | None = None

    if guide_df is not None:
        X_guide, y_guide, _ = prepare_xy(guide_df, used_cols)
        p_guide = model.predict_proba(X_guide)
        result_cross = evaluate_model(
            model_name,
            y_guide,
            p_guide,
            split="cross_assay",
            threshold=optimal_thr,
            store_curves=cfg["evaluation"]["store_curves"],
        )
        all_results.append(result_cross)
        save_predictions(guide_df, p_guide, results_dir / f"predictions_{model_name.lower()}_guideseq.parquet")

    if cfg["logging"]["save_shap"]:
        save_shap(
            model.explain(X_test),
            test,
            used_cols,
            results_dir / f"shap_values_{model_name.lower()}_test.h5",
        )
        if guide_df is not None:
            assert X_guide is not None
            save_shap(
                model.explain(X_guide),
                guide_df,
                used_cols,
                results_dir / f"shap_values_{model_name.lower()}_guide.h5",
            )

    fi = pd.DataFrame(
        {
            "feature": used_cols,
            "importance": model.feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    fi.to_csv(results_dir / f"feature_importance_{model_name.lower()}.csv", index=False)
    log.info("Top-5 feature (%s):\n%s", model_name, fi.head().to_string(index=False))

    ablation_rows = run_ablation(model_name, model_cls, cfg, model_cfg, train, val, test, results_dir)
    return all_results, ablation_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: Path) -> None:
    cfg = load_config(config_path)

    results_dir = ROOT / cfg["logging"]["results_dir"] / cfg["experiment"]["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", results_dir)

    # 1. Dati
    train, val, test = load_splits(cfg)
    guide_df         = load_guideseq(cfg) if cfg["evaluation"]["cross_assay"] else None
    feature_cols     = cfg["features"]["feature_cols"]

    _, _, used_cols = prepare_xy(train, feature_cols)
    log.info("Feature usate (%d): %s", len(used_cols), used_cols)

    # 2. Validazione DAG
    log.info("Validazione DAG...")
    dag_report = validate_dag(train)
    dag_report.to_csv(results_dir / "dag_validation.csv", index=False)

    bio_prior = empirical_sensitivity_profile(train)
    np.save(results_dir / "bio_prior_empirical.npy", bio_prior)
    log.info("Top-3 posizioni sensibili: %s", np.argsort(bio_prior)[::-1][:3] + 1)

    # 3. Training multi-modello
    model_specs = cfg.get("models") or [cfg["model"]]
    all_results: list[EvalResult] = []
    all_ablation_rows: list[dict] = []

    for model_cfg in model_specs:
        model_results, ablation_rows = run_model_pipeline(
            model_cfg,
            cfg,
            train,
            val,
            test,
            guide_df,
            used_cols,
            results_dir,
        )
        all_results.extend(model_results)
        all_ablation_rows.extend(ablation_rows)

    if all_ablation_rows:
        pd.DataFrame(all_ablation_rows).to_csv(
            results_dir / "ablation_results.csv", index=False
        )

    # 4. Risultati finali
    metrics_df = results_to_dataframe(all_results)
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    log.info("\nRisultati finali:\n%s", metrics_df.to_string(index=False))
    log.info("Esperimento completato: %s", results_dir)


def cli_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli_main()