from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dag.independence_tests import validate_dag_implications
from dag.do_calculus import (
    compare_observational_vs_interventional,
    build_intervention_dataset,
    ALTERNATE_DAG_MISMATCH_LABEL,
)
from dag.scm import CRISPRCausalModel
from dag.nodes import CRISPRPairFeatures
from evaluation.ccs import calculate_ccs
from models.baseline.xgboost import XGBoostWrapper
from models.baseline.catboost import CatBoostWrapper

log = logging.getLogger(__name__)

MODEL_LOADERS: dict[str, Callable[[Path], Any]] = {
    "xgboost": XGBoostWrapper.load,
    "catboost": CatBoostWrapper.load,
}


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve chained `_base` entries recursively so configs can inherit multi-level
    # (e.g. config.yaml -> experiments/configs/base.yaml).
    cur = cfg
    while isinstance(cur, dict) and "_base" in cur:
        base_rel = cur.pop("_base")
        base_path = ROOT / base_rel
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        with base_path.open(encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        # Merge current overrides into base
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
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _failure_rate(independence_df: pd.DataFrame) -> float:
    if independence_df.empty:
        return float("nan")

    if "status" in independence_df.columns:
        ok = independence_df[independence_df["status"] == "ok"]
    else:
        ok = independence_df
    if ok.empty:
        return float("nan")

    if "reject_h0_fdr" in ok.columns:
        return float(ok["reject_h0_fdr"].mean())
    if "reject_h0_fwer" in ok.columns:
        return float(ok["reject_h0_fwer"].mean())
    if "reject_h0" in ok.columns:
        return float(ok["reject_h0"].mean())
    return float("nan")


def _build_predict_fn(model: Any, feature_cols: list[str]) -> Callable[[list[str], list[str]], np.ndarray]:
    def predict_fn(sgrnas: list[str], off_seqs: list[str]) -> np.ndarray:
        rows: list[dict[str, float | int | str]] = []
        for sg, off in zip(sgrnas, off_seqs):
            pair = CRISPRPairFeatures(sgRNA_seq=str(sg), off_seq=str(off))
            row = pair.to_feature_dict()
            row.update(pair.to_concept_dict())
            rows.append(row)

        df = pd.DataFrame(rows)
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            raise ValueError("No configured feature columns available for baseline prediction")
        x = df[available].to_numpy(dtype=np.float32)
        proba = np.asarray(model.predict_proba(x), dtype=float)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    return predict_fn


def _compute_baseline_ccs(cfg: dict, split_df: pd.DataFrame, results_dir: Path) -> None:
    ccs_cfg = cfg.get("ccs", {})
    if not ccs_cfg.get("enabled", True):
        log.info("CCS baseline skipped by config")
        return

    baseline_cfg = cfg.get("baseline", {})
    model_type = str(baseline_cfg.get("model_type", "xgboost")).lower()
    model_path = ROOT / str(baseline_cfg.get("model_path", ""))

    output_name = cfg.get("output", {}).get("ccs_baseline_json", "ccs_baseline.json")
    out_path = results_dir / output_name

    if model_type not in MODEL_LOADERS:
        payload = {
            "status": "skipped",
            "reason": f"unsupported model_type: {model_type}",
        }
        _save_json(payload, out_path)
        return

    if not model_path.exists():
        payload = {
            "status": "skipped",
            "reason": f"baseline model not found: {model_path}",
        }
        _save_json(payload, out_path)
        return

    model = MODEL_LOADERS[model_type](model_path)
    feature_cols = list(cfg.get("features", {}).get("feature_cols", []))
    if not feature_cols:
        raise ValueError("features.feature_cols must be configured for CCS baseline")

    guides = sorted(split_df["sgRNA_seq"].astype(str).unique().tolist())
    max_guides = int(ccs_cfg.get("max_guides", 0))
    if max_guides > 0 and len(guides) > max_guides:
        guides = guides[:max_guides]
    predict_fn = _build_predict_fn(model, feature_cols)

    metrics = calculate_ccs(guides, predict_fn, mode=ccs_cfg.get("mode", "6_rules"))
    payload = {
        "status": "ok",
        "model_type": model_type,
        "model_path": str(model_path),
        "guide_count": len(guides),
        "mode": ccs_cfg.get("mode", "6_rules"),
        "metrics": _safe_float_dict(metrics),
    }
    _save_json(payload, out_path)


def _run_interventional_queries(
    scm: CRISPRCausalModel,
    df_eval: pd.DataFrame,
    cfg: dict,
    results_dir: Path,
) -> None:
    queries = cfg.get("interventions", {}).get("queries", [])
    rows: list[pd.DataFrame] = []

    for query in queries:
        name = str(query.get("name", "query"))
        intervention = query.get("intervention", {})
        if not intervention:
            log.warning("Skipping empty intervention query: %s", name)
            continue

        table = compare_observational_vs_interventional(scm, df_eval, intervention)
        table.insert(0, "query_name", name)
        rows.append(table)

    output_name = cfg.get("output", {}).get("observational_vs_do_csv", "observational_vs_do.csv")
    out_path = results_dir / output_name

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame(columns=["query_name", "treatment", "value", "p_observational", "p_do", "delta_do_minus_obs", "n_observational"]).to_csv(out_path, index=False)


def _run_intervention_dataset(
    scm: CRISPRCausalModel,
    df_eval: pd.DataFrame,
    cfg: dict,
    results_dir: Path,
) -> None:
    specs = cfg.get("interventions", {}).get("dataset", [])
    if not specs:
        raise ValueError("interventions.dataset cannot be empty")

    synth = build_intervention_dataset(df_eval, specs, scm=scm)
    output_name = cfg.get("output", {}).get("intervention_dataset_parquet", "intervention_dataset.parquet")
    out_path = results_dir / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth.to_parquet(out_path, index=False)


def main(config_path: Path) -> None:
    cfg = load_config(config_path)

    logging_cfg = cfg.get("logging") or {}
    results_root = logging_cfg.get("results_dir", "experiments/results/")
    results_dir = Path(results_root) if Path(results_root).is_absolute() else ROOT / results_root
    results_dir = results_dir / cfg.get("experiment", {}).get("name", "exp_02_scm")
    results_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", results_dir)

    splits = _load_splits(cfg)
    fit_split = str(cfg.get("phase2", {}).get("fit_split", "train"))
    eval_split = str(cfg.get("phase2", {}).get("evaluation_split", "test"))
    if fit_split not in splits or eval_split not in splits:
        raise ValueError(f"Invalid split selection fit={fit_split}, eval={eval_split}")

    seed = int(cfg.get("seed", 42))
    phase2_cfg = cfg.get("phase2", {})
    indep_cfg = cfg.get("independence", {})

    df_fit = _sample_df(splits[fit_split], phase2_cfg.get("max_fit_rows"), seed)
    df_eval = _sample_df(splits[eval_split], phase2_cfg.get("max_eval_rows"), seed)
    df_indep = _sample_df(df_fit, indep_cfg.get("max_rows"), seed)

    log.info("Effective fit rows: %d", len(df_fit))
    log.info("Effective eval rows: %d", len(df_eval))
    log.info("Effective CI rows: %d", len(df_indep))

    # 1) Independence tests (DAG misspecification gate)
    alpha = float(indep_cfg.get("alpha", 0.05))
    dag_variant_name = str(phase2_cfg.get("dag_variant", "default") or "default")
    if dag_variant_name == "mismatch_label":
        dag_spec = ALTERNATE_DAG_MISMATCH_LABEL
    else:
        dag_spec = None

    dag_report = validate_dag_implications(dag_spec, df_indep, alpha=alpha)

    independence_csv = results_dir / cfg.get("output", {}).get("independence_csv", "dag_independence_tests.csv")
    dag_report.to_csv(independence_csv, index=False)

    failure_rate = _failure_rate(dag_report)
    max_failure_rate = float(indep_cfg.get("max_failure_rate", 0.30))
    if np.isfinite(failure_rate):
        log.info("CI failure rate: %.3f", failure_rate)
        if failure_rate > max_failure_rate:
            log.warning(
                "CI failure rate %.3f exceeds threshold %.3f; DAG revision recommended before continuing",
                failure_rate,
                max_failure_rate,
            )

    # 2) Fit SCM
    scm = CRISPRCausalModel().fit(df_fit)
    params = _safe_float_dict(scm.parameters())
    params["fit_split"] = fit_split
    params["evaluation_split"] = eval_split
    params["ci_failure_rate"] = failure_rate

    params_json = results_dir / cfg.get("output", {}).get("scm_parameters_json", "scm_parameters.json")
    _save_json(params, params_json)

    # 3) Observational vs interventional queries
    _run_interventional_queries(scm, df_eval, cfg, results_dir)

    # 4) CCS baseline (if baseline model is available)
    ccs_split = str(cfg.get("ccs", {}).get("guide_source_split", eval_split))
    if ccs_split not in splits:
        raise ValueError(f"Invalid ccs.guide_source_split: {ccs_split}")
    _compute_baseline_ccs(cfg, splits[ccs_split], results_dir)

    # 5) Build intervention dataset for next phase
    _run_intervention_dataset(scm, df_eval, cfg, results_dir)

    log.info("Phase 2 experiment completed: %s", results_dir)


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
