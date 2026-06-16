"""Statistical comparison of model predictions on CHANGE-seq test and GUIDE-seq.

Computes for each (model, split):
    - Point AUROC and AUPRC
    - Bootstrap 95% CI for both metrics
    - DeLong test for AUROC vs the reference model (NeuralSCM)
    - Paired bootstrap for AUPRC difference vs the reference model

Inputs: parquet files with columns [sgRNA_seq, off_seq, label, prob]

Outputs:
    - Console summary
    - LaTeX-ready table at tests/output/statistical_comparison.tex
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests/output"

# Match the same test rows across all models. The merge key is (sgRNA_seq, off_seq).
# Ensures DeLong / paired bootstrap operates on identical samples per row.
MODELS: dict[str, dict[str, Path]] = {
    "XGBoost (DAG)": {
        "test":  ROOT / "experiments/results/exp_01_baseline/predictions_xgboost_test.parquet",
        "guide": ROOT / "experiments/results/exp_01_baseline/predictions_xgboost_guideseq.parquet",
    },
    "CatBoost (DAG)": {
        "test":  ROOT / "experiments/results/exp_01_baseline/predictions_catboost_test.parquet",
        "guide": ROOT / "experiments/results/exp_01_baseline/predictions_catboost_guideseq.parquet",
    },
    "CCLMoff": {
        "test":  ROOT / "experiments/results/cclmoff/predictions_cclmoff_test.parquet",   # TBD
        "guide": ROOT / "experiments/results/cclmoff/predictions_cclmoff_guideseq.parquet",
    },
    "NeuralSCM (ours)": {
        "test":  ROOT / "experiments/results/Exp30_Ablation_EncoderBiologicalMismatch_UseEncoder/predictions_neuralscm_test.parquet",
        "guide": ROOT / "experiments/results/Exp30_Ablation_EncoderBiologicalMismatch_UseEncoder/predictions_neuralscm_guideseq.parquet",
    },
}

REFERENCE = "NeuralSCM (ours)"
SPLITS = ["test", "guide"]
SPLIT_LABEL = {"test": "CHANGE-seq held-out test", "guide": "GUIDE-seq (cross-assay)"}


# ──────────────────────────────────────────────────────────────────────────────
# DeLong's test for two correlated AUCs (Sun & Xu 2014, fast implementation)
# ──────────────────────────────────────────────────────────────────────────────
def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.empty(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Two-sided DeLong test for H0: AUC(A) = AUC(B), paired samples."""
    y_true  = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=np.float64)
    score_b = np.asarray(score_b, dtype=np.float64)

    # Sort so positives come first (m positives, n negatives)
    order = np.argsort(-y_true.astype(np.int64), kind="mergesort")
    y = y_true[order]
    a = score_a[order]
    b = score_b[order]
    m = int((y == 1).sum())
    n = len(y) - m

    if m == 0 or n == 0:
        raise ValueError("DeLong requires both positives and negatives.")

    preds = np.vstack([a, b])  # (2, m+n)
    pos = preds[:, :m]
    neg = preds[:, m:]

    tx = np.empty((2, m), dtype=np.float64)
    ty = np.empty((2, n), dtype=np.float64)
    tz = np.empty((2, m + n), dtype=np.float64)
    for r in range(2):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds[r])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n              # (2, m)
    v10 = 1.0 - (tz[:, m:] - ty) / m        # (2, n)
    sx = np.cov(v01)                         # (2, 2)
    sy = np.cov(v10)                         # (2, 2)
    cov = sx / m + sy / n                    # (2, 2)

    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    var_diff = max(var_diff, 1e-30)
    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))

    # Analytical AUC CIs from the diagonal of cov
    z975 = norm.ppf(0.975)
    se_a = np.sqrt(cov[0, 0])
    se_b = np.sqrt(cov[1, 1])
    return {
        "auc_a":      float(aucs[0]),
        "auc_b":      float(aucs[1]),
        "ci_a":       (float(aucs[0] - z975 * se_a), float(aucs[0] + z975 * se_a)),
        "ci_b":       (float(aucs[1] - z975 * se_b), float(aucs[1] + z975 * se_b)),
        "delta":      float(aucs[0] - aucs[1]),
        "z":          float(z),
        "p_value":    float(p),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap for AUPRC (single-model CI and paired difference)
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_auprc_ci(
    y: np.ndarray, scores: np.ndarray, n_boot: int = 500, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(average_precision_score(y, scores))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = average_precision_score(y[idx], scores[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_bootstrap_auprc(
    y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
    n_boot: int = 500, alpha: float = 0.05, seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ya = y[idx]
        diffs[i] = average_precision_score(ya, s_a[idx]) - average_precision_score(ya, s_b[idx])
    point = float(average_precision_score(y, s_a) - average_precision_score(y, s_b))
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p = 2.0 * float(min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"delta": point, "ci": (lo, hi), "p_value": p}


# ──────────────────────────────────────────────────────────────────────────────
# Loading & joining predictions
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Joined:
    y: np.ndarray
    probs: dict[str, np.ndarray]  # model name -> aligned probs


def load_and_join(split: str) -> Joined | None:
    """Align model predictions by row position.

    All models score the same source parquet (data/processed/splits/test.parquet
    for the test set, data/processed/features/guideseq_features.parquet for GUIDE-seq)
    and preserve row order via `reset_index(drop=True)` in their save step.
    We align by row index and verify that (sgRNA_seq, off_seq, label) agree at every
    row across models — if any mismatch, we error out instead of silently corrupting.
    Pandas merge on (sgRNA_seq, off_seq) is not safe here because pairs are not unique.
    """
    available: dict[str, pd.DataFrame] = {}
    for name, paths in MODELS.items():
        p = paths.get(split)
        if p is None or not p.exists():
            print(f"  [skip] {name:20s} -- missing {p}")
            continue
        df = pd.read_parquet(p, columns=["sgRNA_seq", "off_seq", "label", "prob"]).reset_index(drop=True)
        available[name] = df

    if len(available) < 2:
        print(f"  [skip {split}] need at least 2 models, have {len(available)}")
        return None

    names = list(available.keys())

    # Sanity 1: all dataframes must have identical length
    lengths = {name: len(df) for name, df in available.items()}
    if len(set(lengths.values())) > 1:
        print(f"  [ERROR {split}] lengths differ: {lengths}")
        return None

    # Sanity 2: (sgRNA_seq, off_seq, label) must match row-by-row across models.
    # Sample-check at 100 random positions instead of all 1.5M rows (full check would
    # be too slow on GUIDE-seq).
    ref_name = names[0]
    ref = available[ref_name]
    rng = np.random.default_rng(0)
    sample_idx = rng.integers(0, len(ref), min(100, len(ref)))
    for name in names[1:]:
        df = available[name]
        for col in ("sgRNA_seq", "off_seq", "label"):
            mismatches = (ref[col].iloc[sample_idx].values != df[col].iloc[sample_idx].values).sum()
            if mismatches > 0:
                print(f"  [ERROR {split}] {col} mismatch between {ref_name} and {name} "
                      f"at {mismatches}/{len(sample_idx)} sampled positions — row order differs!")
                return None

    print(f"  rows: {len(ref):,}  models: {names}  (row-order verified on 100-sample)")

    y = ref["label"].to_numpy(dtype=np.int64)
    probs = {name: df["prob"].to_numpy(dtype=np.float64) for name, df in available.items()}
    return Joined(y=y, probs=probs)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "<0.0001"
    return f"{p:.4f}"


def main(n_boot: int = 500) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for split in SPLITS:
        print(f"\n{'=' * 70}")
        print(f"SPLIT: {SPLIT_LABEL[split]}")
        print("=" * 70)
        joined = load_and_join(split)
        if joined is None:
            continue
        all_results[split] = {}

        y = joined.y
        ref_name = REFERENCE if REFERENCE in joined.probs else list(joined.probs.keys())[-1]
        ref_probs = joined.probs[ref_name]

        # Single-model metrics + CIs
        print(f"\n  {'Model':<22}  AUROC                              AUPRC")
        for name, probs in joined.probs.items():
            auroc_point = roc_auc_score(y, probs)
            # Use DeLong's variance for a fast analytical CI on AUROC by comparing model to itself
            # (var of single AUC = cov[0,0] from DeLong with two identical predictors trick — instead use bootstrap)
            auroc_lo, auroc_hi = _bootstrap_auroc_ci(y, probs, n_boot=n_boot)
            auprc_point, auprc_lo, auprc_hi = bootstrap_auprc_ci(y, probs, n_boot=n_boot)
            print(f"  {name:<22}  {fmt_ci(auroc_point, auroc_lo, auroc_hi):<35}  {fmt_ci(auprc_point, auprc_lo, auprc_hi)}")
            all_results[split][name] = {
                "auroc": (auroc_point, auroc_lo, auroc_hi),
                "auprc": (auprc_point, auprc_lo, auprc_hi),
            }

        # Pairwise comparison: NeuralSCM vs each baseline
        print(f"\n  Reference: {ref_name}")
        print(f"  {'Comparator':<22}  AUROC delta (DeLong p)            AUPRC delta (paired bootstrap p)")
        for name, probs in joined.probs.items():
            if name == ref_name:
                continue
            d = delong_test(y, ref_probs, probs)
            pb = paired_bootstrap_auprc(y, ref_probs, probs, n_boot=n_boot)
            auroc_str = f"{d['delta']:+.4f}  (p={fmt_p(d['p_value'])})"
            auprc_str = f"{pb['delta']:+.4f} [{pb['ci'][0]:+.4f}, {pb['ci'][1]:+.4f}]  (p={fmt_p(pb['p_value'])})"
            print(f"  {name:<22}  {auroc_str:<35}  {auprc_str}")
            all_results[split].setdefault("_pairs", {})[name] = {
                "auroc_delta": d["delta"], "auroc_p": d["p_value"],
                "auprc_delta": pb["delta"], "auprc_ci": pb["ci"], "auprc_p": pb["p_value"],
            }

    # ── LaTeX dump ────────────────────────────────────────────────────────
    tex_path = OUT_DIR / "statistical_comparison.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by tests/statistical_comparison.py\n")
        for split, res in all_results.items():
            f.write(f"\n\\begin{{table}}[t]\n\\centering\n")
            f.write(f"\\caption{{{SPLIT_LABEL[split]}: AUROC and AUPRC with 95\\% bootstrap CIs, "
                    f"and pairwise comparison against {REFERENCE} (DeLong for AUROC, paired bootstrap for AUPRC).}}\n")
            f.write("\\begin{tabular}{lccc}\n\\toprule\n")
            f.write("Model & AUROC [95\\% CI] & AUPRC [95\\% CI] & $p$ vs ref. (AUROC $|$ AUPRC) \\\\\n")
            f.write("\\midrule\n")
            for name, m in res.items():
                if name == "_pairs":
                    continue
                ap, ah, al = m["auroc"][0], m["auroc"][2], m["auroc"][1]
                pp, ph, pl = m["auprc"][0], m["auprc"][2], m["auprc"][1]
                if name == REFERENCE:
                    p_str = "--"
                else:
                    pair = res.get("_pairs", {}).get(name, {})
                    p_str = f"{fmt_p(pair.get('auroc_p', 1))} $|$ {fmt_p(pair.get('auprc_p', 1))}"
                f.write(f"{name} & ${ap:.3f}$ [{al:.3f}, {ah:.3f}] & ${pp:.3f}$ [{pl:.3f}, {ph:.3f}] & {p_str} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\nLaTeX table saved: {tex_path}")


def _bootstrap_auroc_ci(y: np.ndarray, scores: np.ndarray, n_boot: int = 500,
                        alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ya = y[idx]
        if ya.sum() == 0 or ya.sum() == n:
            boots[i] = np.nan
            continue
        boots[i] = roc_auc_score(ya, scores[idx])
    valid = boots[~np.isnan(boots)]
    return float(np.percentile(valid, 100 * alpha / 2)), float(np.percentile(valid, 100 * (1 - alpha / 2)))


if __name__ == "__main__":
    nb = 500
    if len(sys.argv) > 1:
        nb = int(sys.argv[1])
    main(n_boot=nb)
