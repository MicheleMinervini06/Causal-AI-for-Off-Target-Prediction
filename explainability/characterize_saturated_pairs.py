"""Caratterizzazione delle feature delle coppie saturate (CHANGE-seq cell-free).

Test F22.1: identifica gli elementi comuni alle coppie che presentano il fenomeno
di saturazione (off_reads >= on_reads), per capire se sono:
  - errori di misurazione casuali
  - sistematicamente diverse dal resto (es. specifiche regioni del genoma,
    pattern di mismatch, GC bias, particolari guide)

Per ogni feature candidata, calcola:
  - Distribuzione stratificata saturated vs not-saturated
  - Mann-Whitney U test (più robusto del t-test) + Cohen's d
  - Effect size visualizzato in barplot

Output: plot a multi-panel + JSON con metriche.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def gc_fraction(seq: str) -> float:
    if not isinstance(seq, str) or len(seq) == 0:
        return np.nan
    return sum(1 for c in seq.upper() if c in "GC") / len(seq)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled std). Effect size standardizzato."""
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s_pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if s_pooled < 1e-12:
        return 0.0
    return float((b.mean() - a.mean()) / s_pooled)


def mismatch_positions_mask(sgrna: str, off_target: str) -> np.ndarray:
    """Restituisce array binario [20] con 1 dove c'e' mismatch nello spacer (pos 0-19)."""
    sg = sgrna[:20].upper().ljust(20, "N")
    ot = off_target[:20].upper().ljust(20, "N")
    return np.array([1 if sg[i] != ot[i] else 0 for i in range(20)], dtype=np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path,
        default=Path("explainability/batch_results/changeseq_batch_results_shift+2.73.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("explainability/batch_results"),
    )
    args = parser.parse_args()

    print(f"Loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  rows: {len(df)}")

    df["is_saturated"] = (df["off_reads"] >= df["on_reads"]).astype(bool)

    # === Compute derived features ===
    print("Computing derived features...")
    df["gc_sgrna"] = df["sgRNA"].apply(gc_fraction)
    df["gc_offtarget_spacer"] = df["off_target"].apply(lambda x: gc_fraction(str(x)[:20]))
    df["gc_pam"] = df["off_target"].apply(lambda x: gc_fraction(str(x)[20:23]))
    df["gc_delta"] = df["gc_sgrna"] - df["gc_offtarget_spacer"]
    df["pam_seq"] = df["off_target"].apply(lambda x: str(x)[20:23].upper() if isinstance(x, str) and len(x) >= 23 else "NNN")
    df["pam_is_ngg"] = df["pam_seq"].apply(lambda p: p.endswith("GG"))

    # Per-position mismatch flags
    print("  computing per-position mismatch flags...")
    mm_arrays = df.apply(lambda r: mismatch_positions_mask(str(r["sgRNA"]), str(r["off_target"])), axis=1)
    mm_matrix = np.stack(mm_arrays.values)  # shape [N, 20]
    for i in range(20):
        df[f"mm_pos_{i:02d}"] = mm_matrix[:, i]

    # Region-specific mismatch counts
    df["mm_nonseed_count"] = mm_matrix[:, 0:8].sum(axis=1)   # PAM-distal
    df["mm_seed_count"]    = mm_matrix[:, 8:16].sum(axis=1)
    df["mm_prox_count"]    = mm_matrix[:, 16:20].sum(axis=1) # PAM-proximal

    # === Feature comparison ===
    sat = df[df["is_saturated"]]
    nsat = df[~df["is_saturated"]]
    n_sat = len(sat)
    n_nsat = len(nsat)

    print(f"\n=== POPULATIONS ===")
    print(f"  saturated:     n={n_sat:>6d}  ({100*n_sat/len(df):.1f}%)")
    print(f"  not saturated: n={n_nsat:>6d}  ({100*n_nsat/len(df):.1f}%)")

    continuous_features = [
        ("distance",              "Mismatch count (total)"),
        ("mm_nonseed_count",      "Mismatches in non-seed (pos 0-7)"),
        ("mm_seed_count",         "Mismatches in seed (pos 8-15)"),
        ("mm_prox_count",         "Mismatches in PAM-proximal (pos 16-19)"),
        ("gc_sgrna",              "sgRNA GC content"),
        ("gc_offtarget_spacer",   "off-target spacer GC content"),
        ("gc_pam",                "off-target PAM GC content"),
        ("gc_delta",              "GC(sgRNA) - GC(off-target)"),
        ("pam_off_f",             "Model pam_gate (sigmoid output)"),
    ]

    print(f"\n=== CONTINUOUS FEATURE COMPARISON (saturated vs not) ===")
    print(f"{'Feature':<35s}  {'mean_nsat':>10s}  {'mean_sat':>10s}  {'cohen_d':>8s}  {'U_stat':>12s}  {'p_value':>10s}")
    print("-" * 95)

    continuous_results = []
    for key, label in continuous_features:
        a = nsat[key].dropna().values
        b = sat[key].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        d = cohens_d(a, b)
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"{label:<35s}  {a.mean():>10.3f}  {b.mean():>10.3f}  {d:>+8.3f}  {u_stat:>12.0f}  {p_val:>10.2e}")
        continuous_results.append({
            "feature": key,
            "label": label,
            "mean_not_saturated": float(a.mean()),
            "mean_saturated": float(b.mean()),
            "std_not_saturated": float(a.std()),
            "std_saturated": float(b.std()),
            "median_not_saturated": float(np.median(a)),
            "median_saturated": float(np.median(b)),
            "cohens_d": float(d),
            "mannwhitneyu_stat": float(u_stat),
            "p_value": float(p_val),
        })

    # === PAM categorical analysis ===
    print(f"\n=== PAM SEQUENCE BREAKDOWN ===")
    pam_table = pd.crosstab(df["pam_seq"], df["is_saturated"], margins=True, margins_name="Total")
    pam_table.columns = ["NotSat", "Saturated", "Total"]
    pam_table["P(Sat|PAM)"] = pam_table["Saturated"] / pam_table["Total"]
    pam_table = pam_table.sort_values("Total", ascending=False).head(15)
    print(pam_table.to_string())

    # === Per-guide saturation rate ===
    print(f"\n=== PER-GUIDE SATURATION RATE (top 15 most-saturated guides) ===")
    per_guide = df.groupby("name").agg(
        n_pairs=("is_saturated", "size"),
        n_saturated=("is_saturated", "sum"),
    )
    per_guide["sat_rate"] = per_guide["n_saturated"] / per_guide["n_pairs"]
    print(per_guide.sort_values("sat_rate", ascending=False).head(15).to_string())

    print(f"\nMean per-guide saturation rate: {per_guide['sat_rate'].mean():.3f}")
    print(f"Median per-guide saturation rate: {per_guide['sat_rate'].median():.3f}")
    print(f"Number of guides with >50% saturation: {(per_guide['sat_rate'] > 0.5).sum()}/{len(per_guide)}")
    print(f"Number of guides with <10% saturation: {(per_guide['sat_rate'] < 0.1).sum()}/{len(per_guide)}")

    # === Per-position mismatch frequency ===
    print(f"\n=== PER-POSITION MISMATCH FREQUENCY ===")
    pos_freq_nsat = nsat[[f"mm_pos_{i:02d}" for i in range(20)]].mean().values
    pos_freq_sat = sat[[f"mm_pos_{i:02d}" for i in range(20)]].mean().values
    print(f"{'pos':<4s}  {'P(mm|nsat)':>10s}  {'P(mm|sat)':>10s}  {'delta':>8s}")
    for i in range(20):
        delta = pos_freq_sat[i] - pos_freq_nsat[i]
        print(f"{i:<4d}  {pos_freq_nsat[i]:>10.3f}  {pos_freq_sat[i]:>10.3f}  {delta:>+8.3f}")

    # === Plots ===
    # Two separate figures aligned with the narrative of Section 4.4.3:
    #   (a) Effect size by feature  -> changeseq_saturated_effect_size.png
    #   (b) Per-guide saturation rate distribution -> changeseq_saturated_per_guide.png
    # The two are combined as subfigures in the LaTeX source for thesis layout.
    # Palette aligned with the rest of the thesis figures.
    COLOR_NSAT = "#7373FF"          # pgfplots blue!55 — not saturated (baseline)
    COLOR_SAT  = "#D62728"          # crimson — saturated (alert / outlier)
    COLOR_GUIDE_HIST = "#FFA64D"    # pgfplots orange!70 — per-guide distribution

    sns.set_theme(style="whitegrid", context="talk")

    # --- (a) Cohen's d by feature ---
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sorted_results = sorted(continuous_results, key=lambda r: abs(r["cohens_d"]), reverse=True)
    labels = [r["label"] for r in sorted_results]
    ds = [r["cohens_d"] for r in sorted_results]
    colors = [COLOR_SAT if d > 0 else COLOR_NSAT for d in ds]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, ds, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    for lvl, ls in [(0.2, ":"), (0.5, "--"), (0.8, "-")]:
        ax.axvline( lvl, color="gray", linestyle=ls, linewidth=0.6)
        ax.axvline(-lvl, color="gray", linestyle=ls, linewidth=0.6)
    ax.set_xlabel("Cohen's $d$ (saturated $-$ not saturated)")
    plt.tight_layout()
    plot_a = args.output_dir / "changeseq_saturated_effect_size.png"
    plt.savefig(plot_a, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSalvato {plot_a}")

    # --- (b) Per-guide saturation rate distribution ---
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(per_guide["sat_rate"].values, bins=30,
            color=COLOR_GUIDE_HIST, alpha=0.85, edgecolor="white", linewidth=0.5)
    mean_rate = per_guide["sat_rate"].mean()
    med_rate  = per_guide["sat_rate"].median()
    ax.axvline(med_rate, color="black", linestyle=":",
               label=f"median = {med_rate:.2f}", linewidth=1.8)
    ax.axvline(mean_rate, color=COLOR_SAT, linestyle="--",
               label=f"mean = {mean_rate:.2f}", linewidth=1.8)
    # Annotate the sgRNA that contributes the largest absolute volume of
    # saturated rows (not just the highest per-guide rate -- some guides have
    # high rate but very few pairs and are therefore irrelevant in absolute
    # terms). The per_guide DataFrame is indexed by "name".
    total_sat = int(per_guide["n_saturated"].sum())
    outlier = per_guide.sort_values("n_saturated", ascending=False).iloc[0]
    outlier_name = str(outlier.name).replace("_", r"\_")
    outlier_share = outlier["n_saturated"] / total_sat
    ax.annotate(
        f"{outlier_name}\nrate $= {outlier['sat_rate']:.2f}$\n"
        f"({outlier_share * 100:.1f}\\% of all saturated)",
        xy=(outlier["sat_rate"], 1),
        xytext=(outlier["sat_rate"] - 0.30, 12),
        fontsize=10, color=COLOR_SAT,
        arrowprops=dict(arrowstyle="->", color=COLOR_SAT, lw=1.0),
    )
    ax.set_xlabel("Per-guide saturation rate")
    ax.set_ylabel(f"Number of guides ($n = {len(per_guide)}$)")
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plot_b = args.output_dir / "changeseq_saturated_per_guide.png"
    plt.savefig(plot_b, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Salvato {plot_b}")

    # === JSON output ===
    payload = {
        "csv_input": str(args.csv),
        "n_total": len(df),
        "n_saturated": n_sat,
        "n_not_saturated": n_nsat,
        "continuous_features": continuous_results,
        "per_position_mismatch_freq": {
            "not_saturated": pos_freq_nsat.tolist(),
            "saturated": pos_freq_sat.tolist(),
        },
        "pam_breakdown_top": pam_table.head(10).to_dict(),
        "per_guide_saturation_stats": {
            "mean_rate": float(per_guide["sat_rate"].mean()),
            "median_rate": float(per_guide["sat_rate"].median()),
            "std_rate": float(per_guide["sat_rate"].std()),
            "n_guides_above_50pct": int((per_guide["sat_rate"] > 0.5).sum()),
            "n_guides_below_10pct": int((per_guide["sat_rate"] < 0.1).sum()),
            "n_guides_total": int(len(per_guide)),
        },
    }
    json_path = args.output_dir / "changeseq_saturated_pairs_characterization.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Salvato {json_path}")

    # === Verdict ===
    print(f"\n=== VERDICT ===")
    top_features = sorted(continuous_results, key=lambda r: abs(r["cohens_d"]), reverse=True)[:3]
    print(f"Top 3 discriminating features (by |Cohen's d|):")
    for r in top_features:
        print(f"  {r['label']:<40s} d = {r['cohens_d']:+.3f}")


if __name__ == "__main__":
    main()
