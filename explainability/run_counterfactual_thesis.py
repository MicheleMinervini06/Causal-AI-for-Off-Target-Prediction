"""Thesis-grade counterfactual analysis for Chapter 4.

Applies the canonical 4+1 set of intervention primitives to all (guide, off-target)
pairs of a dataset, using the FINAL model (Exp24). Produces:
  - CSV with per-pair counterfactual probabilities and deltas for each primitive
  - Pareto plot (delta_on vs delta_off) overlaying the 5 interventions
  - U distribution plot
  - Summary table (mean delta, fraction in ideal quadrant) per intervention

Interventions (cf Section 3.6 of the thesis):

  Coherence verification (predictable hierarchy):
    1. 5p_heal   (input-level):  sgRNA[0:2]  := off_target[0:2]
    3. ph_heal   (input-level):  sgRNA[16:20] := off_target[16:20]
    4. sh_heal   (input-level):  sgRNA[8:16]  := off_target[8:16]
    2. p14_do    (node-level):   do(pos_14 = 0.0)

  Operational what-if:
    5. pam_canon (input-level):  off_target[21:23] := "GG"  (forces canonical NGG PAM)

Default model: Exp24_MergedSplit_Causal_0p1 (the adopted final configuration).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from explainability._intervention_utils import (
    abduct_U,
    build_offtarget_dataframe,
    compute_gc_context_batch,
    counterfactual_prob_pct,
    filter_saturated_pairs,
    load_neural_scm,
    load_positive_dataset,
    model_forward_batched,
    model_pred_pct,
    resolve_on_target_mode,
)


# ---------- intervention primitives (input-level rewrites) ----------

def heal_5p(guide: str, off: str) -> str:
    """5' healing: sgRNA[0:2] := off_target[0:2]. Removes any mismatch at PAM-distal positions 1-2."""
    return off[0:2] + guide[2:20]


def heal_pamprox(guide: str, off: str) -> str:
    """PAM-proximal healing: sgRNA[16:20] := off_target[16:20]. Removes mismatches at positions 17-20."""
    return guide[0:16] + off[16:20]


def heal_seed(guide: str, off: str) -> str:
    """Seed healing: sgRNA[8:16] := off_target[8:16]. Removes mismatches in the seed (positions 9-16)."""
    return guide[0:8] + off[8:16] + guide[16:20]


def canonize_pam(target: str) -> str:
    """PAM canonization: force positions 21-22 to 'GG' (canonical NGG). Leaves N at position 20."""
    return target[0:21] + "GG"


# ---------- pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["changeseq", "guideseq"], required=True)
    parser.add_argument(
        "--model_path",
        default="experiments/results/Exp24_MergedSplit_Causal_0p1/neural_scm.pt",
        help="Path to the .pt checkpoint. Defaults to Exp24 (the adopted final configuration).",
    )
    parser.add_argument("--pam-mode", choices=["additive", "multiplicative"], default="additive")
    parser.add_argument("--output_dir", default="explainability/thesis_results")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--on-target-mode",
        choices=["drop", "per_run", "global_max"],
        default=None,
        help="Default: per_run per guideseq, drop per changeseq",
    )
    parser.add_argument("--filter-saturated", action="store_true")

    args = parser.parse_args()
    args.on_target_mode = resolve_on_target_mode(args.on_target_mode, args.dataset)

    print(f"Dataset:          {args.dataset}")
    print(f"Model:            {args.model_path}")
    print(f"PAM mode:         {args.pam_mode}")
    print(f"On-target mode:   {args.on_target_mode}")
    print(f"Filter saturated: {args.filter_saturated}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model, device, context_dim = load_neural_scm(args.model_path, pam_mode=args.pam_mode)
    print(f"Device: {device}  |  context_dim={context_dim}")

    # 2. Load dataset
    df, reads_col = load_positive_dataset(args.dataset)
    off_df = build_offtarget_dataframe(df, reads_col, on_target_mode=args.on_target_mode)
    if args.filter_saturated:
        off_df, _ = filter_saturated_pairs(off_df, verbose=True)

    # 3. Build counterfactual inputs for the 4 input-level interventions
    guides_wt = off_df["sgRNA"].tolist()
    off_targets = off_df["off_target"].tolist()
    on_targets = off_df["on_target_seq"].tolist()

    guides_5h = [heal_5p(g, o) for g, o in zip(guides_wt, off_targets)]
    guides_ph = [heal_pamprox(g, o) for g, o in zip(guides_wt, off_targets)]
    guides_sh = [heal_seed(g, o) for g, o in zip(guides_wt, off_targets)]
    # For PAM canonization, on-target healing uses on_target itself (since we heal sgRNA to on-target);
    # but here we modify the OFF-target's PAM. On-target healing for healing primitives uses on_target slice.
    off_targets_pc = [canonize_pam(t) for t in off_targets]
    on_targets_pc = [canonize_pam(t) for t in on_targets]

    # For the healing primitives applied to the on-target prediction, we apply the same
    # healing logic but on (guide, on_target) — for the on-target the "off" reference is the on-target itself.
    guides_5h_on = [heal_5p(g, o) for g, o in zip(guides_wt, on_targets)]
    guides_ph_on = [heal_pamprox(g, o) for g, o in zip(guides_wt, on_targets)]
    guides_sh_on = [heal_seed(g, o) for g, o in zip(guides_wt, on_targets)]

    # 4. Forward passes
    def fwd(guides: list[str], targets: list[str], intervention: dict | None = None):
        ctx = compute_gc_context_batch(guides, targets, device)
        return model_forward_batched(model, guides, targets, ctx, args.batch_size, intervention=intervention)

    print("Forward factual...")
    logit_off_f, pam_off_f = fwd(guides_wt, off_targets)
    logit_on_f, pam_on_f = fwd(guides_wt, on_targets)

    print("Forward 5' healing...")
    logit_off_5h, pam_off_5h = fwd(guides_5h, off_targets)
    logit_on_5h, pam_on_5h = fwd(guides_5h_on, on_targets)

    # Node-level interventions: do(pos_k = 0) for k in {2, 10, 14, 18}
    # Representative positions: 5'-distal (2), seed-edge (10), mid-seed (14), PAM-proximal (18).
    POS_NODE = [2, 10, 14, 18]
    logit_off_pos, pam_off_pos = {}, {}
    logit_on_pos, pam_on_pos = {}, {}
    for k in POS_NODE:
        print(f"Forward do(pos_{k} = 0)...")
        logit_off_pos[k], pam_off_pos[k] = fwd(guides_wt, off_targets, intervention={f"pos_{k}": 0.0})
        logit_on_pos[k], pam_on_pos[k] = fwd(guides_wt, on_targets, intervention={f"pos_{k}": 0.0})

    print("Forward PAM-prox healing...")
    logit_off_ph, pam_off_ph = fwd(guides_ph, off_targets)
    logit_on_ph, pam_on_ph = fwd(guides_ph_on, on_targets)

    print("Forward seed healing...")
    logit_off_sh, pam_off_sh = fwd(guides_sh, off_targets)
    logit_on_sh, pam_on_sh = fwd(guides_sh_on, on_targets)

    print("Forward PAM canonization...")
    logit_off_pc, pam_off_pc = fwd(guides_wt, off_targets_pc)
    logit_on_pc, pam_on_pc = fwd(guides_wt, on_targets_pc)

    # 5. Factual predictions
    off_df["pam_off_f"] = pam_off_f
    off_df["pam_on_f"] = pam_on_f
    off_df["y_pred_off_prob"] = model_pred_pct(logit_off_f, pam_off_f, args.pam_mode)
    off_df["y_pred_on_prob"] = model_pred_pct(logit_on_f, pam_on_f, args.pam_mode)

    # 6. Abduction (off-target)
    off_df["U_off"] = abduct_U(
        np.asarray(off_df["y_obs_off_prob"].values),
        logit_off_f, pam_off_f, pam_mode=args.pam_mode,
    )
    U_off = np.asarray(off_df["U_off"].values)

    cf = lambda l, p: counterfactual_prob_pct(l, p, U_off, pam_mode=args.pam_mode)

    # Input-level interventions
    off_df["y_cf_off_5h"] = cf(logit_off_5h, pam_off_5h)
    off_df["y_cf_off_ph"] = cf(logit_off_ph, pam_off_ph)
    off_df["y_cf_off_sh"] = cf(logit_off_sh, pam_off_sh)
    off_df["y_cf_off_pc"] = cf(logit_off_pc, pam_off_pc)
    # Node-level interventions: do(pos_k = 0)
    for k in POS_NODE:
        off_df[f"y_cf_off_p{k}"] = cf(logit_off_pos[k], pam_off_pos[k])

    DELTA_KEYS = ["5h", "ph", "sh", "pc"] + [f"p{k}" for k in POS_NODE]
    for key in DELTA_KEYS:
        off_df[f"delta_off_{key}"] = off_df[f"y_cf_off_{key}"] - off_df["y_obs_off_prob"]

    # Mismatch presence at the node-level intervention positions (for stratification)
    for k in POS_NODE:
        off_df[f"has_mm_p{k}"] = (
            off_df["sgRNA"].str[k] != off_df["off_target"].str[k]
        )

    # 7. On-target counterfactuals
    if args.on_target_mode == "drop":
        # No abducted U_on; use raw model prediction
        off_df["U_on"] = np.nan
        baseline_on = off_df["y_pred_on_prob"]
        on_pred = lambda l, p: model_pred_pct(l, p, args.pam_mode)
        off_df["y_cf_on_5h"] = on_pred(logit_on_5h, pam_on_5h)
        off_df["y_cf_on_ph"] = on_pred(logit_on_ph, pam_on_ph)
        off_df["y_cf_on_sh"] = on_pred(logit_on_sh, pam_on_sh)
        off_df["y_cf_on_pc"] = on_pred(logit_on_pc, pam_on_pc)
        for k in POS_NODE:
            off_df[f"y_cf_on_p{k}"] = on_pred(logit_on_pos[k], pam_on_pos[k])
    else:
        off_df["U_on"] = abduct_U(
            np.asarray(off_df["y_obs_on_prob"].values),
            logit_on_f, pam_on_f, pam_mode=args.pam_mode,
        )
        U_on = np.asarray(off_df["U_on"].values)
        cf_on = lambda l, p: counterfactual_prob_pct(l, p, U_on, pam_mode=args.pam_mode)
        off_df["y_cf_on_5h"] = cf_on(logit_on_5h, pam_on_5h)
        off_df["y_cf_on_ph"] = cf_on(logit_on_ph, pam_on_ph)
        off_df["y_cf_on_sh"] = cf_on(logit_on_sh, pam_on_sh)
        off_df["y_cf_on_pc"] = cf_on(logit_on_pc, pam_on_pc)
        for k in POS_NODE:
            off_df[f"y_cf_on_p{k}"] = cf_on(logit_on_pos[k], pam_on_pos[k])
        baseline_on = off_df["y_obs_on_prob"]

    for key in DELTA_KEYS:
        off_df[f"delta_on_{key}"] = off_df[f"y_cf_on_{key}"] - baseline_on

    # 8. Save CSV — filename suffix tracks the filter
    file_suffix = "_filtsat" if args.filter_saturated else ""
    keep_cols = [
        "name", "sgRNA", "off_target", "on_target_seq", "distance",
        "off_reads", "on_reads",
        "pam_off_f", "pam_on_f",
        "y_obs_off_prob", "y_pred_off_prob", "U_off",
        "y_obs_on_prob", "y_pred_on_prob", "U_on",
    ]
    for key in DELTA_KEYS:
        keep_cols += [f"y_cf_off_{key}", f"y_cf_on_{key}", f"delta_off_{key}", f"delta_on_{key}"]
    keep_cols += [f"has_mm_p{k}" for k in POS_NODE]
    out_csv = output_dir / f"thesis_cf_{args.dataset}{file_suffix}.csv"
    off_df[keep_cols].to_csv(out_csv, index=False)
    print(f"\nCSV saved: {out_csv}")

    # 9a. Headline distribution plot — Δ_off per intervention (input-level + PAM canonization + p14 summary)
    interventions = [
        ("5h", "5' healing",        "steelblue"),
        ("ph", "PAM-prox healing",  "forestgreen"),
        ("sh", "Seed healing",      "darkorange"),
        ("p14", "do(pos_14 = 0)",   "crimson"),
        ("pc", "PAM canonization",  "purple"),
    ]
    sns.set_theme(style="whitegrid")

    plot_df = pd.concat([
        pd.DataFrame({
            "intervention": label,
            "delta_off": off_df[f"delta_off_{key}"].values,
        }) for key, label, _ in interventions
    ], ignore_index=True)
    palette = {label: color for _, label, color in interventions}
    order = [label for _, label, _ in interventions]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=plot_df, x="intervention", y="delta_off", hue="intervention",
        order=order, palette=palette, inner="quartile", cut=0, linewidth=1.0,
        ax=ax, legend=False,
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel(r"$\Delta_{\mathrm{off}}$ (counterfactual - factual) [%]")
    title_suffix = " (no-saturated)" if args.filter_saturated else ""
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    dist_path = output_dir / f"thesis_cf_{args.dataset}{file_suffix}_delta_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print(f"Delta-distribution plot saved: {dist_path}")

    # 9b. Node-level interventions stratified by mismatch presence at the target position
    node_records = []
    for k in POS_NODE:
        d_off = off_df[f"delta_off_p{k}"].values
        has_mm = off_df[f"has_mm_p{k}"].values
        for delta, mm in zip(d_off, has_mm):
            node_records.append({
                "position": f"P{k}",
                "stratum": "mismatch at P_k" if mm else "match at P_k",
                "delta_off": float(delta),
            })
    node_df = pd.DataFrame(node_records)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=node_df, x="position", y="delta_off", hue="stratum",
        order=[f"P{k}" for k in POS_NODE],
        hue_order=["match at P_k", "mismatch at P_k"],
        palette={"match at P_k": "lightgray", "mismatch at P_k": "crimson"},
        split=True, inner="quartile", cut=0, linewidth=1.0, ax=ax,
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Intervention target position $k$ (DAG node $P_k$)")
    ax.set_ylabel(r"$\Delta_{\mathrm{off}}$ from $do(P_k = 0)$ [%]")
    ax.legend(title="", loc="best")
    plt.tight_layout()
    node_path = output_dir / f"thesis_cf_{args.dataset}{file_suffix}_node_level.png"
    plt.savefig(node_path, dpi=200)
    plt.close()
    print(f"Node-level plot saved: {node_path}")

    # 9c. PAM canonization stratified by original off-target PAM
    off_df["off_pam"] = off_df["off_target"].str[20:23]
    off_df["pam_class"] = np.where(off_df["off_pam"].str[1:3] == "GG", "Canonical (NGG)", "Non-canonical (non-NGG)")
    pc_df = off_df[["pam_class", "off_pam", "delta_off_pc"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 2]})

    sns.violinplot(
        data=pc_df, x="pam_class", y="delta_off_pc", hue="pam_class",
        order=["Canonical (NGG)", "Non-canonical (non-NGG)"],
        palette={"Canonical (NGG)": "lightgray", "Non-canonical (non-NGG)": "purple"},
        inner="quartile", cut=0, linewidth=1.0, ax=axes[0], legend=False,
    )
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("PAM class")
    axes[0].set_ylabel(r"$\Delta_{\mathrm{off}}$ from PAM canonization [%]")

    pam_counts = pc_df["off_pam"].value_counts()
    top_pams = pam_counts[pam_counts >= max(20, int(0.005 * len(pc_df)))].index.tolist()
    pc_top = pc_df[pc_df["off_pam"].isin(top_pams)].copy()
    pam_order = pam_counts.loc[top_pams].sort_index().index.tolist()
    pam_palette = {p: ("lightgray" if p[1:3] == "GG" else "purple") for p in pam_order}

    sns.violinplot(
        data=pc_top, x="off_pam", y="delta_off_pc", hue="off_pam",
        order=pam_order, palette=pam_palette,
        inner="quartile", cut=0, linewidth=1.0, ax=axes[1], legend=False,
    )
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Original off-target PAM (positions 21-23)")
    axes[1].set_ylabel("")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(0)

    plt.tight_layout()
    pc_path = output_dir / f"thesis_cf_{args.dataset}{file_suffix}_pam_canonization.png"
    plt.savefig(pc_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"PAM canonization plot saved: {pc_path}")

    # 10. U distribution
    has_u_on = args.on_target_mode != "drop"
    n_panels = 2 if has_u_on else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    panels = [(axes[0, 0], "U_off", "steelblue", r"$\hat U_{\mathrm{off}}$")]
    if has_u_on:
        panels.append((axes[0, 1], "U_on", "darkorange", r"$\hat U_{\mathrm{on}}$"))
    for ax, col, color, title in panels:
        vals = off_df[col].dropna().values
        ax.hist(vals, bins=60, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(float(np.mean(vals)), color="red", linestyle="--",
                   label=f"mean={float(np.mean(vals)):+.3f}")
        ax.axvline(float(np.median(vals)), color="black", linestyle=":",
                   label=f"median={float(np.median(vals)):+.3f}")
        ax.set_xlabel(title)
        ax.legend()
    plt.tight_layout()
    u_path = output_dir / f"thesis_cf_{args.dataset}{file_suffix}_U_distribution.png"
    plt.savefig(u_path, dpi=200)
    plt.close()
    print(f"U distribution plot saved: {u_path}")

    # 11. Summary table (headline interventions)
    summary_rows = []
    for key, label, _ in interventions:
        d_off = off_df[f"delta_off_{key}"].values
        d_on = off_df[f"delta_on_{key}"].values
        summary_rows.append({
            "intervention": label,
            "n_pairs": int(len(d_off)),
            "mean_delta_off_%": float(np.mean(d_off)),
            "median_delta_off_%": float(np.median(d_off)),
            "std_delta_off_%": float(np.std(d_off)),
            "mean_delta_on_%": float(np.mean(d_on)),
            "median_delta_on_%": float(np.median(d_on)),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"thesis_cf_{args.dataset}{file_suffix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary table saved: {summary_path}")

    # 11b. Stratified summary for node-level interventions (treated vs untreated effect)
    node_summary_rows = []
    for k in POS_NODE:
        d_off = off_df[f"delta_off_p{k}"].values
        has_mm = off_df[f"has_mm_p{k}"].values.astype(bool)
        n_mm = int(has_mm.sum())
        n_nomm = int((~has_mm).sum())
        mean_mm = float(np.mean(d_off[has_mm])) if n_mm > 0 else float("nan")
        median_mm = float(np.median(d_off[has_mm])) if n_mm > 0 else float("nan")
        mean_nomm = float(np.mean(d_off[~has_mm])) if n_nomm > 0 else float("nan")
        node_summary_rows.append({
            "position": f"P{k}",
            "n_pairs_total": int(len(d_off)),
            "n_with_mismatch": n_mm,
            "frac_with_mismatch_%": float(100.0 * n_mm / max(len(d_off), 1)),
            "mean_delta_off_overall_%": float(np.mean(d_off)),
            "mean_delta_off_mismatch_stratum_%": mean_mm,
            "median_delta_off_mismatch_stratum_%": median_mm,
            "mean_delta_off_match_stratum_%": mean_nomm,
        })
    node_summary = pd.DataFrame(node_summary_rows)
    node_summary_path = output_dir / f"thesis_cf_{args.dataset}{file_suffix}_node_level_summary.csv"
    node_summary.to_csv(node_summary_path, index=False)
    print(f"Node-level stratified summary saved: {node_summary_path}")

    # 12. Console summary
    print("\n=== SUMMARY (per-pair) ===")
    print(f"Pairs: {len(off_df)}  |  unique sgRNAs: {off_df['name'].nunique()}")
    print(f"U_off mean={off_df['U_off'].mean():+.3f}  median={off_df['U_off'].median():+.3f}  std={off_df['U_off'].std():.3f}")
    print()
    print("--- Headline interventions ---")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print()
    print("--- Node-level interventions stratified by mismatch presence ---")
    print(node_summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # 13. Provenance metadata
    meta = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "pam_mode": args.pam_mode,
        "on_target_mode": args.on_target_mode,
        "filter_saturated": args.filter_saturated,
        "n_pairs": int(len(off_df)),
        "n_sgRNAs": int(off_df["name"].nunique()),
        "headline_interventions": [
            {"key": k, "label": lab} for k, lab, _ in interventions
        ],
        "node_level_positions": POS_NODE,
    }
    meta_path = output_dir / f"thesis_cf_{args.dataset}{file_suffix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nProvenance metadata: {meta_path}")


if __name__ == "__main__":
    main()
