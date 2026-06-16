"""Batch counterfactual analysis: applica interventi a tutte le coppie
(guide, off-target) di un dataset e produce CSV + plot Pareto + distribuzioni di U.

Default model: Exp18_Positional_AdditivePAM (pam_mode=additive, encoding 4-dim).

Interventi implementati:
  1) truncate_5p          (sequence-level):  guide → "NN" + guide[2:]
  2) do(pos_14=0)         (DAG node-level):  forza penalità a 0 sul nodo P_14
  3) diversity ACGT       (sequence-level, Treatment-Control)
  4) repeat seed          (sequence-level, Treatment-Control)

Calibrazione assay (P1): tramite `--assay-shift <float>` o
`--assay-shift-from <path/to/calibration.json>` si applica un offset additivo
al logit del modello prima dell'abduction. È pensato per il caso "modello
addestrato su CHANGE-seq, valutato su GUIDE-seq" — il shift `b̂` viene stimato
da `calibrate_assay_shift.py` su un piccolo set di calibrazione.

Helper condivisi: vedi `_intervention_utils.py`.
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
    sigmoid,
)


# ---------- interventi a livello sequenza ----------

def truncate_5p(guide: str) -> str:
    """Maschera le prime 2 basi (troncamento 5'). Sequence-level."""
    return "NN" + guide[2:]


def force_pamprox_acgt(guide: str) -> str:
    """Diversity Treatment: forza guide[16:20] = "ACGT" (max diversità A/C/G/T)."""
    return guide[:16] + "ACGT"


def force_pamprox_aaaa(guide: str) -> str:
    """Diversity Control: forza guide[16:20] = "AAAA" (nessuna diversità)."""
    return guide[:16] + "AAAA"


def force_seed_repeat(guide: str) -> str:
    """Repeat Treatment: guide[8:16] = "ATATATAT" (perfect period-2 repeat)."""
    return guide[:8] + "ATATATAT" + guide[16:]


def force_seed_block(guide: str) -> str:
    """Repeat Control: guide[8:16] = "AAAATTTT" (stessa composizione A/T, no period-2)."""
    return guide[:8] + "AAAATTTT" + guide[16:]


# ---------- assay-shift resolution ----------

def resolve_assay_shift(
    assay_shift_value: float | None,
    assay_shift_from: Path | None,
) -> tuple[float, str]:
    """Risolve il valore di `b̂` da CLI argument o file JSON.

    Restituisce (shift_value, source_label) dove source_label è un identificatore
    leggibile (es. "explicit=-2.61", "file=N=5_median").
    """
    if assay_shift_value is not None and assay_shift_from is not None:
        raise ValueError("Specificare solo uno tra --assay-shift e --assay-shift-from")

    if assay_shift_value is not None:
        return float(assay_shift_value), f"explicit={assay_shift_value:+.3f}"

    if assay_shift_from is not None:
        with open(assay_shift_from, "r") as f:
            payload = json.load(f)
        # Convention: il JSON contiene una chiave "selected_shift" con il valore preferito
        # (es. il median bootstrap di un N_calib scelto). Vedi calibrate_assay_shift.py.
        if "selected_shift" in payload:
            shift = float(payload["selected_shift"])
            return shift, f"file={assay_shift_from.name}"
        # Fallback: prendi il median bootstrap del N_calib più piccolo disponibile
        sweep = payload.get("sweep", [])
        if not sweep:
            raise ValueError(f"Nessun campo 'selected_shift' o 'sweep' in {assay_shift_from}")
        # Prende il primo entry
        entry = sweep[0]
        shift = float(entry["b_shift"]["median"])
        return shift, f"file={assay_shift_from.name}(N={entry['n_calib']})"

    return 0.0, "none"


# ---------- pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["changeseq", "guideseq"], default="guideseq")
    parser.add_argument(
        "--model_path",
        default="experiments/results/Exp24_MergedSplit_Causal_0p1/neural_scm.pt",
    )
    parser.add_argument("--pam-mode", choices=["additive", "multiplicative"], default="additive")
    parser.add_argument("--output_dir", default="explainability/batch_results")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--on-target-mode",
        choices=["drop", "per_run", "global_max"],
        default=None,
        help="Default: per_run per guideseq, drop per changeseq",
    )
    parser.add_argument(
        "--filter-saturated",
        action="store_true",
        help="Rimuove coppie saturated (off_reads >= on_reads). Utile per Analysis A "
             "post Exp20 (testa F23 sul regime operativo del modello).",
    )

    # Assay shift (P1 calibration)
    group_shift = parser.add_mutually_exclusive_group()
    group_shift.add_argument(
        "--assay-shift", type=float, default=None,
        help="Offset additivo del logit (P1 calibration). Es. -2.6 per ricalibrare CHANGE-seq → GUIDE-seq.",
    )
    group_shift.add_argument(
        "--assay-shift-from", type=Path, default=None,
        help="Path al JSON prodotto da calibrate_assay_shift.py. Usa il campo 'selected_shift'.",
    )

    args = parser.parse_args()

    args.on_target_mode = resolve_on_target_mode(args.on_target_mode, args.dataset)
    assay_shift, shift_source = resolve_assay_shift(args.assay_shift, args.assay_shift_from)

    print(f"PAM mode:         {args.pam_mode}")
    print(f"On-target mode:   {args.on_target_mode}")
    print(f"Model:            {args.model_path}")
    print(f"Assay shift:      {assay_shift:+.4f}  (source: {shift_source})")
    print(f"Filter saturated: {args.filter_saturated}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Modello
    model, device, context_dim = load_neural_scm(args.model_path, pam_mode=args.pam_mode)
    print(f"Device: {device}")
    print(f"Modello caricato (context_dim={context_dim}, pam_mode={args.pam_mode})")

    # 2-5. Dataset
    df, reads_col = load_positive_dataset(args.dataset)
    print(f"Caricate {len(df)} righe dal dataset {args.dataset}")
    off_df = build_offtarget_dataframe(df, reads_col, on_target_mode=args.on_target_mode)

    # 5b. Optional filter: rimuovi coppie saturated (off_reads >= on_reads)
    # Coerente con il filtro applicato in training (vedi F23 in findings.md).
    if args.filter_saturated:
        off_df, _ = filter_saturated_pairs(off_df, verbose=True)
        if len(off_df) == 0:
            raise RuntimeError("No pairs left after --filter-saturated. Aborting.")

    # 6. Costruzione sequenze post-intervento (sequence-level)
    off_df["sgRNA_truncated"] = off_df["sgRNA"].apply(truncate_5p)
    off_df["sgRNA_divT"] = off_df["sgRNA"].apply(force_pamprox_acgt)
    off_df["sgRNA_divC"] = off_df["sgRNA"].apply(force_pamprox_aaaa)
    off_df["sgRNA_repT"] = off_df["sgRNA"].apply(force_seed_repeat)
    off_df["sgRNA_repC"] = off_df["sgRNA"].apply(force_seed_block)

    guides_wt = off_df["sgRNA"].tolist()
    guides_tru = off_df["sgRNA_truncated"].tolist()
    guides_divT = off_df["sgRNA_divT"].tolist()
    guides_divC = off_df["sgRNA_divC"].tolist()
    guides_repT = off_df["sgRNA_repT"].tolist()
    guides_repC = off_df["sgRNA_repC"].tolist()
    off_targets = off_df["off_target"].tolist()
    on_targets = off_df["on_target_seq"].tolist()

    def fwd_pair(guides: list[str], intervention: dict | None = None):
        """Forward su (guides, off_targets) e (guides, on_targets)."""
        ctx_off = compute_gc_context_batch(guides, off_targets, device)
        ctx_on = compute_gc_context_batch(guides, on_targets, device)
        l_off, p_off = model_forward_batched(model, guides, off_targets, ctx_off, args.batch_size, intervention=intervention)
        l_on, p_on = model_forward_batched(model, guides, on_targets, ctx_on, args.batch_size, intervention=intervention)
        return (l_off, p_off), (l_on, p_on)

    # 7a-e. Forward
    print("Forward factual...")
    (logit_off_f, pam_off_f), (logit_on_f, pam_on_f) = fwd_pair(guides_wt)
    print("Forward truncation 5' (sequence intervention)...")
    (logit_off_t, pam_off_t), (logit_on_t, pam_on_t) = fwd_pair(guides_tru)
    print("Forward do(pos_14 = 0.0) (DAG node intervention)...")
    (logit_off_p14, pam_off_p14), (logit_on_p14, pam_on_p14) = fwd_pair(guides_wt, intervention={"pos_14": 0.0})
    print("Forward diversity ACGT (T) e AAAA (C) in pos 16-19 (sequence intervention)...")
    (logit_off_divT, pam_off_divT), (logit_on_divT, pam_on_divT) = fwd_pair(guides_divT)
    (logit_off_divC, pam_off_divC), (logit_on_divC, pam_on_divC) = fwd_pair(guides_divC)
    print("Forward repeat ATATATAT (T) e AAAATTTT (C) in pos 8-15 (sequence intervention)...")
    (logit_off_repT, pam_off_repT), (logit_on_repT, pam_on_repT) = fwd_pair(guides_repT)
    (logit_off_repC, pam_off_repC), (logit_on_repC, pam_on_repC) = fwd_pair(guides_repC)

    # 8. Predizioni factual — coerenti col modello in base alla modalità + shift
    off_df["pam_off_f"] = pam_off_f
    off_df["pam_on_f"] = pam_on_f
    off_df["y_pred_off_prob"] = model_pred_pct(logit_off_f, pam_off_f, args.pam_mode, assay_shift)
    off_df["y_pred_on_prob"] = model_pred_pct(logit_on_f, pam_on_f, args.pam_mode, assay_shift)

    # 9. Abduzione off-target (mode-aware + shift)
    off_df["U_off"] = abduct_U(
        np.asarray(off_df["y_obs_off_prob"].values),
        logit_off_f, pam_off_f,
        pam_mode=args.pam_mode, assay_shift=assay_shift,
    )
    U_off_arr = np.asarray(off_df["U_off"].values)

    # 10. Controfattuali off-target (con shift applicato a struct_logit_cf)
    cf = lambda l, p: counterfactual_prob_pct(l, p, U_off_arr, pam_mode=args.pam_mode, assay_shift=assay_shift)
    off_df["y_cf_off_tru_prob"] = cf(logit_off_t, pam_off_t)
    off_df["y_cf_off_p14_prob"] = cf(logit_off_p14, pam_off_p14)
    off_df["y_cf_off_divT_prob"] = cf(logit_off_divT, pam_off_divT)
    off_df["y_cf_off_divC_prob"] = cf(logit_off_divC, pam_off_divC)
    off_df["y_cf_off_repT_prob"] = cf(logit_off_repT, pam_off_repT)
    off_df["y_cf_off_repC_prob"] = cf(logit_off_repC, pam_off_repC)

    off_df["delta_off_tru"] = off_df["y_cf_off_tru_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_p14"] = off_df["y_cf_off_p14_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_divT"] = off_df["y_cf_off_divT_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_divC"] = off_df["y_cf_off_divC_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_repT"] = off_df["y_cf_off_repT_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_repC"] = off_df["y_cf_off_repC_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_divTC"] = off_df["y_cf_off_divT_prob"] - off_df["y_cf_off_divC_prob"]
    off_df["delta_off_repTC"] = off_df["y_cf_off_repT_prob"] - off_df["y_cf_off_repC_prob"]

    # 11. On-target: due regimi
    def model_pred(l, p):
        return model_pred_pct(l, p, args.pam_mode, assay_shift)

    if args.on_target_mode == "drop":
        off_df["U_on"] = np.nan
        off_df["y_cf_on_tru_prob"] = model_pred(logit_on_t, pam_on_t)
        off_df["y_cf_on_p14_prob"] = model_pred(logit_on_p14, pam_on_p14)
        off_df["y_cf_on_divT_prob"] = model_pred(logit_on_divT, pam_on_divT)
        off_df["y_cf_on_divC_prob"] = model_pred(logit_on_divC, pam_on_divC)
        off_df["y_cf_on_repT_prob"] = model_pred(logit_on_repT, pam_on_repT)
        off_df["y_cf_on_repC_prob"] = model_pred(logit_on_repC, pam_on_repC)
        baseline_on = off_df["y_pred_on_prob"]
    else:
        off_df["U_on"] = abduct_U(
            np.asarray(off_df["y_obs_on_prob"].values),
            logit_on_f, pam_on_f,
            pam_mode=args.pam_mode, assay_shift=assay_shift,
        )
        U_on_arr = np.asarray(off_df["U_on"].values)
        cf_on = lambda l, p: counterfactual_prob_pct(l, p, U_on_arr, pam_mode=args.pam_mode, assay_shift=assay_shift)
        off_df["y_cf_on_tru_prob"] = cf_on(logit_on_t, pam_on_t)
        off_df["y_cf_on_p14_prob"] = cf_on(logit_on_p14, pam_on_p14)
        off_df["y_cf_on_divT_prob"] = cf_on(logit_on_divT, pam_on_divT)
        off_df["y_cf_on_divC_prob"] = cf_on(logit_on_divC, pam_on_divC)
        off_df["y_cf_on_repT_prob"] = cf_on(logit_on_repT, pam_on_repT)
        off_df["y_cf_on_repC_prob"] = cf_on(logit_on_repC, pam_on_repC)
        baseline_on = off_df["y_obs_on_prob"]

    off_df["delta_on_tru"] = off_df["y_cf_on_tru_prob"] - baseline_on
    off_df["delta_on_p14"] = off_df["y_cf_on_p14_prob"] - baseline_on
    off_df["delta_on_divT"] = off_df["y_cf_on_divT_prob"] - baseline_on
    off_df["delta_on_divC"] = off_df["y_cf_on_divC_prob"] - baseline_on
    off_df["delta_on_repT"] = off_df["y_cf_on_repT_prob"] - baseline_on
    off_df["delta_on_repC"] = off_df["y_cf_on_repC_prob"] - baseline_on
    off_df["delta_on_divTC"] = off_df["y_cf_on_divT_prob"] - off_df["y_cf_on_divC_prob"]
    off_df["delta_on_repTC"] = off_df["y_cf_on_repT_prob"] - off_df["y_cf_on_repC_prob"]

    # 12. Salvataggio CSV
    keep_cols = [
        "name", "sgRNA", "off_target", "on_target_seq", "distance",
        "off_reads", "on_reads",
        "pam_off_f", "pam_on_f",
        "y_obs_off_prob", "y_pred_off_prob", "U_off",
        "y_obs_on_prob", "y_pred_on_prob", "U_on",
        "y_cf_off_tru_prob", "y_cf_on_tru_prob", "delta_off_tru", "delta_on_tru",
        "y_cf_off_p14_prob", "y_cf_on_p14_prob", "delta_off_p14", "delta_on_p14",
        "y_cf_off_divT_prob", "y_cf_off_divC_prob",
        "y_cf_on_divT_prob", "y_cf_on_divC_prob",
        "delta_off_divT", "delta_off_divC", "delta_off_divTC",
        "delta_on_divT", "delta_on_divC", "delta_on_divTC",
        "y_cf_off_repT_prob", "y_cf_off_repC_prob",
        "y_cf_on_repT_prob", "y_cf_on_repC_prob",
        "delta_off_repT", "delta_off_repC", "delta_off_repTC",
        "delta_on_repT", "delta_on_repC", "delta_on_repTC",
    ]
    # Suffisso al filename: include shift e filtro saturated per evitare sovrascritture
    suffix_parts = []
    if assay_shift != 0.0:
        suffix_parts.append(f"shift{assay_shift:+.2f}")
    if args.filter_saturated:
        suffix_parts.append("filtsat")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    out_csv = output_dir / f"{args.dataset}_batch_results{suffix}.csv"
    off_df[keep_cols].to_csv(out_csv, index=False)
    print(f"\nSalvato {out_csv}")

    # 13. Plot Pareto
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(off_df["delta_on_tru"], off_df["delta_off_tru"],
               alpha=0.30, color="steelblue", s=10, label="Truncation 5' (sequence)")
    ax.scatter(off_df["delta_on_p14"], off_df["delta_off_p14"],
               alpha=0.30, color="crimson", s=10, label="do(pos_14 = 0) (DAG node)")
    ax.scatter(off_df["delta_on_divTC"], off_df["delta_off_divTC"],
               alpha=0.30, color="forestgreen", s=10, label="Diversity ACGT vs AAAA, T-C contrast")
    ax.scatter(off_df["delta_on_repTC"], off_df["delta_off_repTC"],
               alpha=0.30, color="darkorange", s=10, label="Repeat ATATATAT vs AAAATTTT, T-C contrast")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Delta On-Target Probability (cf - baseline) [%]")
    ax.set_ylabel("Delta Off-Target Probability (cf - baseline) [%]")
    title_extra = f", b̂={assay_shift:+.3f}" if assay_shift != 0.0 else ""
    ax.set_title(f"Pareto Trade-Off Causale ({args.dataset}, pam_mode={args.pam_mode}{title_extra})\n"
                 f"Quadrante in basso-a-destra = ideale (off↓, on↑)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    pareto_path = output_dir / f"{args.dataset}_pareto{suffix}.png"
    plt.savefig(pareto_path, dpi=200)
    plt.close()
    print(f"Salvato {pareto_path}")

    # 14. Plot distribuzione U
    has_u_on = args.on_target_mode != "drop"
    n_panels = 2 if has_u_on else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    panels = [(axes[0, 0], "U_off", "steelblue", "Rumore Esogeno U_off")]
    if has_u_on:
        panels.append((axes[0, 1], "U_on", "darkorange", f"Rumore Esogeno U_on ({args.on_target_mode})"))
    for ax, col, color, title in panels:
        vals = off_df[col].dropna().values
        ax.hist(vals, bins=60, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--",
                   label=f"mean={np.mean(vals):+.3f}")
        ax.axvline(np.median(vals), color="black", linestyle=":",
                   label=f"median={np.median(vals):+.3f}")
        ax.set_xlabel(col)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    u_path = output_dir / f"{args.dataset}_U_distribution{suffix}.png"
    plt.savefig(u_path, dpi=200)
    plt.close()
    print(f"Salvato {u_path}")

    # 15. Sommario globale
    print("\n=== SOMMARIO GLOBALE (per coppia) ===")
    print(f"PAM mode:          {args.pam_mode}")
    print(f"Assay shift b̂:     {assay_shift:+.4f}  (source: {shift_source})")
    print(f"Coppie analizzate: {len(off_df)}")
    print(f"Guide uniche:      {off_df['name'].nunique()}")
    print(f"\npam_off  mean={off_df['pam_off_f'].mean():.3f}  std={off_df['pam_off_f'].std():.3f}")
    print(f"pam_on   mean={off_df['pam_on_f'].mean():.3f}  std={off_df['pam_on_f'].std():.3f}")
    print(f"\nU_off  mean={off_df['U_off'].mean():+.3f}  std={off_df['U_off'].std():.3f}  "
          f"median={off_df['U_off'].median():+.3f}")
    if has_u_on:
        print(f"U_on   mean={off_df['U_on'].mean():+.3f}  std={off_df['U_on'].std():.3f}  "
              f"median={off_df['U_on'].median():+.3f}")
    else:
        print("U_on   N/A (on-target mode = drop, nessuna abduzione)")

    interventions_summary = [
        ("Truncation 5' (sequence)", "delta_off_tru", "delta_on_tru"),
        ("do(pos_14 = 0) (DAG node)", "delta_off_p14", "delta_on_p14"),
        ("Diversity ACGT vs AAAA, T-C contrast", "delta_off_divTC", "delta_on_divTC"),
        ("Repeat ATATATAT vs AAAATTTT, T-C contrast", "delta_off_repTC", "delta_on_repTC"),
    ]
    for label, dcoff, dcon in interventions_summary:
        print(f"\n{label}:")
        print(f"  Delta off mean={off_df[dcoff].mean():+.2f}%  std={off_df[dcoff].std():.2f}")
        print(f"  Delta on  mean={off_df[dcon].mean():+.2f}%  std={off_df[dcon].std():.2f}")
        ideal = ((off_df[dcoff] < 0) & (off_df[dcon] >= -5)).sum()
        print(f"  Coppie nel quadrante ideale (Deltaoff<0 e Deltaon>=-5%): {ideal} ({100*ideal/len(off_df):.1f}%)")

    print("\n--- Diagnostica T vs C per interventi diversity/repeat (per coppia) ---")
    for prefix, label in [("div", "Diversity"), ("rep", "Repeat")]:
        for side in ("off", "on"):
            T_col = f"delta_{side}_{prefix}T"
            C_col = f"delta_{side}_{prefix}C"
            print(f"  {label} {side}-target:  "
                  f"delta_T mean={off_df[T_col].mean():+.2f}%  "
                  f"delta_C mean={off_df[C_col].mean():+.2f}%  "
                  f"contrast(T-C) mean={(off_df[T_col]-off_df[C_col]).mean():+.2f}%")

    # 16. Per-guida
    delta_cols = [
        "delta_off_tru", "delta_off_p14", "delta_off_divTC", "delta_off_repTC",
        "delta_on_tru", "delta_on_p14", "delta_on_divTC", "delta_on_repTC",
    ]
    u_cols = ["U_off"] + (["U_on"] if has_u_on else [])
    per_guide = off_df.groupby("name")[delta_cols + u_cols].median()

    print(f"\n=== SOMMARIO PER-GUIDA (mediana entro guida -> distribuzione su {len(per_guide)} guide) ===")
    print()
    summary = per_guide.describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
    print(summary.to_string())
    print()
    for label, dcoff, dcon in interventions_summary:
        n_g = len(per_guide)
        ideal_g = ((per_guide[dcoff] < 0) & (per_guide[dcon] >= -5)).sum()
        print(f"{label}: guide nel quadrante ideale (mediana Deltaoff<0 e Deltaon>=-5%): "
              f"{ideal_g}/{n_g} ({100*ideal_g/n_g:.1f}%)")

    per_guide_path = output_dir / f"{args.dataset}_per_guide_medians{suffix}.csv"
    per_guide.to_csv(per_guide_path)
    print(f"\nSalvato {per_guide_path}")


if __name__ == "__main__":
    main()
