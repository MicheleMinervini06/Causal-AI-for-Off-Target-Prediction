"""P1 calibration — assay-shift estimation via small calibration set.

Stima del parametro scalare `b̂_shift` che corregge le predizioni di un modello
addestrato su un assay (default: CHANGE-seq, E=vitro) quando applicato a un assay
target (default: GUIDE-seq, E=vivo). Sotto l'ipotesi SMS (Sparse Mechanism Shift),
il gap tra i due assay è dominato da uno shift logit additivo che assorbe la
componente "rumore esogeno" assay-specifica.

Metodologia:
  1. Scelta uniforme di N_calib guide (sweep su N_calib = {1, 2, 5, 10, 20, ...}).
  2. Calcolo di b̂ = median_over_pairs( logit(y_obs) − model_logit ) sul calibration set.
  3. Sul rest delle guide (eval set): predizioni con e senza shift.
  4. Bootstrap (con seed) su scelta casuale delle guide → stima di varianza di b̂.
  5. Confronto con stima empirica F9 (median di U_off da simulate_intervention_batch.py)
     se disponibile via --baseline-shift.

Output:
  - JSON con summary statistics, sweep, bootstrap, e campo `selected_shift`
    consumabile da `simulate_intervention_batch.py --assay-shift-from <json>`.
  - CSV con i dettagli per ogni bootstrap.
  - Plot di convergenza N_calib → b̂.
  - Plot di calibrazione (y_pred vs y_obs) prima/dopo shift.

NOTE su AUPRC/AUROC:
  Lo shift è una trasformazione monotona del logit → AUPRC e AUROC sui dataset
  positivi-only NON CAMBIANO. Le metriche che cambiano sono di tipo calibration:
  bias medio, MAE, Brier-like su valori continui, e la *forma* della curva
  y_pred vs y_obs. Vedi sezione "Cosa misura P1c" del findings.
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
    EPS,
    build_offtarget_dataframe,
    compute_gc_context_batch,
    filter_saturated_pairs,
    load_neural_scm,
    load_positive_dataset,
    logit_from_prob_pct,
    model_forward_batched,
    resolve_on_target_mode,
    sigmoid,
)


# ---------- core calibration ----------

def compute_shift_from_pairs(
    logit_pred: np.ndarray,
    y_obs_prob_pct: np.ndarray,
    pam_gate: np.ndarray | None = None,
    pam_mode: str = "additive",
) -> float:
    """Stima `b̂` = median(L_true − L_pred) sul set di calibrazione.

    additive:        L_true = logit(y_obs)                 ;  L_pred = struct_logit
    multiplicative:  L_true = logit(y_obs / pam_gate)      ;  L_pred = struct_logit

    Il median è preferito alla media per robustezza vs outlier (es. coppie con
    y_obs vicino a 99% saturation cap).
    """
    if pam_mode == "additive":
        L_true = logit_from_prob_pct(y_obs_prob_pct)
    elif pam_mode == "multiplicative":
        if pam_gate is None:
            raise ValueError("pam_gate richiesto per multiplicative mode")
        p_unit = np.clip(y_obs_prob_pct / 100.0 / pam_gate, EPS, 1.0 - EPS)
        L_true = np.log(p_unit / (1.0 - p_unit))
    else:
        raise ValueError(f"pam_mode non riconosciuto: {pam_mode}")
    return float(np.median(L_true - logit_pred))


def calibration_metrics(
    logit_pred: np.ndarray,
    y_obs_prob_pct: np.ndarray,
    shift: float = 0.0,
    pam_gate: np.ndarray | None = None,
    pam_mode: str = "additive",
) -> dict[str, float]:
    """Metriche di calibrazione tra y_pred (eventualmente shiftato) e y_obs.

    Restituisce un dict con:
      - bias_pct:   mean(y_pred - y_obs)        [in punti percentuali]
      - mae_pct:    mean(|y_pred - y_obs|)
      - rmse_pct:   sqrt(mean((y_pred - y_obs)^2))
      - logit_bias: median(L_pred - L_true)     [zero = perfect calibration in logit space]
      - n:          numero di coppie
    """
    if pam_mode == "additive":
        y_pred_pct = sigmoid(logit_pred + shift) * 100.0
    elif pam_mode == "multiplicative":
        if pam_gate is None:
            raise ValueError("pam_gate richiesto per multiplicative mode")
        y_pred_pct = pam_gate * sigmoid(logit_pred + shift) * 100.0
    else:
        raise ValueError(f"pam_mode non riconosciuto: {pam_mode}")

    diff = y_pred_pct - y_obs_prob_pct  # in punti percentuali
    L_pred_shifted = logit_pred + shift
    if pam_mode == "additive":
        L_true = logit_from_prob_pct(y_obs_prob_pct)
    else:
        p_unit = np.clip(y_obs_prob_pct / 100.0 / pam_gate, EPS, 1.0 - EPS)
        L_true = np.log(p_unit / (1.0 - p_unit))

    return {
        "bias_pct": float(np.mean(diff)),
        "mae_pct": float(np.mean(np.abs(diff))),
        "rmse_pct": float(np.sqrt(np.mean(diff ** 2))),
        "logit_bias": float(np.median(L_pred_shifted - L_true)),
        "n": int(len(diff)),
    }


# ---------- pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["changeseq", "guideseq"], default="guideseq",
        help="Assay target di calibrazione (default: guideseq = E=vivo).",
    )
    parser.add_argument(
        "--model_path",
        default="experiments/results/Exp18_Positional_AdditivePAM/neural_scm.pt",
    )
    parser.add_argument("--pam-mode", choices=["additive", "multiplicative"], default="additive")
    parser.add_argument(
        "--n-calib", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20],
        help="Sweep su numero di guide di calibrazione.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=200,
        help="Numero di campionamenti bootstrap per ogni N_calib.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selected-n", type=int, default=5,
        help="N_calib da usare nel campo JSON 'selected_shift' (per uso downstream).",
    )
    parser.add_argument(
        "--baseline-shift", type=float, default=None,
        help="Stima F9 empirica (es. median(U_off_vitro) − median(U_off_vivo)) per confronto.",
    )
    parser.add_argument(
        "--on-target-mode", choices=["drop", "per_run", "global_max"], default=None,
    )
    parser.add_argument(
        "--filter-saturated",
        action="store_true",
        help="Rimuove coppie saturated (off_reads >= on_reads) prima del calcolo di b̂. "
             "Coerente col regime di training di Exp20+ (vedi F23 in findings.md).",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="explainability/calibration_results")
    args = parser.parse_args()

    args.on_target_mode = resolve_on_target_mode(args.on_target_mode, args.dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print(f"Dataset target:      {args.dataset}")
    print(f"PAM mode:            {args.pam_mode}")
    print(f"On-target mode:      {args.on_target_mode}")
    print(f"Model:               {args.model_path}")
    print(f"N_calib sweep:       {args.n_calib}")
    print(f"Bootstrap n:         {args.n_bootstrap}")
    print(f"Selected N for JSON: {args.selected_n}")
    print(f"Filter saturated:    {args.filter_saturated}")
    print()

    # 1. Modello
    model, device, _ = load_neural_scm(args.model_path, pam_mode=args.pam_mode)
    print(f"Device: {device}")

    # 2. Dataset
    df, reads_col = load_positive_dataset(args.dataset)
    print(f"Caricate {len(df)} righe dal dataset {args.dataset}")
    off_df = build_offtarget_dataframe(df, reads_col, on_target_mode=args.on_target_mode)

    # 2b. Optional: filtra coppie saturated (coerente con training di Exp20+)
    if args.filter_saturated:
        off_df, _ = filter_saturated_pairs(off_df, verbose=True)
        if len(off_df) == 0:
            raise RuntimeError("No pairs left after --filter-saturated. Aborting.")

    # 3. Forward factual una sola volta (riusato in tutti i bootstrap)
    print("Forward factual su tutte le coppie...")
    guides = off_df["sgRNA"].tolist()
    off_targets = off_df["off_target"].tolist()
    ctx = compute_gc_context_batch(guides, off_targets, device)
    logit_off, pam_off = model_forward_batched(model, guides, off_targets, ctx, args.batch_size)
    off_df["logit_off_factual"] = logit_off
    off_df["pam_off_factual"] = pam_off

    all_guides = off_df["name"].unique().tolist()
    n_total = len(all_guides)
    print(f"Guide totali: {n_total}")
    print()

    # 4. Bootstrap sweep
    bootstrap_records = []  # CSV details per bootstrap
    sweep_summary = []      # JSON summary per N_calib

    for n_calib in args.n_calib:
        if n_calib >= n_total:
            print(f"[WARN] Saltato N_calib={n_calib} (≥ n_total={n_total})")
            continue

        shifts = []
        metrics_uncal = []
        metrics_cal = []

        for boot_idx in range(args.n_bootstrap):
            # Campionamento guide di calibrazione (senza rimpiazzo entro un bootstrap)
            calib_guides = rng.choice(all_guides, size=n_calib, replace=False)
            calib_mask = off_df["name"].isin(calib_guides)
            eval_mask = ~calib_mask

            calib_set = off_df[calib_mask]
            eval_set = off_df[eval_mask]

            if len(eval_set) == 0:
                continue

            # Stima shift sul calibration set
            shift = compute_shift_from_pairs(
                logit_pred=np.asarray(calib_set["logit_off_factual"].values),
                y_obs_prob_pct=np.asarray(calib_set["y_obs_off_prob"].values),
                pam_gate=np.asarray(calib_set["pam_off_factual"].values),
                pam_mode=args.pam_mode,
            )
            shifts.append(shift)

            # Metriche sull'eval set (no shift)
            m_uncal = calibration_metrics(
                logit_pred=np.asarray(eval_set["logit_off_factual"].values),
                y_obs_prob_pct=np.asarray(eval_set["y_obs_off_prob"].values),
                shift=0.0,
                pam_gate=np.asarray(eval_set["pam_off_factual"].values),
                pam_mode=args.pam_mode,
            )
            metrics_uncal.append(m_uncal)

            # Metriche con shift
            m_cal = calibration_metrics(
                logit_pred=np.asarray(eval_set["logit_off_factual"].values),
                y_obs_prob_pct=np.asarray(eval_set["y_obs_off_prob"].values),
                shift=shift,
                pam_gate=np.asarray(eval_set["pam_off_factual"].values),
                pam_mode=args.pam_mode,
            )
            metrics_cal.append(m_cal)

            bootstrap_records.append({
                "n_calib": n_calib,
                "boot_idx": boot_idx,
                "shift": shift,
                "bias_pct_uncal": m_uncal["bias_pct"],
                "bias_pct_cal": m_cal["bias_pct"],
                "mae_pct_uncal": m_uncal["mae_pct"],
                "mae_pct_cal": m_cal["mae_pct"],
                "rmse_pct_uncal": m_uncal["rmse_pct"],
                "rmse_pct_cal": m_cal["rmse_pct"],
                "logit_bias_uncal": m_uncal["logit_bias"],
                "logit_bias_cal": m_cal["logit_bias"],
                "n_eval_pairs": m_uncal["n"],
            })

        shifts = np.asarray(shifts, dtype=np.float64)
        ci_lo, ci_med, ci_hi = np.percentile(shifts, [2.5, 50.0, 97.5]).tolist()

        def agg(key: str, source: list[dict]) -> dict[str, float]:
            vals = np.asarray([m[key] for m in source], dtype=np.float64)
            return {
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "std": float(vals.std()),
            }

        sweep_summary.append({
            "n_calib": n_calib,
            "n_bootstrap_realized": len(shifts),
            "b_shift": {
                "mean": float(shifts.mean()),
                "median": ci_med,
                "std": float(shifts.std()),
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
            },
            "metrics_uncal": {
                "bias_pct": agg("bias_pct", metrics_uncal),
                "mae_pct": agg("mae_pct", metrics_uncal),
                "rmse_pct": agg("rmse_pct", metrics_uncal),
                "logit_bias": agg("logit_bias", metrics_uncal),
            },
            "metrics_cal": {
                "bias_pct": agg("bias_pct", metrics_cal),
                "mae_pct": agg("mae_pct", metrics_cal),
                "rmse_pct": agg("rmse_pct", metrics_cal),
                "logit_bias": agg("logit_bias", metrics_cal),
            },
        })

        print(f"N_calib={n_calib:3d} | b̂ median={ci_med:+.3f} CI95=[{ci_lo:+.3f}, {ci_hi:+.3f}] | "
              f"bias_pct uncal→cal: {agg('bias_pct', metrics_uncal)['mean']:+.2f}% → {agg('bias_pct', metrics_cal)['mean']:+.2f}% | "
              f"mae uncal→cal: {agg('mae_pct', metrics_uncal)['mean']:.2f}% → {agg('mae_pct', metrics_cal)['mean']:.2f}%")

    # 5. Stima full-data per confronto (uses ALL guides for calibration)
    print()
    full_shift = compute_shift_from_pairs(
        logit_pred=np.asarray(off_df["logit_off_factual"].values),
        y_obs_prob_pct=np.asarray(off_df["y_obs_off_prob"].values),
        pam_gate=np.asarray(off_df["pam_off_factual"].values),
        pam_mode=args.pam_mode,
    )
    print(f"Full-data shift (no bootstrap, all guides as calibration): b̂={full_shift:+.3f}")

    # 6. Confronto con baseline F9 (se fornito)
    if args.baseline_shift is not None:
        diff = full_shift - args.baseline_shift
        print(f"Baseline F9 shift fornito:  {args.baseline_shift:+.3f}")
        print(f"Differenza b̂(full) − baseline: {diff:+.3f}")

    # 7. Seleziona shift per JSON downstream
    selected_entry = next(
        (s for s in sweep_summary if s["n_calib"] == args.selected_n),
        None,
    )
    selected_shift = (
        selected_entry["b_shift"]["median"] if selected_entry is not None else full_shift
    )
    print(f"\nSelected shift for downstream (N_calib={args.selected_n}): b̂={selected_shift:+.3f}")

    # Suffisso al filename quando il filtro è applicato
    file_suffix = "_filtsat" if args.filter_saturated else ""

    # 8. JSON output
    json_path = output_dir / f"{args.dataset}_calibration{file_suffix}.json"
    payload = {
        "model_path": str(args.model_path),
        "pam_mode": args.pam_mode,
        "dataset": args.dataset,
        "on_target_mode": args.on_target_mode,
        "selected_n_calib": args.selected_n,
        "selected_shift": selected_shift,
        "full_data_shift": full_shift,
        "baseline_shift_provided": args.baseline_shift,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "n_guides_total": n_total,
        "n_pairs_total": int(len(off_df)),
        "sweep": sweep_summary,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSalvato {json_path}")

    # 9. CSV con dettagli bootstrap
    csv_path = output_dir / f"{args.dataset}_calibration_bootstrap{file_suffix}.csv"
    pd.DataFrame(bootstrap_records).to_csv(csv_path, index=False)
    print(f"Salvato {csv_path}")

    # 10. Plot di convergenza
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    ns = [s["n_calib"] for s in sweep_summary]
    medians = [s["b_shift"]["median"] for s in sweep_summary]
    ci_lo = [s["b_shift"]["ci95_lo"] for s in sweep_summary]
    ci_hi = [s["b_shift"]["ci95_hi"] for s in sweep_summary]
    ax.plot(ns, medians, marker="o", color="steelblue", label="b̂ median (bootstrap)")
    ax.fill_between(ns, ci_lo, ci_hi, color="steelblue", alpha=0.20, label="95% CI bootstrap")
    ax.axhline(full_shift, color="black", linestyle="--", linewidth=1,
               label=f"b̂ full-data = {full_shift:+.3f}")
    if args.baseline_shift is not None:
        ax.axhline(args.baseline_shift, color="crimson", linestyle=":", linewidth=1,
                   label=f"Baseline F9 = {args.baseline_shift:+.3f}")
    ax.set_xlabel("N calibration guides")
    ax.set_ylabel("b̂ shift  (logit-space)")
    ax.set_title(f"Assay-shift estimator convergence ({args.dataset}, pam_mode={args.pam_mode})")
    ax.legend(loc="best")
    ax.set_xscale("log")
    plt.tight_layout()
    conv_path = output_dir / f"{args.dataset}_convergence{file_suffix}.png"
    plt.savefig(conv_path, dpi=200)
    plt.close()
    print(f"Salvato {conv_path}")

    # 11. Plot di calibration curve (y_pred vs y_obs, before/after shift)
    if args.pam_mode == "additive":
        y_pred_uncal = sigmoid(off_df["logit_off_factual"].values) * 100.0
        y_pred_cal = sigmoid(off_df["logit_off_factual"].values + selected_shift) * 100.0
    else:
        y_pred_uncal = off_df["pam_off_factual"].values * sigmoid(off_df["logit_off_factual"].values) * 100.0
        y_pred_cal = off_df["pam_off_factual"].values * sigmoid(off_df["logit_off_factual"].values + selected_shift) * 100.0

    y_obs = off_df["y_obs_off_prob"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, title in [
        (axes[0], y_pred_uncal, "Before shift (uncalibrated)"),
        (axes[1], y_pred_cal, f"After shift b̂={selected_shift:+.3f} (calibrated)"),
    ]:
        ax.scatter(y_obs, y_pred, alpha=0.20, s=8, color="steelblue")
        lim = max(np.max(y_obs), np.max(y_pred))
        ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--", label="y=x")
        bias = np.mean(y_pred - y_obs)
        mae = np.mean(np.abs(y_pred - y_obs))
        ax.set_xlabel("y_obs (%)")
        ax.set_ylabel("y_pred (%)")
        ax.set_title(f"{title}\nbias={bias:+.2f}%  mae={mae:.2f}%")
        ax.legend()
    plt.tight_layout()
    cal_path = output_dir / f"{args.dataset}_calibration_curve{file_suffix}.png"
    plt.savefig(cal_path, dpi=200)
    plt.close()
    print(f"Salvato {cal_path}")

    # 12. Sommario finale stdout
    print("\n=== SOMMARIO FINALE ===")
    print(f"Modello: {args.model_path}")
    print(f"Dataset target: {args.dataset}  (pam_mode={args.pam_mode})")
    print(f"Full-data b̂:      {full_shift:+.4f}")
    print(f"Selected b̂ (N={args.selected_n}): {selected_shift:+.4f}")
    if args.baseline_shift is not None:
        print(f"Baseline F9 b̂:    {args.baseline_shift:+.4f}")
        print(f"Δ vs baseline:    {(full_shift - args.baseline_shift):+.4f}")
    print()
    print(f"Per applicarlo a simulate_intervention_batch.py:")
    print(f"  python explainability/simulate_intervention_batch.py \\")
    print(f"     --dataset {args.dataset} --assay-shift-from {json_path}")


if __name__ == "__main__":
    main()
