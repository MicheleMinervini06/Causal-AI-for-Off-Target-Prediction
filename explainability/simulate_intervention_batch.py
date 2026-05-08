"""Batch counterfactual analysis: applica interventi fissi a tutte le coppie
(guide, off-target) di un dataset e produce CSV + plot Pareto + distribuzioni di U."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM


# ---------- utility numeriche vettorizzate ----------

def reads_to_prob(reads: np.ndarray, max_reads: np.ndarray, method: str = "log") -> np.ndarray:
    reads = np.maximum(0, reads).astype(np.float64)
    max_reads = np.maximum(reads, max_reads).astype(np.float64)
    if method == "log":
        p = np.log1p(reads) / np.log1p(max_reads)
    elif method == "linear":
        p = reads / max_reads
    else:
        raise ValueError(f"Metodo {method} non supportato.")
    return np.minimum(p * 100.0, 99.0)


def logit_from_prob(prob_pct: np.ndarray) -> np.ndarray:
    p = np.clip(prob_pct / 100.0, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gc_fraction(seq: str) -> float:
    return sum(1 for c in seq if c in "GC") / max(len(seq), 1)


def compute_gc_context_batch(guides: list[str], targets: list[str], device: torch.device) -> torch.Tensor:
    gc_sg = np.array([gc_fraction(g) for g in guides], dtype=np.float32)
    gc_tg = np.array([gc_fraction(t) for t in targets], dtype=np.float32)
    delta = gc_sg - gc_tg
    arr = np.stack([gc_sg, gc_tg, delta], axis=1)
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ---------- forward batched ----------

def model_logits(
    model: NeuralSCM,
    guides: list[str],
    targets: list[str],
    ctx: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    n = len(guides)
    logits = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            out = model(guides[i:j], targets[i:j], context_features=ctx[i:j])
            logits[i:j] = out["logit"].squeeze(-1).cpu().numpy()
    return logits


# ---------- interventi fissi ----------

def truncate_5p(guide: str) -> str:
    """Maschera le prime 2 basi (troncamento 5')."""
    return "NN" + guide[2:]


def mutate_pos15(guide: str) -> str:
    """Forza la 15esima base (indice 14) a 'A' — replica il single-pair."""
    return guide[:14] + "A" + guide[15:]


# ---------- pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["changeseq", "guideseq"], default="guideseq")
    parser.add_argument(
        "--model_path",
        default="experiments/results/Exp15_Positional_ExtendedOneCycle/neural_scm.pt",
    )
    parser.add_argument("--output_dir", default="explainability/batch_results")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--on-target-mode",
        choices=["drop", "per_run", "global_max"],
        default=None,
        help="Come gestire l'abduzione on-target. Default: per_run per guideseq, drop per changeseq",
    )
    args = parser.parse_args()

    if args.on_target_mode is None:
        args.on_target_mode = "per_run" if args.dataset == "guideseq" else "drop"
    print(f"On-target mode: {args.on_target_mode}")

    if args.dataset == "changeseq":
        csv_path = "data/raw/changeseq/CHANGEseq_positive.csv"
        reads_col = "CHANGEseq_reads"
    else:
        csv_path = "data/raw/guideseq/GUIDEseq_positive.csv"
        reads_col = "GUIDEseq_reads"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Caricamento modello (context_dim ricavato dal checkpoint)
    state_dict = torch.load(args.model_path, map_location=device)
    context_dim = 0
    if "context_net.0.weight" in state_dict:
        context_dim = state_dict["context_net.0.weight"].shape[1]

    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(
        encoder=encoder,
        architecture="positional_mlp",
        hidden_dim=8,
        context_dim=context_dim,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Modello caricato (context_dim={context_dim})")

    # 2. Dataset
    df = pd.read_csv(csv_path)
    print(f"Caricate {len(df)} righe da {csv_path}")

    # 3. Lookup on-target per name (riga distance=0 con max reads)
    on_rows = df[df["distance"] == 0].copy()
    on_lookup = (
        on_rows.sort_values(reads_col, ascending=False)
        .drop_duplicates("name")
        .set_index("name")[["offtarget_sequence", reads_col]]
        .rename(columns={"offtarget_sequence": "on_target_seq", reads_col: "on_reads"})
    )
    print(f"Guide con on-target di riferimento: {len(on_lookup)}")

    # 4. Off-target rows
    off_df = df[df["distance"] > 0].copy().join(on_lookup, on="name", how="inner")
    off_df["sgRNA"] = off_df["target"].str[:20]
    off_df["off_target"] = off_df["offtarget_sequence"]
    off_df["off_reads"] = off_df[reads_col]

    valid = (off_df["sgRNA"].str.len() == 20) & (off_df["off_target"].str.len() == 23) & (
        off_df["on_target_seq"].str.len() == 23
    )
    dropped = (~valid).sum()
    if dropped:
        print(f"[WARN] Scartate {dropped} righe per lunghezze incompatibili")
    off_df = off_df[valid].reset_index(drop=True)
    print(f"Coppie analizzabili: {len(off_df)}")

    # 5a. Probabilità osservate off-target (denominatore = on-target reads della stessa guida)
    off_df["y_obs_off_prob"] = reads_to_prob(off_df["off_reads"].values, off_df["on_reads"].values)
    off_df["y_obs_off_logit"] = logit_from_prob(off_df["y_obs_off_prob"].values)

    # 5b. Probabilità osservate on-target — dipende dalla modalità
    if args.on_target_mode == "drop":
        # Nessuna abduzione: useremo la predizione del modello come baseline
        off_df["y_obs_on_prob"] = np.nan
        off_df["y_obs_on_logit"] = np.nan
    elif args.on_target_mode == "per_run":
        if "run" not in off_df.columns:
            raise ValueError(f"Dataset {args.dataset} non ha colonna 'run', usa --on-target-mode drop o global_max")
        run_max = df.groupby("run")[reads_col].max().to_dict()
        off_df["run_max_reads"] = off_df["run"].map(run_max).astype(np.float64)
        off_df["y_obs_on_prob"] = reads_to_prob(off_df["on_reads"].values, off_df["run_max_reads"].values)
        off_df["y_obs_on_logit"] = logit_from_prob(off_df["y_obs_on_prob"].values)
        print(f"Run-level max reads: {run_max}")
    elif args.on_target_mode == "global_max":
        global_max = float(df[reads_col].max())
        off_df["y_obs_on_prob"] = reads_to_prob(
            off_df["on_reads"].values, np.full(len(off_df), global_max)
        )
        off_df["y_obs_on_logit"] = logit_from_prob(off_df["y_obs_on_prob"].values)
        print(f"Global max reads: {global_max:.0f}")

    # 6. Costruzione sequenze post-intervento
    off_df["sgRNA_truncated"] = off_df["sgRNA"].apply(truncate_5p)
    off_df["sgRNA_mutated"] = off_df["sgRNA"].apply(mutate_pos15)

    guides_wt = off_df["sgRNA"].tolist()
    guides_tru = off_df["sgRNA_truncated"].tolist()
    guides_mut = off_df["sgRNA_mutated"].tolist()
    off_targets = off_df["off_target"].tolist()
    on_targets = off_df["on_target_seq"].tolist()

    # 7. Sei forward pass (factual/truncated/mutated × off/on)
    print("Forward off-target × {factual, truncated, mutated}...")
    ctx_off_f = compute_gc_context_batch(guides_wt, off_targets, device)
    ctx_off_t = compute_gc_context_batch(guides_tru, off_targets, device)
    ctx_off_m = compute_gc_context_batch(guides_mut, off_targets, device)
    logit_off_f = model_logits(model, guides_wt, off_targets, ctx_off_f, args.batch_size)
    logit_off_t = model_logits(model, guides_tru, off_targets, ctx_off_t, args.batch_size)
    logit_off_m = model_logits(model, guides_mut, off_targets, ctx_off_m, args.batch_size)

    print("Forward on-target × {factual, truncated, mutated}...")
    ctx_on_f = compute_gc_context_batch(guides_wt, on_targets, device)
    ctx_on_t = compute_gc_context_batch(guides_tru, on_targets, device)
    ctx_on_m = compute_gc_context_batch(guides_mut, on_targets, device)
    logit_on_f = model_logits(model, guides_wt, on_targets, ctx_on_f, args.batch_size)
    logit_on_t = model_logits(model, guides_tru, on_targets, ctx_on_t, args.batch_size)
    logit_on_m = model_logits(model, guides_mut, on_targets, ctx_on_m, args.batch_size)

    # 8. Abduzione off-target (sempre attiva)
    off_df["y_pred_off_prob"] = sigmoid(logit_off_f) * 100
    off_df["y_pred_on_prob"] = sigmoid(logit_on_f) * 100
    off_df["U_off"] = off_df["y_obs_off_logit"].values - logit_off_f
    off_df["y_cf_off_tru_prob"] = sigmoid(logit_off_t + off_df["U_off"].values) * 100
    off_df["y_cf_off_mut_prob"] = sigmoid(logit_off_m + off_df["U_off"].values) * 100
    off_df["delta_off_tru"] = off_df["y_cf_off_tru_prob"] - off_df["y_obs_off_prob"]
    off_df["delta_off_mut"] = off_df["y_cf_off_mut_prob"] - off_df["y_obs_off_prob"]

    # 9. On-target: due regimi distinti
    if args.on_target_mode == "drop":
        # Nessuna abduzione. Baseline = y_pred_on, controfattuale = pure model con intervento.
        # delta_on = "credenza del modello sull'effetto dell'intervento"
        off_df["U_on"] = np.nan
        off_df["y_cf_on_tru_prob"] = sigmoid(logit_on_t) * 100
        off_df["y_cf_on_mut_prob"] = sigmoid(logit_on_m) * 100
        off_df["delta_on_tru"] = off_df["y_cf_on_tru_prob"] - off_df["y_pred_on_prob"]
        off_df["delta_on_mut"] = off_df["y_cf_on_mut_prob"] - off_df["y_pred_on_prob"]
    else:
        # Abduzione classica: U_on = y_obs_on_logit - y_pred_on_logit
        off_df["U_on"] = off_df["y_obs_on_logit"].values - logit_on_f
        off_df["y_cf_on_tru_prob"] = sigmoid(logit_on_t + off_df["U_on"].values) * 100
        off_df["y_cf_on_mut_prob"] = sigmoid(logit_on_m + off_df["U_on"].values) * 100
        off_df["delta_on_tru"] = off_df["y_cf_on_tru_prob"] - off_df["y_obs_on_prob"]
        off_df["delta_on_mut"] = off_df["y_cf_on_mut_prob"] - off_df["y_obs_on_prob"]

    # 11. Salvataggio CSV (sottoinsieme leggibile delle colonne)
    keep_cols = [
        "name", "sgRNA", "off_target", "on_target_seq", "distance",
        "off_reads", "on_reads",
        "y_obs_off_prob", "y_pred_off_prob", "U_off",
        "y_obs_on_prob", "y_pred_on_prob", "U_on",
        "y_cf_off_tru_prob", "y_cf_off_mut_prob",
        "y_cf_on_tru_prob", "y_cf_on_mut_prob",
        "delta_off_tru", "delta_off_mut", "delta_on_tru", "delta_on_mut",
    ]
    out_csv = output_dir / f"{args.dataset}_batch_results.csv"
    off_df[keep_cols].to_csv(out_csv, index=False)
    print(f"\nSalvato {out_csv}")

    # 12. Plot Pareto trade-off
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(off_df["delta_on_tru"], off_df["delta_off_tru"],
               alpha=0.35, color="steelblue", s=10, label="Truncation 5' (NN+...)")
    ax.scatter(off_df["delta_on_mut"], off_df["delta_off_mut"],
               alpha=0.35, color="crimson", s=10, label="Mutation pos 15 -> A")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Delta On-Target Probability (cf - obs) [%]")
    ax.set_ylabel("Delta Off-Target Probability (cf - obs) [%]")
    ax.set_title(f"Pareto Trade-Off Causale ({args.dataset})\n"
                 f"Quadrante in basso-a-destra = ideale (off↓, on↑)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    pareto_path = output_dir / f"{args.dataset}_pareto.png"
    plt.savefig(pareto_path, dpi=200)
    plt.close()
    print(f"Salvato {pareto_path}")

    # 13. Plot distribuzione del rumore U
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
    u_path = output_dir / f"{args.dataset}_U_distribution.png"
    plt.savefig(u_path, dpi=200)
    plt.close()
    print(f"Salvato {u_path}")

    # 14. Sommario su stdout — aggregato globale (per coppia)
    print("\n=== SOMMARIO GLOBALE (per coppia) ===")
    print(f"Coppie analizzate: {len(off_df)}")
    print(f"Guide uniche:      {off_df['name'].nunique()}")
    print(f"\nU_off  mean={off_df['U_off'].mean():+.3f}  std={off_df['U_off'].std():.3f}  "
          f"median={off_df['U_off'].median():+.3f}")
    if has_u_on:
        print(f"U_on   mean={off_df['U_on'].mean():+.3f}  std={off_df['U_on'].std():.3f}  "
              f"median={off_df['U_on'].median():+.3f}")
    else:
        print("U_on   N/A (on-target mode = drop, nessuna abduzione)")

    for label, dcoff, dcon in [
        ("Truncation 5'", "delta_off_tru", "delta_on_tru"),
        ("Mutation pos15->A", "delta_off_mut", "delta_on_mut"),
    ]:
        print(f"\n{label}:")
        print(f"  Delta off mean={off_df[dcoff].mean():+.2f}%  std={off_df[dcoff].std():.2f}")
        print(f"  Delta on  mean={off_df[dcon].mean():+.2f}%  std={off_df[dcon].std():.2f}")
        ideal = ((off_df[dcoff] < 0) & (off_df[dcon] >= -5)).sum()
        print(f"  Coppie nel quadrante ideale (Deltaoff<0 e Deltaon>=-5%): {ideal} ({100*ideal/len(off_df):.1f}%)")

    # 15. Sommario per-guida: mediana entro guida, poi statistiche su quei valori
    delta_cols = ["delta_off_tru", "delta_off_mut", "delta_on_tru", "delta_on_mut"]
    u_cols = ["U_off"] + (["U_on"] if has_u_on else [])
    per_guide = off_df.groupby("name")[delta_cols + u_cols].median()

    print(f"\n=== SOMMARIO PER-GUIDA (mediana entro guida -> distribuzione su {len(per_guide)} guide) ===")
    print("(mitiga il bias delle guide con molti off-target nella media globale)")
    print()
    summary = per_guide.describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
    print(summary.to_string())
    print()
    for label, dcoff, dcon in [
        ("Truncation 5'", "delta_off_tru", "delta_on_tru"),
        ("Mutation pos15->A", "delta_off_mut", "delta_on_mut"),
    ]:
        n_g = len(per_guide)
        ideal_g = ((per_guide[dcoff] < 0) & (per_guide[dcon] >= -5)).sum()
        print(f"{label}: guide nel quadrante ideale (mediana Deltaoff<0 e Deltaon>=-5%): "
              f"{ideal_g}/{n_g} ({100*ideal_g/n_g:.1f}%)")

    per_guide_path = output_dir / f"{args.dataset}_per_guide_medians.csv"
    per_guide.to_csv(per_guide_path)
    print(f"\nSalvato {per_guide_path}")


if __name__ == "__main__":
    main()
