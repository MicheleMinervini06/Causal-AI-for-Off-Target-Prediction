"""Thesis-ready figures generated with matplotlib (the bar charts and the
forest plot live directly in the LaTeX source via pgfplots; the figures
below require matplotlib because they involve either a checkpoint read
or a per-row simulation sweep that does not fit in a static .tex table).

  - Figure A: per-position effective weights |w_i| of the adopted model
              (Exp30), with the seed / non-seed / PAM-proximal regional
              partition shaded for reference.
  - Figure B: U_off stratified by mismatch distance, CHANGEseq vs GUIDEseq.
  - Figure C: per-position interventional sensitivity, the predicted
              activity drop |Delta Pr(Y=1)| under do(P_i = v) on a fixed
              strong penalty value v, averaged over a sample of the two
              evaluation pools.

Colour palette is kept consistent with the pgfplots figures inserted in
4_results.tex: a dark navy blue for the in-vitro / structural quantities,
a warm orange for the in-vivo / cross-assay quantities.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM


# ---------- paths ----------

RESULTS = Path("experiments/results")
OUT_DIR = Path("explainability/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADOPTED_RUN_DIR = "Exp30_Ablation_EncoderBiologicalMismatch_UseEncoder"
FINAL_MODEL = RESULTS / ADOPTED_RUN_DIR / "neural_scm.pt"

GUIDESEQ_BATCH  = Path("explainability/batch_results/guideseq_batch_results.csv")
CHANGESEQ_BATCH = Path("explainability/batch_results/changeseq_batch_results.csv")

# Palette aligned with the pgfplots tints in 4_results.tex (saturated, vivid).
# In-vitro / structural quantities: LaTeX blue family.
# In-vivo / cross-assay quantities: LaTeX orange family.
COLOR_INVITRO_FILL = "#7373FF"   # pgfplots blue!55  = 0.55*blue + 0.45*white
COLOR_INVITRO_EDGE = "#0000B3"   # pgfplots blue!70!black = 0.7*blue + 0.3*black
COLOR_INVIVO_FILL  = "#FFA64D"   # pgfplots orange!70 = 0.7*orange + 0.3*white
COLOR_INVIVO_EDGE  = "#D96C00"   # pgfplots orange!85!black = 0.85*orange + 0.15*black

# Region shading: light tints in the same blue / amber / salmon family.
COLOR_REGION_NONSEED = "#E8E8E8"
COLOR_REGION_SEED    = "#FFEBA8"
COLOR_REGION_PAMPROX = "#FFCDAA"


# ============================================================================
# FIGURE A — per-position effective weights of the adopted model
# ============================================================================

def _load_adopted_model() -> NeuralSCM:
    state = torch.load(FINAL_MODEL, map_location="cpu")
    ctx_dim = state["context_net.0.weight"].shape[1] if "context_net.0.weight" in state else 0
    pos_in = state["pos_node.0.weight"].shape[1] if "pos_node.0.weight" in state else 4
    use_enc = (pos_in != 4)
    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(
        encoder=encoder,
        architecture="positional_mlp",
        hidden_dim=8,
        context_dim=ctx_dim,
        positional_use_encoder=use_enc,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def fig_w_pos() -> Path:
    model = _load_adopted_model()
    # w_pos_eff = -softplus(w_pos)  →  effective per-position penalty (non-positive).
    # We plot |w_pos_eff| as the importance magnitude.
    w_pos_eff = -F.softplus(model.w_pos).detach().cpu().numpy()
    w_mag = np.abs(w_pos_eff)
    positions = np.arange(1, 21)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 5.5))

    # region shading: non-seed 1-8, seed 9-16, PAM-proximal 17-20
    ax.axvspan(0.5,  8.5,  color=COLOR_REGION_NONSEED, alpha=0.55, zorder=0,
               label="Non-seed ($i = 1$–$8$)")
    ax.axvspan(8.5,  16.5, color=COLOR_REGION_SEED,    alpha=0.55, zorder=0,
               label="Seed ($i = 9$–$16$)")
    ax.axvspan(16.5, 20.5, color=COLOR_REGION_PAMPROX, alpha=0.55, zorder=0,
               label="PAM-proximal ($i = 17$–$20$)")

    bars = ax.bar(positions, w_mag,
                  color=COLOR_INVITRO_FILL,
                  edgecolor=COLOR_INVITRO_EDGE,
                  linewidth=0.8, zorder=2)
    for bar, v in zip(bars, w_mag):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(w_mag)*0.012,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                color=COLOR_INVITRO_EDGE)

    ax.set_xlabel("Spacer position ($5' \\to 3'$, PAM at $3'$ end)")
    ax.set_ylabel("$|w_{\\mathrm{pos},i}|$  (effective per-position penalty)")
    ax.set_xticks(positions)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, max(w_mag) * 1.18)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    plt.tight_layout()
    path = OUT_DIR / "thesis_fig2_w_pos.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ============================================================================
# FIGURE B — U_off stratified by mismatch distance
# ============================================================================

def fig_u_off_by_distance() -> Path:
    gs = pd.read_csv(GUIDESEQ_BATCH)
    cs = pd.read_csv(CHANGESEQ_BATCH)

    def agg(df):
        return df.groupby("distance")["U_off"].agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            n="size",
        ).reset_index()

    gs_g = agg(gs)
    cs_g = agg(cs)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", zorder=1)

    # CHANGE-seq (in vitro) — navy blue
    ax.fill_between(cs_g["distance"], cs_g["q25"], cs_g["q75"],
                    color=COLOR_INVITRO_FILL, alpha=0.30, zorder=2)
    ax.plot(cs_g["distance"], cs_g["median"], marker="o", markersize=10,
            linewidth=2.5, color=COLOR_INVITRO_EDGE,
            label=f"CHANGE-seq (in vitro, n={len(cs)})", zorder=3)

    # GUIDE-seq (in vivo) — warm orange
    ax.fill_between(gs_g["distance"], gs_g["q25"], gs_g["q75"],
                    color=COLOR_INVIVO_FILL, alpha=0.35, zorder=2)
    ax.plot(gs_g["distance"], gs_g["median"], marker="s", markersize=10,
            linewidth=2.5, color=COLOR_INVIVO_EDGE,
            label=f"GUIDE-seq (in vivo, n={len(gs)})", zorder=3)

    ax.set_xlabel("Number of mismatches (distance from on-target)")
    ax.set_ylabel("$\\hat U_{\\mathrm{off}}$ (median $\\pm$ IQR, logit scale)")
    ax.set_xticks(sorted(set(cs_g["distance"]).union(gs_g["distance"])))
    ax.legend(loc="upper left", fontsize=11, frameon=True)

    plt.tight_layout()
    path = OUT_DIR / "thesis_fig3_u_off_by_distance.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ---------- main ----------

def main():
    print(f"Generating thesis-ready figures (adopted run: {ADOPTED_RUN_DIR})...\n")
    for name, fn in [
        ("Figure A — per-position weights",        fig_w_pos),
        ("Figure B — U_off by distance",           fig_u_off_by_distance),
    ]:
        try:
            out = fn()
            print(f"  [OK]  {name:40s} -> {out}")
        except Exception as e:
            print(f"  [ERR] {name:40s} -> {e}")
            raise


if __name__ == "__main__":
    main()
