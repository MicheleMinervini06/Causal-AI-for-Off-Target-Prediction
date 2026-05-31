"""Causal DAG figure for Chapter 3 of the thesis.

This is the BIOLOGICAL CAUSAL GRAPH that the Neural SCM is asked to
instantiate. It is not the network architecture per se; it is the
structural template from which the architecture is derived
(Section 3.1 of the thesis), then formalized as a Structural Causal
Model in Section 3.2.

Reflects the final adopted model (positional_mlp + additive PAM,
Exp20 in findings.md):

  - 20 positional penalty nodes P_0..P_19 (one per protospacer position),
    indices following the convention in models/deep/neural_scm.py:
    P_0 is the most PAM-distal (5' end), P_19 the most PAM-proximal.
  - Regional groupings (non-seed, seed, PAM-proximal) are shown as
    DERIVED VIEWS via brackets, not as independent causal nodes
    (Section 3.1 of the thesis).
  - PAM compatibility g_pam = psi(X_pam) enters ADDITIVELY into the
    structural logit (Section 3.2.2: additive vs multiplicative choice
    is discussed and motivated there).
  - Exogenous noise U enters additively at the final logit, independent
    of X under the Independent Causal Mechanisms principle of
    Section 2.4.1 (recovered algebraically, Section 3.3).
  - The structural logit l_struct combines all deterministic
    contributions per Equation 3.2.3 (eq:struct-logit); the sigmoid
    produces the observed outcome Y per Equation 3.2.4 (eq:cas9-anm).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("explainability/plots/thesis_fig0_dag.png")

# --- visual palette ----------------------------------------------------
# Coherent with the PhD interview slide deck (phd-interview/slides/phd-interview):
#   bg #faf8f4 (warm paper), text #0d1117, accent #1c4fa3 (royal academic blue),
#   muted #5a6470, card #f1ede4, paper #fdfcf8.
# Node fills are kept distinguishable but pulled into a warmer, more
# desaturated family aligned with the deck.
BG_FACE       = "#faf8f4"   # figure & axes background (matches slide bg)
TEXT_PRIMARY  = "#0d1117"   # primary text color
TEXT_MUTED    = "#5a6470"   # muted text

COLOR_INPUT   = "#d0dbeb"   # cool blue-gray  - observed inputs (sgRNA, PAM)
COLOR_LATENT  = "#ede0c3"   # warm sand       - latent biological constructs (P_i, g_pam)
COLOR_EXOG    = "#dcd5db"   # muted lavender  - exogenous noise U (dashed border)
COLOR_LOGIT   = "#cbd6e8"   # pale royal-blue - structural logit (combiner)
COLOR_OUTCOME = "#e8c8b8"   # muted terracotta - measured outcome Y

COLOR_EDGE_F  = "#3a4954"   # slate           - structural assignment (phi, psi, sigma)
COLOR_EDGE_W  = "#1c4fa3"   # deck accent     - weighted edges (w_pos,i)
COLOR_EDGE_A  = "#2960ad"   # vibrant blue    - additive contribution
COLOR_EDGE_U  = "#6b4f7a"   # muted purple    - exogenous noise (dashed)
COLOR_REGION  = "#5a6470"   # deck muted      - regional grouping brackets


def box(ax, xy, text, kind, w=1.8, h=0.9, fontsize=10, dashed=False):
    color = {
        "input":   COLOR_INPUT,
        "latent":  COLOR_LATENT,
        "exog":    COLOR_EXOG,
        "logit":   COLOR_LOGIT,
        "outcome": COLOR_OUTCOME,
    }[kind]
    x, y = xy
    linestyle = "--" if dashed else "-"
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        linewidth=1.6 if dashed else 1.2,
        edgecolor="black", facecolor=color,
        linestyle=linestyle, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", zorder=4)


def pos_node(ax, xy, label, w=0.85, h=0.55, fontsize=10):
    """Small box for individual positional nodes P_i."""
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.07",
        linewidth=1.0, edgecolor="black", facecolor=COLOR_LATENT, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", zorder=4)


def arrow(ax, src, dst, color, label=None, label_pos=0.5,
          style="-", curvature=0.0, lw=1.4,
          shrinkA=10, shrinkB=10, mutation_scale=12):
    connectionstyle = f"arc3,rad={curvature}"
    a = FancyArrowPatch(
        src, dst,
        arrowstyle="-|>", mutation_scale=mutation_scale,
        linestyle=style, linewidth=lw, color=color,
        connectionstyle=connectionstyle, zorder=2,
        shrinkA=shrinkA, shrinkB=shrinkB,
    )
    ax.add_patch(a)
    if label:
        mx = src[0] + (dst[0] - src[0]) * label_pos
        my = src[1] + (dst[1] - src[1]) * label_pos
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.9), zorder=5)


def region_bracket(ax, x_left, x_right, y, label, color=COLOR_REGION,
                   text_offset=0.40):
    """Underline-style bracket BELOW a row of nodes, with label below."""
    bracket_y = y - 0.18
    text_y = y - text_offset
    ax.plot([x_left, x_left], [y, bracket_y], color=color, lw=1.4)
    ax.plot([x_right, x_right], [y, bracket_y], color=color, lw=1.4)
    ax.plot([x_left, x_right], [bracket_y, bracket_y], color=color, lw=1.4)
    ax.text((x_left + x_right) / 2, text_y, label,
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor(BG_FACE)
    ax.set_facecolor(BG_FACE)
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-1.8, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # =====================================================================
    # Layer 1: Observed inputs
    # =====================================================================
    box(ax, (3.5, 7.8), "sgRNA × off-target\n(20 nt protospacer)",
        "input", w=3.4, h=0.9, fontsize=10)
    box(ax, (10.0, 7.8), "PAM\n(3 nt, e.g. NGG)",
        "input", w=2.4, h=0.9, fontsize=10)

    # =====================================================================
    # Layer 2: Latent biological constructs (positional nodes + PAM compat)
    # =====================================================================
    pos_y = 5.4

    # Representative positional nodes with ellipses inside regions.
    # Six concrete nodes (one near each region boundary) plus ellipses
    # to convey "20 nodes in total".
    positions = [
        (0.5,  r"$P_0$"),
        (1.7,  "..."),
        (2.9,  r"$P_7$"),
        (4.1,  r"$P_8$"),
        (5.3,  "..."),
        (6.5,  r"$P_{15}$"),
        (7.7,  r"$P_{16}$"),
        (8.9,  "..."),
        (10.1, r"$P_{19}$"),
    ]
    for x, lbl in positions:
        if lbl == "...":
            ax.text(x, pos_y, "⋯", ha="center", va="center",
                    fontsize=16, color="#888", fontweight="bold")
        else:
            pos_node(ax, (x, pos_y), lbl)

    # Regional brackets below the positional nodes (derived views; the
    # "derived views, not causal nodes" qualification lives in the LaTeX
    # caption of Figure 3.1 and in the prose of Section 3.1, so we do not
    # repeat it inside the figure to keep the layout uncluttered).
    region_bracket(ax, 0.10, 3.30, pos_y - 0.30, "non-seed (i = 0..7)")
    region_bracket(ax, 3.70, 6.90, pos_y - 0.30, "seed (i = 8..15)")
    region_bracket(ax, 7.30, 10.50, pos_y - 0.30, "PAM-proximal (i = 16..19)")

    # PAM compatibility node
    box(ax, (11.8, pos_y), r"$g_{\mathrm{pam}}$" + "\nPAM compat.",
        "latent", w=1.6, h=0.85, fontsize=10)

    # =====================================================================
    # Layer 3: Exogenous noise (independent of X, dashed border)
    # =====================================================================
    box(ax, (12.7, 2.8),
        r"$U$" + "\nexogenous\n(cellular &\nassay context)",
        "exog", w=1.9, h=1.5, fontsize=9, dashed=True)

    # =====================================================================
    # Layer 4: Structural logit (combiner)
    # =====================================================================
    box(ax, (5.0, 2.8),
        "Structural logit\n" + r"$\ell_{\mathrm{struct}}(X)$",
        "logit", w=2.8, h=0.95, fontsize=10)

    # =====================================================================
    # Layer 5: Outcome
    # =====================================================================
    box(ax, (6.8, 0.5),
        r"$Y = \sigma(\ell_{\mathrm{struct}}(X) + U)$",
        "outcome", w=4.6, h=0.85, fontsize=11)

    # =====================================================================
    # Edges
    # =====================================================================

    # ---- Inputs -> positional nodes (sgRNA -> each P_i, shared phi) ----
    sgRNA_anchor = (3.5, 7.35)
    for x, lbl in positions:
        if lbl == "...":
            continue
        arrow(ax, sgRNA_anchor, (x, pos_y + 0.30),
              COLOR_EDGE_F, lw=0.7, mutation_scale=7, shrinkA=4, shrinkB=4)
    # Label the shared phi assignment once
    ax.text(0.4, 6.65, r"$P_i = \phi(X_i)$" + "\n(shared)",
            ha="center", va="center", fontsize=9, color=COLOR_EDGE_F,
            fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor=COLOR_EDGE_F, alpha=0.95))

    # ---- PAM -> g_pam (separate function psi) ----
    arrow(ax, (10.0, 7.35), (11.8, 5.85),
          COLOR_EDGE_F, lw=1.4)
    ax.text(11.8, 6.65,
            r"$g_{\mathrm{pam}} = \psi(X_{\mathrm{pam}})$",
            ha="center", va="center", fontsize=9, color=COLOR_EDGE_F,
            fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor=COLOR_EDGE_F, alpha=0.95))

    # ---- Positional nodes -> structural logit (weighted, w_pos,i <= 0) ----
    logit_anchor_in = (5.0, 3.30)
    for x, lbl in positions:
        if lbl == "...":
            continue
        arrow(ax, (x, pos_y - 0.30), logit_anchor_in,
              COLOR_EDGE_W, lw=0.7, mutation_scale=7,
              shrinkA=2, shrinkB=4)
    # Label weight constraint once
    ax.text(1.5, 4.05,
            r"$w_{\mathrm{pos},i}\, P_i$, " + "\n"
            + r"with $w_{\mathrm{pos},i} \leq 0$ (hard prior)",
            ha="center", va="center", fontsize=9, color=COLOR_EDGE_W,
            fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor=COLOR_EDGE_W, alpha=0.95))

    # ---- g_pam -> structural logit (additive, NOT multiplicative) ----
    arrow(ax, (11.8, 4.95), (6.4, 2.8),
          COLOR_EDGE_A, label=r"$+\,g_{\mathrm{pam}}$",
          label_pos=0.45, curvature=-0.22, lw=1.5)

    # ---- Structural logit -> outcome (sigmoid) ----
    arrow(ax, (5.0, 2.32), (6.2, 0.95),
          COLOR_EDGE_F, label=r"$\sigma(\cdot)$",
          label_pos=0.5, lw=1.5)

    # ---- U -> outcome (additive, dashed for exogenous) ----
    arrow(ax, (12.7, 2.05), (8.6, 0.95),
          COLOR_EDGE_U, label=r"$+\,U$",
          label_pos=0.50, style="--", curvature=-0.15, lw=1.4)

    # =====================================================================
    # Legend
    # =====================================================================
    legend_nodes = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_INPUT,
                       edgecolor="black", label="Observed input"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_LATENT,
                       edgecolor="black",
                       label="Latent biological construct"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_EXOG,
                       edgecolor="black",
                       linestyle="--", linewidth=1.4,
                       label=r"Exogenous noise ($U$, indep. of $X$)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_LOGIT,
                       edgecolor="black",
                       label="Structural logit (combiner)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_OUTCOME,
                       edgecolor="black", label="Observed outcome"),
    ]
    legend_edges = [
        Line2D([0], [0], color=COLOR_EDGE_F, lw=2,
               label=r"Structural assignment ($\phi$, $\psi$, $\sigma$)"),
        Line2D([0], [0], color=COLOR_EDGE_W, lw=2,
               label=r"Weighted edge ($w_{\mathrm{pos},i} \leq 0$, hard prior)"),
        Line2D([0], [0], color=COLOR_EDGE_A, lw=2,
               label=r"Additive contribution ($+$)"),
        Line2D([0], [0], color=COLOR_EDGE_U, lw=2, linestyle="--",
               label=r"Exogenous noise ($+\,U$, dashed)"),
    ]
    leg1 = ax.legend(handles=legend_nodes, loc="lower left",
                     bbox_to_anchor=(-0.02, -0.16), fontsize=8.5,
                     frameon=True, title="Nodes")
    ax.add_artist(leg1)
    ax.legend(handles=legend_edges, loc="lower right",
              bbox_to_anchor=(1.02, -0.16), fontsize=8.5,
              frameon=True, title="Edges")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG_FACE)
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
