"""Data preprocessing pipeline figure for Chapter 3 of the thesis (Figure 3.3).

Visualizes the flow described in Section 3.4 of the thesis, from raw assay
measurements (Lazzarotto et al. 2020 study, processed via Yaish et al. 2022
protocol with Cas-OFFinder-derived negatives) to the four model-ready data
splits used in Sections 3.5 and 3.6:

  - CHANGE-seq is processed, per-guide partitioned into training, validation
    and within-distribution test splits;
  - the saturation filter is applied to the TRAINING split only (a post-hoc
    decision motivated in Section 3.4.3 by the U-residual diagnostic of
    Chapter 4);
  - GUIDE-seq is reserved as the cross-assay test set and never enters the
    training distribution (it bypasses both the per-guide partition and the
    filter);
  - all four splits feed a single sequence-encoding step producing the
    model-ready tensors of Section 3.4.2.

Visual language (palette and helper functions) is kept consistent with
make_dag_figure.py and make_architecture_figure.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("explainability/plots/thesis_fig0c_data_pipeline.png")

# --- visual palette ----------------------------------------------------
# Mirrors make_dag_figure.py / make_architecture_figure.py.
COLOR_INPUT   = "#bcd6ec"   # light blue   - raw data sources (assays, enumeration)
COLOR_MODULE  = "#fff2b3"   # light yellow - processing operations (protocol, partition, filter)
COLOR_LATENT  = "#ffd9a8"   # light orange - intermediate / encoded data
COLOR_LOGIT   = "#d7e6c9"   # light green  - model-ready splits

COLOR_EDGE_F  = "#555555"   # gray       - generic data flow
COLOR_EDGE_W  = "#0b5e2b"   # dark green - "use" / fed-to-model arrows
COLOR_EDGE_U  = "#7a3a99"   # purple     - held-out path (dashed)


def box(ax, xy, text, kind, w=1.8, h=0.9, fontsize=10):
    color = {
        "input":  COLOR_INPUT,
        "module": COLOR_MODULE,
        "latent": COLOR_LATENT,
        "logit":  COLOR_LOGIT,
    }[kind]
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        linewidth=1.2, edgecolor="black", facecolor=color, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
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
                fontsize=8.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.92), zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.5, 11)
    ax.set_aspect("equal")
    ax.axis("off")

    # =====================================================================
    # Layer 1: Raw data sources
    # =====================================================================
    # CHANGE-seq positives (in vitro)
    box(ax, (3.0, 9.4),
        "CHANGE-seq positives\nin vitro, 110 sgRNAs",
        "input", w=3.6, h=1.0, fontsize=9.5)
    # GUIDE-seq positives (in cellula, matched subset)
    box(ax, (7.5, 9.4),
        "GUIDE-seq positives\nin cellula, 58 sgRNAs\n(subset of CHANGE-seq)",
        "input", w=3.6, h=1.2, fontsize=9.5)
    # Cas-OFFinder candidate enumeration
    box(ax, (12.0, 9.4),
        "Cas-OFFinder candidates\n(genome-wide,\n$\\leq 6$ mismatches)",
        "input", w=3.6, h=1.0, fontsize=9.5)

    # =====================================================================
    # Layer 2: Yaish 2022 preprocessing protocol (wide banner)
    # =====================================================================
    box(ax, (7.5, 7.4),
        "Yaish et al. 2022 preprocessing protocol\n"
        "(read-count transformation, activity threshold, "
        "inclusion of inactive sites)",
        "module", w=11.0, h=1.1, fontsize=10)

    # arrows from raw sources to Yaish
    arrow(ax, (3.0, 8.85), (4.5, 8.05), COLOR_EDGE_F, lw=1.2,
          shrinkA=4, shrinkB=4)
    arrow(ax, (7.5, 8.75), (7.5, 8.05), COLOR_EDGE_F, lw=1.2,
          shrinkA=4, shrinkB=4)
    arrow(ax, (12.0, 8.85), (10.5, 8.05), COLOR_EDGE_F, lw=1.2,
          shrinkA=4, shrinkB=4)

    # =====================================================================
    # Layer 3: Per-guide partition (CHANGE-seq) + GUIDE-seq passthrough
    # =====================================================================
    # Per-guide partition operation (CHANGE-seq branch, left)
    box(ax, (4.0, 5.5),
        "Per-guide partition of CHANGE-seq\n(disjoint sgRNAs across splits)",
        "module", w=4.8, h=1.0, fontsize=10)

    # Arrows from Yaish to per-guide partition (left) and to GUIDE-seq passthrough (right)
    arrow(ax, (5.0, 6.85), (4.0, 6.05), COLOR_EDGE_F, lw=1.4,
          shrinkA=4, shrinkB=4)

    # GUIDE-seq passthrough — dashed purple arrow indicating "held out"
    arrow(ax, (10.5, 6.85), (12.5, 3.0), COLOR_EDGE_U, lw=1.5,
          label="held out\n(no split, no filter)",
          label_pos=0.45, style="--", curvature=0.10,
          shrinkA=4, shrinkB=4)

    # =====================================================================
    # Layer 4: Final splits (CHANGE-seq side: 3 splits; GUIDE-seq: 1)
    # =====================================================================
    # CHANGE-seq splits
    box(ax, (1.5, 3.0), "train", "logit", w=1.8, h=0.7, fontsize=10)
    box(ax, (4.0, 3.0), "val",   "logit", w=1.5, h=0.7, fontsize=10)
    box(ax, (6.5, 3.0), "in-dist\ntest", "logit", w=1.8, h=0.85, fontsize=10)
    # GUIDE-seq cross-assay test (held out)
    box(ax, (12.5, 3.0),
        "cross-assay test\n(GUIDE-seq)",
        "logit", w=3.0, h=0.85, fontsize=10)

    # Arrows from per-guide partition to 3 CHANGE-seq splits
    # The train arrow is the only one that passes through the saturation filter
    arrow(ax, (3.0, 5.05), (1.7, 3.40),
          COLOR_EDGE_W, lw=1.5,
          label="saturation filter\n(train only)",
          label_pos=0.55, curvature=-0.15)
    arrow(ax, (4.0, 5.05), (4.0, 3.40),
          COLOR_EDGE_F, lw=1.2,
          shrinkA=4, shrinkB=4)
    arrow(ax, (5.0, 5.05), (6.3, 3.40),
          COLOR_EDGE_F, lw=1.2,
          shrinkA=4, shrinkB=4, curvature=0.10)

    # =====================================================================
    # Layer 5: Encoding (wide banner, applies to all four splits)
    # =====================================================================
    box(ax, (7.0, 1.1),
        "Sequence encoding"
        "4-dim mismatch-type per protospacer position  +  one-hot PAM tokens",
        "latent", w=13.5, h=0.9, fontsize=10)

    # arrows from each split to the encoding banner (fan)
    for src_x in [1.5, 4.0, 6.5]:
        arrow(ax, (src_x, 2.65), (src_x + 1.2, 1.55),
              COLOR_EDGE_F, lw=0.8, mutation_scale=7, shrinkA=4, shrinkB=4)
    arrow(ax, (12.5, 2.58), (10.0, 1.55),
          COLOR_EDGE_F, lw=0.8, mutation_scale=7, shrinkA=4, shrinkB=4)

    # =====================================================================
    # Final arrow: to model fitting / evaluation
    # =====================================================================
    arrow(ax, (7.0, 0.65), (7.0, -0.30),
          COLOR_EDGE_W, lw=1.6, mutation_scale=14,
          label="to model fitting and counterfactual inference",
          label_pos=0.5)

    # =====================================================================
    # Title
    # =====================================================================
    ax.set_title(
        "Data preprocessing pipeline\n",
        fontsize=13, fontweight="bold", pad=18)

    # =====================================================================
    # Legend
    # =====================================================================
    legend_nodes = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_INPUT,
                       edgecolor="black", label="Raw data source"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_MODULE,
                       edgecolor="black", label="Processing operation"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_LATENT,
                       edgecolor="black", label="Encoded tensor (model-ready)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_LOGIT,
                       edgecolor="black", label="Data split"),
    ]
    legend_edges = [
        Line2D([0], [0], color=COLOR_EDGE_F, lw=2, label="Data flow"),
        Line2D([0], [0], color=COLOR_EDGE_W, lw=2,
               label="Saturation filter / fed to model"),
        Line2D([0], [0], color=COLOR_EDGE_U, lw=2, linestyle="--",
               label="Held out (no preprocessing)"),
    ]
    leg1 = ax.legend(handles=legend_nodes, loc="lower left",
                     bbox_to_anchor=(-0.02, -0.13), fontsize=8.5,
                     frameon=True, title="Boxes")
    ax.add_artist(leg1)
    ax.legend(handles=legend_edges, loc="lower right",
              bbox_to_anchor=(1.02, -0.13), fontsize=8.5,
              frameon=True, title="Edges")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
