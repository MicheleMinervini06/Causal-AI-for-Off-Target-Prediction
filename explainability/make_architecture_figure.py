"""Neural SCM architecture figure for Chapter 3 of the thesis.

This figure shows the IMPLEMENTATION that realizes the biological causal
DAG of Figure 3.1 (see make_dag_figure.py). Each DAG node is instantiated
by a dedicated trainable module or by a deterministic computation; arrows
mirror the causal edges of the DAG and make the architectural choices
explicit.

Reflects the final adopted model (positional_mlp + additive PAM +
GC context module, Exp24 in findings.md), per Section 3.2 of the thesis:

  - Shared positional MLP phi (Equation eq:phi-i) applied independently
    to each of the 20 protospacer positions (weight sharing).
  - Separate PAM module psi (Equation eq:psi-pam).
  - Separate context module rho (Equation eq:rho-ctx) consuming the three
    GC-composition features (gc_sgRNA, gc_offtarget, |delta|).
  - Hard prior 1: P_i >= 0 via ReLU on the output of phi.
  - Hard prior 2: w_pos,i <= 0 via -softplus reparam (Equation eq:wpos-reparam).
  - Combiner producing the structural logit l_struct (Equation eq:struct-logit)
    as additive composition of positional, context, and PAM contributions.
  - Exogenous noise U recovered ALGEBRAICALLY (Section 3.3, eq:U-abduction),
    not learned: entering additively at the final logit.
  - Sigmoid producing the cleavage outcome Y (Equation eq:cas9-anm).

Visual language (palette and helper functions) is kept consistent with
make_dag_figure.py, with the addition of a yellow tone (COLOR_MODULE) for
the trainable components — the element that distinguishes the
implementation view from the abstract DAG view.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("explainability/plots/thesis_fig0b_architecture.png")

# --- visual palette ----------------------------------------------------
# Mirrors make_dag_figure.py for stylistic coherence + COLOR_MODULE for
# trainable components.
COLOR_INPUT   = "#bcd6ec"   # light blue   - raw inputs (sgRNA, PAM)
COLOR_LATENT  = "#ffd9a8"   # light orange - encoded tensors, P_i, g_pam
COLOR_MODULE  = "#fff2b3"   # light yellow - trainable modules (phi, psi)
COLOR_EXOG    = "#e8e0f0"   # light purple - exogenous noise U (dashed border)
COLOR_LOGIT   = "#d7e6c9"   # light green  - structural logit (combiner)
COLOR_OUTCOME = "#f5a8a8"   # light red    - measured outcome Y

COLOR_EDGE_F  = "#555555"   # gray         - structural / functional (phi, psi, sigma, ReLU)
COLOR_EDGE_W  = "#0b5e2b"   # dark green   - weighted edges (w_pos,i)
COLOR_EDGE_A  = "#1f3b6e"   # dark blue    - additive contribution
COLOR_EDGE_U  = "#7a3a99"   # purple       - exogenous noise (dashed)


def box(ax, xy, text, kind, w=1.8, h=0.9, fontsize=10, dashed=False):
    color = {
        "input":   COLOR_INPUT,
        "latent":  COLOR_LATENT,
        "module":  COLOR_MODULE,
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
    """Small box for individual positional nodes P_i (matches DAG figure)."""
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


def main():
    fig, ax = plt.subplots(figsize=(15, 10.5))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.8, 10.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # =====================================================================
    # Layer 1: Raw inputs (three columns: spacer, context, PAM)
    # =====================================================================
    box(ax, (3.5, 9.8), "sgRNA × off-target\n(20 nt protospacer)",
        "input", w=3.4, h=0.9, fontsize=10)
    box(ax, (8.5, 9.8), "GC composition\n(3 scalar features)",
        "input", w=2.6, h=0.9, fontsize=10)
    box(ax, (12.5, 9.8), "PAM\n(3 nt, e.g. NGG)",
        "input", w=2.4, h=0.9, fontsize=10)

    # =====================================================================
    # Layer 2: Encoded tensors
    # =====================================================================
    box(ax, (3.5, 8.2),
        "Mismatch-type encoding\n(4 classes per position)\n[B × 20 × 4]",
        "latent", w=3.4, h=1.0, fontsize=9)
    box(ax, (8.5, 8.2),
        "GC feature vector\n[B × 3]",
        "latent", w=2.4, h=1.0, fontsize=9)
    box(ax, (12.5, 8.2),
        "PAM nucleotide encoding\n(one-hot per base)\n[B × 3 × 4]",
        "latent", w=2.8, h=1.0, fontsize=9)

    # =====================================================================
    # Layer 3: Trainable modules
    # =====================================================================
    box(ax, (3.5, 6.5),
        r"$\phi$  shared positional MLP" + "\n(Linear → ReLU → Linear)",
        "module", w=3.8, h=0.95, fontsize=10)
    box(ax, (8.5, 6.5),
        r"$\rho$  context module (MLP)",
        "module", w=2.8, h=0.95, fontsize=10)
    box(ax, (12.5, 6.5),
        r"$\psi$  PAM module (MLP)",
        "module", w=2.8, h=0.95, fontsize=10)

    # =====================================================================
    # Layer 4: Per-position penalties P_i (post-ReLU), g_ctx and g_pam
    # =====================================================================
    pos_y = 4.7
    positions = [
        (0.4,  r"$P_0$"),
        (1.2,  "..."),
        (2.0,  r"$P_7$"),
        (2.8,  r"$P_8$"),
        (3.6,  "..."),
        (4.4,  r"$P_{15}$"),
        (5.2,  r"$P_{16}$"),
        (6.0,  "..."),
        (6.8,  r"$P_{19}$"),
    ]
    for x, lbl in positions:
        if lbl == "...":
            ax.text(x, pos_y, "⋯", ha="center", va="center",
                    fontsize=14, color="#888", fontweight="bold")
        else:
            pos_node(ax, (x, pos_y), lbl, w=0.75, h=0.55, fontsize=9)

    # Annotation: P_i are the 20 per-position penalties (post-ReLU, so >= 0)
    ax.text(3.6, pos_y - 0.65,
            r"per-position penalties $P_i \geq 0$  (Hard Prior 1: ReLU on $\phi$ output)",
            ha="center", va="top", fontsize=8.5, color="#0b5e2b",
            fontweight="bold", style="italic")

    # g_ctx node (context contribution)
    box(ax, (8.5, pos_y), r"$g_{\mathrm{ctx}}$",
        "latent", w=1.4, h=0.7, fontsize=11)

    # g_pam node
    box(ax, (12.5, pos_y), r"$g_{\mathrm{pam}}$",
        "latent", w=1.4, h=0.7, fontsize=11)

    # =====================================================================
    # Layer 5: Combiner ℓ_struct + U exogenous
    # =====================================================================
    box(ax, (5.5, 2.7),
        "Combiner  " + r"$\ell_{\mathrm{struct}}(X)$" + "\n"
        + r"$= \sum_{i=0}^{19} w_{\mathrm{pos},i}\, P_i + g_{\mathrm{ctx}} + g_{\mathrm{pam}} + c$",
        "logit", w=8.6, h=1.1, fontsize=10)

    # U exogenous: dashed border, recovered algebraically (NOT a trainable component)
    box(ax, (13.8, 2.7),
        r"$U$" + "\nexogenous\n(algebraic\nabduction)",
        "exog", w=2.3, h=1.5, fontsize=9, dashed=True)

    # =====================================================================
    # Layer 6: Outcome
    # =====================================================================
    box(ax, (8.0, 0.7),
        r"$Y = \sigma(\ell_{\mathrm{struct}}(X) + U)$",
        "outcome", w=4.8, h=0.85, fontsize=11)

    # =====================================================================
    # Edges
    # =====================================================================

    # ---- raw inputs -> encoded tensors ----
    arrow(ax, (3.5, 9.35), (3.5, 8.75), COLOR_EDGE_F, lw=1.3,
          label="encode", label_pos=0.5)
    arrow(ax, (8.5, 9.35), (8.5, 8.75), COLOR_EDGE_F, lw=1.3,
          label="GC stats", label_pos=0.5)
    arrow(ax, (12.5, 9.35), (12.5, 8.75), COLOR_EDGE_F, lw=1.3,
          label="encode", label_pos=0.5)

    # ---- encoded tensors -> trainable modules ----
    arrow(ax, (3.5, 7.70), (3.5, 7.00), COLOR_EDGE_F, lw=1.3)
    arrow(ax, (8.5, 7.70), (8.5, 7.00), COLOR_EDGE_F, lw=1.3)
    arrow(ax, (12.5, 7.70), (12.5, 7.00), COLOR_EDGE_F, lw=1.3)

    # ---- phi -> per-position outputs (fan, showing weight-sharing) ----
    phi_anchor = (3.5, 6.02)
    for x, lbl in positions:
        if lbl == "...":
            continue
        arrow(ax, phi_anchor, (x, pos_y + 0.30),
              COLOR_EDGE_F, lw=0.7, mutation_scale=7, shrinkA=4, shrinkB=4)

    # ---- rho -> g_ctx ----
    arrow(ax, (8.5, 6.02), (8.5, 5.05), COLOR_EDGE_F, lw=1.3)

    # ---- psi -> g_pam ----
    arrow(ax, (12.5, 6.02), (12.5, 5.05), COLOR_EDGE_F, lw=1.3)

    # ---- per-position outputs -> combiner (weighted, w_pos,i <= 0) ----
    logit_anchor_in = (4.5, 3.30)
    for x, lbl in positions:
        if lbl == "...":
            continue
        arrow(ax, (x, pos_y - 0.30), logit_anchor_in,
              COLOR_EDGE_W, lw=0.7, mutation_scale=7,
              shrinkA=2, shrinkB=4)
    # Annotation: weighted edges + Hard Prior 2 (softplus reparam)
    ax.text(0.6, 3.85,
            "Hard Prior 2:\n"
            + r"$w_{\mathrm{pos},i} = -\,\mathrm{softplus}(\tilde{w}_{\mathrm{pos},i})$" + "\n"
            + r"$\Rightarrow w_{\mathrm{pos},i} \leq 0$",
            ha="center", va="center", fontsize=8.5, color=COLOR_EDGE_W,
            fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor=COLOR_EDGE_W, alpha=0.95))

    # ---- g_ctx -> combiner (additive) ----
    arrow(ax, (8.5, 4.35), (7.2, 3.25),
          COLOR_EDGE_A, label=r"$+\,g_{\mathrm{ctx}}$",
          label_pos=0.55, curvature=-0.15, lw=1.5)

    # ---- g_pam -> combiner (additive) ----
    arrow(ax, (12.5, 4.35), (9.4, 3.10),
          COLOR_EDGE_A, label=r"$+\,g_{\mathrm{pam}}$",
          label_pos=0.55, curvature=-0.20, lw=1.5)

    # ---- combiner -> outcome (sigmoid) ----
    arrow(ax, (5.5, 2.15), (7.0, 1.13),
          COLOR_EDGE_F, label=r"$\sigma(\cdot)$",
          label_pos=0.55, lw=1.5)

    # ---- U -> outcome ----
    arrow(ax, (13.8, 1.95), (9.4, 1.13),
          COLOR_EDGE_U, label=r"$+\,U$",
          label_pos=0.45, style="--", curvature=-0.15, lw=1.4)

    # =====================================================================
    # Title
    # =====================================================================
    ax.set_title(
        "Neural SCM architecture: trainable modules and hard priors\n",
        fontsize=13, fontweight="bold", pad=18)

    # =====================================================================
    # Legend
    # =====================================================================
    legend_nodes = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_INPUT,
                       edgecolor="black", label="Raw input"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_LATENT,
                       edgecolor="black",
                       label="Encoded tensor / intermediate value"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_MODULE,
                       edgecolor="black",
                       label="Trainable module"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_LOGIT,
                       edgecolor="black",
                       label="Structural logit (combiner)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_EXOG,
                       edgecolor="black",
                       linestyle="--", linewidth=1.4,
                       label=r"Exogenous noise ($U$, algebraic abduction)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COLOR_OUTCOME,
                       edgecolor="black",
                       label="Observed outcome"),
    ]
    legend_edges = [
        Line2D([0], [0], color=COLOR_EDGE_F, lw=2,
               label=r"Structural / functional ($\phi$, $\psi$, ReLU, $\sigma$)"),
        Line2D([0], [0], color=COLOR_EDGE_W, lw=2,
               label=r"Weighted edge ($w_{\mathrm{pos},i} \leq 0$, hard prior)"),
        Line2D([0], [0], color=COLOR_EDGE_A, lw=2,
               label=r"Additive contribution ($+$)"),
        Line2D([0], [0], color=COLOR_EDGE_U, lw=2, linestyle="--",
               label=r"Exogenous noise injection ($+\,U$, algebraic)"),
    ]
    leg1 = ax.legend(handles=legend_nodes, loc="lower left",
                     bbox_to_anchor=(-0.02, -0.14), fontsize=8.5,
                     frameon=True, title="Nodes / boxes")
    ax.add_artist(leg1)
    ax.legend(handles=legend_edges, loc="lower right",
              bbox_to_anchor=(1.02, -0.14), fontsize=8.5,
              frameon=True, title="Edges")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
