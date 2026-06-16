"""Causal DAG figure — TUFTE-STYLED TEST VARIANT.

This is an experimental restyling of ``make_dag_figure.py`` that applies
Edward Tufte's principles (VDQI 1983/2001) to the causal DAG of Chapter 3.
It does NOT replace the original; it writes to a separate output file so the
two can be compared side by side.

Content is identical to the canonical figure (same nodes, edges, equations).
Only the *visual treatment* changes, following the kill-list:

  - Near-monochrome palette. Node ROLE is encoded by subtle luminance/tint
    (warm vs cool faint grays), not by five saturated hues competing for
    attention ("smallest effective difference", Visual Explanations p. 73).
  - A SINGLE accent colour, reserved for the focal outcome node Y, plus one
    restrained muted green for the thesis's signature relationship (the
    monotonic hard-prior weighted edges). Two meaningful accents, no rainbow.
  - Thin light-gray borders instead of heavy black ones (erase non-data-ink).
  - Regular weight type; bold reserved for the focal outcome only
    (layering and separation — data on top, scaffolding faintest).
  - Frameless legends, frameless edge labels (erase the box, keep the word).

See ``.claude/skills/tufte-claude-skill/`` for the principles applied.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("explainability/plots/thesis_fig0_dag_tufte.png")

# --- visual palette (Tufte: near-monochrome + one focal accent) --------
BG_FACE       = "#ffffff"
INK           = "#2b2b2b"   # near-black: primary text / focal data
INK_MUTED     = "#8a8a8a"   # gray: scaffolding, secondary labels

# Node fills: faint tints that encode ROLE by subtle warm/cool luminance,
# not by saturated hue. The border style / accent carries the special cases.
FILL_INPUT    = "#eaf0f4"   # faint cool gray   - observed inputs
FILL_LATENT   = "#f4f1ec"   # faint warm gray   - latent constructs
FILL_EXOG     = "#ffffff"   # white             - exogenous (dashed border = meaning)
FILL_LOGIT    = "#edf0ea"   # faint green-gray  - combiner
FILL_OUTCOME  = "#f6e3e1"   # faint red tint    - outcome (the focal node)

BORDER        = "#b9b9b9"   # light gray: default node border
BORDER_EXOG   = "#9a9a9a"   # exogenous (dashed)
ACCENT        = "#b23b39"   # the single accent: focal outcome Y

# Edges: gray by default; differentiate by STYLE and label, not by rainbow.
COLOR_EDGE_F  = "#9a9a9a"   # gray  - structural assignment (phi, rho, psi, sigma)
COLOR_EDGE_W  = "#4f7a5f"   # muted green - weighted hard-prior edges (signature)
COLOR_EDGE_A  = "#9a9a9a"   # gray  - additive ("+" label carries the meaning)
COLOR_EDGE_U  = "#9a9a9a"   # gray, dashed - exogenous noise
COLOR_REGION  = "#9a9a9a"   # gray  - regional grouping brackets


def box(ax, xy, text, kind, w=1.8, h=0.9, fontsize=10, dashed=False):
    fill = {
        "input":   FILL_INPUT,
        "latent":  FILL_LATENT,
        "exog":    FILL_EXOG,
        "logit":   FILL_LOGIT,
        "outcome": FILL_OUTCOME,
    }[kind]
    # Focal node gets the accent border + bold text; everything else is quiet.
    focal = kind == "outcome"
    edge = ACCENT if focal else (BORDER_EXOG if dashed else BORDER)
    x, y = xy
    linestyle = "--" if dashed else "-"
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        linewidth=1.4 if (dashed or focal) else 1.0,
        edgecolor=edge, facecolor=fill,
        linestyle=linestyle, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=INK,
            fontweight="bold" if focal else "normal", zorder=4)


def pos_node(ax, xy, label, w=0.85, h=0.55, fontsize=10):
    """Small box for individual positional nodes P_i."""
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.07",
        linewidth=0.8, edgecolor=BORDER, facecolor=FILL_LATENT, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=4)


def arrow(ax, src, dst, color, label=None, label_pos=0.5,
          style="-", curvature=0.0, lw=1.2,
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
        # Frameless label: a faint white halo for legibility over crossing
        # lines (layering/separation), but no boxed border (erase the box).
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.75), zorder=5)


def region_bracket(ax, x_left, x_right, y, label, color=COLOR_REGION,
                   text_offset=0.40):
    """Underline-style bracket BELOW a row of nodes, with label below."""
    bracket_y = y - 0.18
    text_y = y - text_offset
    ax.plot([x_left, x_left], [y, bracket_y], color=color, lw=1.0)
    ax.plot([x_right, x_right], [y, bracket_y], color=color, lw=1.0)
    ax.plot([x_left, x_right], [bracket_y, bracket_y], color=color, lw=1.0)
    ax.text((x_left + x_right) / 2, text_y, label,
            ha="center", va="top", fontsize=8.5, color=INK_MUTED,
            style="italic")


def edge_label(ax, x, y, text, color):
    """Frameless italic annotation for a structural assignment."""
    ax.text(x, y, text, ha="center", va="center",
            fontsize=9, color=color, style="italic", zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.patch.set_facecolor(BG_FACE)
    ax.set_facecolor(BG_FACE)
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-1.8, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # =====================================================================
    # Layer 1: Observed inputs (three columns: spacer, context, PAM)
    # =====================================================================
    box(ax, (3.5, 7.8), "sgRNA × off-target\n(20 nt protospacer)",
        "input", w=3.4, h=0.9, fontsize=10)
    box(ax, (8.5, 7.8), "GC composition\n(deterministic slice of $X$)",
        "input", w=2.8, h=0.9, fontsize=9.5)
    box(ax, (12.5, 7.8), "PAM\n(3 nt, e.g. NGG)",
        "input", w=2.4, h=0.9, fontsize=10)

    # =====================================================================
    # Layer 2: Latent biological constructs
    # =====================================================================
    pos_y = 5.4

    # Compressed grid of 20 positional nodes (ellipses for in-between)
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
                    fontsize=14, color=INK_MUTED)
        else:
            pos_node(ax, (x, pos_y), lbl, w=0.75, h=0.55, fontsize=9)

    # Regional brackets below the positional nodes (derived views).
    region_bracket(ax, 0.05, 2.40, pos_y - 0.30, "non-seed (i = 0..7)")
    region_bracket(ax, 2.45, 4.75, pos_y - 0.30, "seed (i = 8..15)")
    region_bracket(ax, 4.85, 7.20, pos_y - 0.30, "PAM-proximal (i = 16..19)")

    # Context node (GC duplex stability)
    box(ax, (8.5, pos_y), r"$g_{\mathrm{ctx}}$" + "\ncontext\n(GC duplex)",
        "latent", w=1.9, h=0.95, fontsize=9.5)

    # PAM compatibility node
    box(ax, (12.5, pos_y), r"$g_{\mathrm{pam}}$" + "\nPAM compat.",
        "latent", w=1.7, h=0.95, fontsize=10)

    # =====================================================================
    # Layer 3: Exogenous noise (independent of X, dashed border)
    # =====================================================================
    box(ax, (13.6, 2.8),
        r"$U$" + "\nexogenous\n(cellular &\nassay context)",
        "exog", w=1.9, h=1.5, fontsize=9, dashed=True)

    # =====================================================================
    # Layer 4: Structural logit (combiner)
    # =====================================================================
    box(ax, (5.5, 2.8),
        "Structural logit\n" + r"$\ell_{\mathrm{struct}}(X)$",
        "logit", w=3.0, h=0.95, fontsize=10)

    # =====================================================================
    # Layer 5: Outcome (focal node)
    # =====================================================================
    box(ax, (8.0, 0.5),
        r"$Y = \sigma(\ell_{\mathrm{struct}}(X) + U)$",
        "outcome", w=4.6, h=0.85, fontsize=11)

    # =====================================================================
    # Edges
    # =====================================================================

    # ---- sgRNA × off-target -> each P_i (shared phi) ----
    sgRNA_anchor = (3.5, 7.35)
    for x, lbl in positions:
        if lbl == "...":
            continue
        arrow(ax, sgRNA_anchor, (x, pos_y + 0.30),
              COLOR_EDGE_F, lw=0.6, mutation_scale=7, shrinkA=4, shrinkB=4)
    edge_label(ax, 0.4, 6.65, r"$P_i = \phi(X_i)$  (shared)", COLOR_EDGE_F)

    # ---- GC composition -> g_ctx (separate function rho) ----
    arrow(ax, (8.5, 7.35), (8.5, 5.95), COLOR_EDGE_F, lw=1.2)
    edge_label(ax, 8.5, 6.65, r"$g_{\mathrm{ctx}} = \rho(X_{\mathrm{ctx}})$",
               COLOR_EDGE_F)

    # ---- PAM -> g_pam (separate function psi) ----
    arrow(ax, (12.5, 7.35), (12.5, 5.95), COLOR_EDGE_F, lw=1.2)
    edge_label(ax, 12.5, 6.65, r"$g_{\mathrm{pam}} = \psi(X_{\mathrm{pam}})$",
               COLOR_EDGE_F)

    # ---- Positional nodes -> structural logit (weighted, w_pos,i <= 0) ----
    logit_anchor_in = (5.5, 3.30)
    for x, lbl in positions:
        if lbl == "...":
            continue
        arrow(ax, (x, pos_y - 0.30), logit_anchor_in,
              COLOR_EDGE_W, lw=0.6, mutation_scale=7,
              shrinkA=2, shrinkB=4)
    edge_label(ax, 0.6, 4.05,
               r"$w_{\mathrm{pos},i}\, P_i,\ w_{\mathrm{pos},i} \leq 0$"
               + "\n(hard prior)", COLOR_EDGE_W)

    # ---- g_ctx -> structural logit (additive) ----
    arrow(ax, (8.5, 4.92), (6.6, 3.30),
          COLOR_EDGE_A, label=r"$+\,g_{\mathrm{ctx}}$",
          label_pos=0.55, curvature=-0.18, lw=1.2)

    # ---- g_pam -> structural logit (additive) ----
    arrow(ax, (12.5, 4.92), (7.0, 2.95),
          COLOR_EDGE_A, label=r"$+\,g_{\mathrm{pam}}$",
          label_pos=0.50, curvature=-0.22, lw=1.2)

    # ---- Structural logit -> outcome (sigmoid) ----
    arrow(ax, (5.5, 2.32), (7.2, 0.95),
          COLOR_EDGE_F, label=r"$\sigma(\cdot)$",
          label_pos=0.5, lw=1.2)

    # ---- U -> outcome (additive, dashed for exogenous) ----
    arrow(ax, (13.6, 2.05), (9.4, 0.95),
          COLOR_EDGE_U, label=r"$+\,U$",
          label_pos=0.50, style="--", curvature=-0.15, lw=1.2)

    # =====================================================================
    # Legend (frameless: erase the box, keep the key)
    # =====================================================================
    legend_nodes = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=FILL_INPUT,
                       edgecolor=BORDER, label="Observed input"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=FILL_LATENT,
                       edgecolor=BORDER,
                       label="Latent biological construct"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=FILL_EXOG,
                       edgecolor=BORDER_EXOG,
                       linestyle="--", linewidth=1.2,
                       label=r"Exogenous noise ($U$, indep. of $X$)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=FILL_LOGIT,
                       edgecolor=BORDER,
                       label="Structural logit (combiner)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=FILL_OUTCOME,
                       edgecolor=ACCENT, label="Observed outcome (focal)"),
    ]
    legend_edges = [
        Line2D([0], [0], color=COLOR_EDGE_F, lw=1.6,
               label=r"Structural assignment ($\phi$, $\psi$, $\sigma$)"),
        Line2D([0], [0], color=COLOR_EDGE_W, lw=1.6,
               label=r"Weighted edge ($w_{\mathrm{pos},i} \leq 0$, hard prior)"),
        Line2D([0], [0], color=COLOR_EDGE_A, lw=1.6,
               label=r"Additive contribution ($+$)"),
        Line2D([0], [0], color=COLOR_EDGE_U, lw=1.6, linestyle="--",
               label=r"Exogenous noise ($+\,U$, dashed)"),
    ]
    leg1 = ax.legend(handles=legend_nodes, loc="lower left",
                     bbox_to_anchor=(-0.02, -0.16), fontsize=8.5,
                     frameon=False, title="Nodes",
                     labelcolor=INK)
    leg1.get_title().set_color(INK_MUTED)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=legend_edges, loc="lower right",
                     bbox_to_anchor=(1.02, -0.16), fontsize=8.5,
                     frameon=False, title="Edges",
                     labelcolor=INK)
    leg2.get_title().set_color(INK_MUTED)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG_FACE)
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
