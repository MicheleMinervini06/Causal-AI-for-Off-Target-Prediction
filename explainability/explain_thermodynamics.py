import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from models.deep.encoding import BiologicalMismatchEncoder
from models.deep.neural_scm import NeuralSCM


def _mismatch_type_vec(guide_base: str, target_base: str) -> list[float]:
    """Restituisce il vettore one-hot 4D [Match, Wobble, Transition, Transversion]."""
    if guide_base == target_base:
        return [1.0, 0.0, 0.0, 0.0]
    pair = {guide_base, target_base}
    if pair == {"G", "T"}:
        return [0.0, 1.0, 0.0, 0.0]
    if pair in [{"A", "G"}, {"C", "T"}]:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Caricamento Modello (adopted: Exp30)
    ADOPTED_RUN_DIR = "Exp30_Ablation_EncoderBiologicalMismatch_UseEncoder"
    model_path = Path("experiments/results") / ADOPTED_RUN_DIR / "neural_scm.pt"
    state_dict = torch.load(model_path, map_location=device)

    # Auto-detect dal state_dict: context_dim e positional_use_encoder
    context_dim = 0
    if "context_net.0.weight" in state_dict:
        context_dim = state_dict["context_net.0.weight"].shape[1]
    positional_use_encoder = False
    if "pos_node.0.weight" in state_dict:
        pos_in_dim = state_dict["pos_node.0.weight"].shape[1]
        positional_use_encoder = (pos_in_dim != 4)

    encoder = BiologicalMismatchEncoder()
    model = NeuralSCM(
        encoder=encoder,
        architecture="positional_mlp",
        hidden_dim=8,
        context_dim=context_dim,
        positional_use_encoder=positional_use_encoder,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Estrazione del peso strutturale globale W_i (shape: [20])
    w_pos_eff = -F.softplus(model.w_pos).detach().cpu().numpy()

    # 3. Firme chimiche: coppie (guide_base, target_base) che rappresentano i 4 tipi
    mismatch_pairs = {
        "Match":        ("A", "A"),
        "Wobble":       ("G", "T"),
        "Transition":   ("A", "G"),
        "Transversion": ("A", "C"),
    }

    effective_penalties = {name: [] for name in mismatch_pairs.keys()}
    positions = np.arange(1, 21)

    print("Computing 2D structural-penalty matrix (position x mismatch type)...")

    with torch.no_grad():
        for name, (guide_base, target_base) in mismatch_pairs.items():
            if positional_use_encoder:
                # pos_node consumes the rich encoder output (12-dim biological mismatch).
                # Replicate the same target sequence at all 20 positions to obtain
                # the canonical per-type penalty (the encoder's biological_mismatch
                # output depends only on the (sgRNA base, target base) pair, not on
                # the absolute position, so the input is the same across positions).
                sg_str = guide_base * 20
                ot_str = target_base * 20
                x_spacer, _ = encoder([sg_str], [ot_str])  # [1, 20, 12]
                pos_penalties = F.relu(model.pos_node(x_spacer).squeeze(-1)).cpu().numpy()[0]
            else:
                # Legacy path (positional_use_encoder=False): pos_node consumes the
                # 4-dim mismatch-type one-hot recomputed internally.
                type_vec = _mismatch_type_vec(guide_base, target_base)
                s_typed = torch.tensor([[type_vec] * 20], dtype=torch.float32, device=device)
                pos_penalties = F.relu(model.pos_node(s_typed).squeeze(-1)).cpu().numpy()[0]

            for i in range(20):
                penalty = abs(w_pos_eff[i]) * pos_penalties[i]
                effective_penalties[name].append(penalty)

    # --- Plot ---
    # Saturated palette aligned with the pgfplots tints in 4_results.tex.
    # The mismatch-type colors recall the canonical Cas9 biophysical hierarchy:
    #   match (reference, no penalty): neutral gray, dashed
    #   wobble: vivid blue (least disruptive non-match)
    #   transition: vivid orange (intermediate)
    #   transversion: vivid red (most disruptive)
    COLOR_REGION_NONSEED = "#E8E8E8"
    COLOR_REGION_SEED    = "#FFEBA8"
    COLOR_REGION_PAMPROX = "#FFCDAA"

    colors = {
        "Match":        "#808080",
        "Wobble":       "#1F77FF",
        "Transition":   "#FF8C00",
        "Transversion": "#D62728",
    }
    styles = {"Match": "--", "Wobble": "-", "Transition": "-", "Transversion": "-"}
    markers = {"Match": "x", "Wobble": "o", "Transition": "s", "Transversion": "D"}

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Region shading
    ax.axvspan(0.5,  8.5,  color=COLOR_REGION_NONSEED, alpha=0.55, zorder=0,
               label="Non-seed ($i = 1$–$8$)")
    ax.axvspan(8.5,  16.5, color=COLOR_REGION_SEED,    alpha=0.55, zorder=0,
               label="Seed ($i = 9$–$16$)")
    ax.axvspan(16.5, 20.5, color=COLOR_REGION_PAMPROX, alpha=0.55, zorder=0,
               label="PAM-proximal ($i = 17$–$20$)")

    for name, penalties in effective_penalties.items():
        ax.plot(positions, penalties,
                marker=markers[name], markersize=8, linewidth=2.2,
                color=colors[name], linestyle=styles[name], label=name, zorder=3)

    ax.set_xlabel("Spacer position ($5' \\to 3'$, PAM at $3'$ end)")
    ax.set_ylabel("$|w_{\\mathrm{pos},i}| \\cdot \\phi(\\mathrm{type})$  (structural penalty)")
    ax.set_xticks(positions)
    ax.set_xlim(0.5, 20.5)
    ax.legend(loc="upper left", fontsize=10, frameon=True, ncol=2)

    plt.tight_layout()
    output_path = Path("explainability/plots/thermodynamic_profile.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path.resolve()}")

if __name__ == "__main__":
    main()