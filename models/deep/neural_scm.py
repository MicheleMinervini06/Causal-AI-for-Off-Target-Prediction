from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import PairwiseTokenEncoder
from .modules import NonSeedModule, PAMModule, ProximalModule, SeedExtensionModule


class NeuralSCM(nn.Module):
    """
    Neural Structural Causal Model per previsioni CRISPR.
    Assembla moduli indipendenti in un DAG causale esplicito.
    """

    def __init__(self, embed_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        
        self.encoder = PairwiseTokenEncoder(embed_dim=embed_dim)

        self.pam_node = PAMModule(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.proximal_node = ProximalModule(embed_dim=embed_dim)
        self.seed_node = SeedExtensionModule(embed_dim=embed_dim)
        self.nonseed_node = NonSeedModule(embed_dim=embed_dim)

        self.w_proximal = nn.Parameter(torch.randn(1))
        self.w_seed = nn.Parameter(torch.randn(1))
        self.w_nonseed = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def _base_forward(
        self, 
        sgrnas: list[str], 
        off_targets: list[str], 
        intervention: dict[str, float] | None = None
    ) -> dict[str, torch.Tensor]:
        
        if intervention is None:
            intervention = {}

        x_spacer, x_pam = self.encoder(sgrnas, off_targets)
        B = len(sgrnas)
        device = x_spacer.device

        # --- Nodo PAM ---
        if "pam_gate" in intervention:
            pam_gate = torch.full((B, 1), intervention["pam_gate"], device=device, dtype=torch.float32)
            _, repr_pam = self.pam_node(x_pam) 
        else:
            pam_gate, repr_pam = self.pam_node(x_pam)

        # --- Nodo Proximal ---
        if "proximal" in intervention:
            s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32)
            _, repr_prox = self.proximal_node(x_spacer)
        else:
            s_prox, repr_prox = self.proximal_node(x_spacer)

        # --- Nodo Seed Extension ---
        if "seed" in intervention:
            s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32)
            _, repr_seed = self.seed_node(x_spacer)
        else:
            s_seed, repr_seed = self.seed_node(x_spacer)

        # --- Nodo Non-Seed ---
        if "non_seed" in intervention:
            s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32)
            _, repr_nonseed = self.nonseed_node(x_spacer)
        else:
            s_nonseed, repr_nonseed = self.nonseed_node(x_spacer)

        # --- NORMALIZZAZIONE BIOLOGICA DELLE RAPPRESENTAZIONI ---
        # 1. Il PAM Gate DEVE essere una probabilità [0, 1] (AND logico)
        pam_gate = torch.sigmoid(pam_gate)

        # 2. Le penalità energetiche NON POSSONO essere negative. 
        # Usiamo ReLU: 0 = sequenza perfetta, >0 = danno da mismatch.
        s_prox = F.relu(s_prox)
        s_seed = F.relu(s_seed)
        s_nonseed = F.relu(s_nonseed)

        # --- Equazione Strutturale Combinata (HARD CONSTRAINTS TOTALI) ---
        # 1. Pesi termodinamici: SOLO penalità (w <= 0)
        w_prox_eff = -F.softplus(self.w_proximal)
        w_seed_eff = -F.softplus(self.w_seed)
        w_nonseed_eff = -F.softplus(self.w_nonseed)

        # 2. Bias biologico: L'attività basale non può essere < 1% o > 99.3%
        # Usiamo clamp sul tensore del parametro per limitarne l'influenza
        bias_eff = torch.clamp(self.bias, min=-4.0, max=3.0)

        # 3. Logit combinato
        logit = (s_prox * w_prox_eff) + (s_seed * w_seed_eff) + (s_nonseed * w_nonseed_eff) + bias_eff
        activity_prob = pam_gate * torch.sigmoid(logit)

        return {
            "pam_gate": pam_gate,
            "proximal_scalar": s_prox,
            "seed_scalar": s_seed,
            "nonseed_scalar": s_nonseed,
            "activity_probability": activity_prob,
            "repr_pam": repr_pam,
            "repr_proximal": repr_prox,
            "repr_seed": repr_seed,
            "repr_nonseed": repr_nonseed
        }

    def forward(self, sgrnas: list[str] | str, off_targets: list[str] | str) -> dict[str, torch.Tensor]:
        """Esecuzione standard osservazionale."""
        if isinstance(sgrnas, str): 
            sgrnas = [sgrnas]
        if isinstance(off_targets, str): 
            off_targets = [off_targets]
        return self._base_forward(sgrnas, off_targets)

    def do(self, sgrnas: list[str] | str, off_targets: list[str] | str, intervention: dict[str, float]) -> dict[str, torch.Tensor]:
        """Esecuzione sotto intervento causale (G-computation forward)."""
        if isinstance(sgrnas, str): 
            sgrnas = [sgrnas]
        if isinstance(off_targets, str): 
            off_targets = [off_targets]
        return self._base_forward(sgrnas, off_targets, intervention=intervention)

    def predict_proba_batch(self, sgrnas: list[str], off_targets: list[str]) -> torch.Tensor:
        """Restituisce Tensor[B] di probabilità — per training e valutazione."""
        out = self._base_forward(sgrnas, off_targets)
        return out["activity_probability"].squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, sgrna: str, off_seq: str) -> float:
        """Helper per inferenza singola (scollegato dal grafo computazionale)."""
        out = self.forward([sgrna], [off_seq])
        return float(out["activity_probability"].item())

    @torch.no_grad()
    def explain(self, sgrna: str, off_seq: str) -> dict[str, float]:
        """Restituisce il contributo esatto di ogni sottomodulo causale."""
        out = self.forward([sgrna], [off_seq])
        return {
            "pam_gate": float(out["pam_gate"].item()),
            "proximal_penalty": float(out["proximal_scalar"].item()),
            "seed_penalty": float(out["seed_scalar"].item()),
            "nonseed_penalty": float(out["nonseed_scalar"].item()),
            "final_probability": float(out["activity_probability"].item())
        }