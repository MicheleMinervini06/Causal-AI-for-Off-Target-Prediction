from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import PairwiseTokenEncoder, BiologicalMismatchEncoder, BaseEncoder
from .modules import NonSeedModule, PAMModule, ProximalModule, SeedExtensionModule, MismatchVectorModule, TypedMismatchModule


class NeuralSCM(nn.Module):
    """
    Neural Structural Causal Model per previsioni CRISPR.
    Assembla moduli indipendenti in un DAG causale esplicito.
    """

    def __init__(
        self, 
        architecture: str = "mini_mlp", 
        embed_dim: int = 16, 
        hidden_dim: int = 4,  # SOLO UNO! Sarà 4 o 32 a seconda di cosa gli passiamo
        encoder: BaseEncoder | None = None,
        context_dim: int = 0
    ):
        super().__init__()
        self.architecture = architecture
        self.context_dim = context_dim

        # 1. Inizializzazione Encoder
        if encoder is None:
            encoder = PairwiseTokenEncoder(embed_dim=embed_dim)
        
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim

        # 2. Nodo PAM 
        self.pam_node = PAMModule(embed_dim=self.embed_dim, hidden_dim=hidden_dim)

        # 3. Inizializzazione ARCHITETTURA-DIPENDENTE
        if self.architecture == "mini_mlp":
            # Usa l'UNICO hidden_dim fornito (che in run.py assicureremo essere 4)
            self.nonseed_node = MismatchVectorModule(region_size=8, hidden_dim=hidden_dim)
            self.seed_node = MismatchVectorModule(region_size=8, hidden_dim=hidden_dim)
            self.proximal_node = MismatchVectorModule(region_size=4, hidden_dim=hidden_dim)
            
        elif self.architecture == "deep_scm":
            # Usa l'UNICO hidden_dim fornito (che in run.py assicureremo essere 32)
            self.proximal_node = ProximalModule(embed_dim=self.embed_dim)
            self.seed_node = SeedExtensionModule(embed_dim=self.embed_dim)
            self.nonseed_node = NonSeedModule(embed_dim=self.embed_dim)
            
        elif self.architecture == "linear_bypass":
            self.proximal_node = None
            self.seed_node = None
            self.nonseed_node = None

        elif self.architecture == "typed_mlp":
            # Run 10: MLP basate sul tipo di mismatch (input dimension = region_size * 4)
            self.nonseed_node = TypedMismatchModule(region_size=8, hidden_dim=hidden_dim)
            self.seed_node = TypedMismatchModule(region_size=8, hidden_dim=hidden_dim)
            self.proximal_node = TypedMismatchModule(region_size=4, hidden_dim=hidden_dim)

        elif self.architecture == "learned_mlp":
            # Run 12: End-to-end Representation Learning
            # 1. Istanziamo l'encoder DENTRO l'SCM perché il gradiente lo addestri
            self.pairwise_encoder = PairwiseTokenEncoder(embed_dim=4, use_learned_embeddings=True)
            
            # 2. Possiamo riciclare il modulo della Run 10 (aspetta region_size * 4)
            self.nonseed_node = TypedMismatchModule(region_size=8, hidden_dim=hidden_dim)
            self.seed_node = TypedMismatchModule(region_size=8, hidden_dim=hidden_dim)
            self.proximal_node = TypedMismatchModule(region_size=4, hidden_dim=hidden_dim)
            
        elif self.architecture == "context_aware_mlp":
            self.nonseed_node = TypedMismatchModule(region_size=8, hidden_dim=hidden_dim, input_dim_per_pos=8)
            self.seed_node = TypedMismatchModule(region_size=8, hidden_dim=hidden_dim, input_dim_per_pos=8)
            self.proximal_node = TypedMismatchModule(region_size=4, hidden_dim=hidden_dim, input_dim_per_pos=8)
        
        elif self.architecture == "positional_mlp":
            # Un singolo modulo "filtro" che processa ogni nucleotide in modo indipendente
            # L'input è 4 (Match, Wobble, Transition, Transversion)
            self.pos_node = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        else:
            raise ValueError(f"Architettura non riconosciuta: {self.architecture}")

        # 4. Parametri del Combiner
        self.w_proximal = nn.Parameter(torch.randn(1))
        self.w_seed = nn.Parameter(torch.randn(1))
        self.w_nonseed = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(1))
        if self.architecture == "positional_mlp":
            self.w_pos = nn.Parameter(torch.randn(20))

        # Nodo Esogeno (Contesto U)
        if self.context_dim > 0:
            # Una rete che mappa il GC Content in un "offset ambientale" (Logit)
            self.context_net = nn.Sequential(
                nn.Linear(self.context_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1) 
            )

    def _base_forward(
        self, 
        sgrnas: list[str], 
        off_targets: list[str], 
        context_features: torch.Tensor | None = None,
        intervention: dict[str, float] | None = None
    ) -> dict[str, torch.Tensor]:
        
        if intervention is None:
            intervention = {}

        # --- 1. PARTE COMUNE: ENCODER E NODO PAM ---
        x_spacer, x_pam = self.encoder(sgrnas, off_targets)
        B = len(sgrnas)
        device = x_spacer.device

        if "pam_gate" in intervention:
            pam_gate = torch.full((B, 1), intervention["pam_gate"], device=device, dtype=torch.float32)
            _, repr_pam = self.pam_node(x_pam) 
        else:
            pam_gate, repr_pam = self.pam_node(x_pam)

        # Inizializziamo le rappresentazioni a zero (servono solo per deep_scm per non rompere il dict)
        repr_prox = torch.zeros(B, self.embed_dim, device=device)
        repr_seed = torch.zeros(B, self.embed_dim, device=device)
        repr_nonseed = torch.zeros(B, self.embed_dim, device=device)
        s_prox = torch.zeros(B, 1, device=device)
        s_seed = torch.zeros(B, 1, device=device)
        s_nonseed = torch.zeros(B, 1, device=device)

        # =====================================================================
        # --- 2. SWITCH ARCHITETTURALE PER I NODI CAUSALI ---
        # =====================================================================
        
        if self.architecture == "deep_scm":
            # Run 7: Reti pesanti sugli embeddings completi
            assert self.proximal_node is not None and self.seed_node is not None and self.nonseed_node is not None

            if "proximal" in intervention:
                s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32)
                _, repr_prox = self.proximal_node(x_spacer)
            else:
                s_prox, repr_prox = self.proximal_node(x_spacer)

            if "seed" in intervention:
                s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32)
                _, repr_seed = self.seed_node(x_spacer)
            else:
                s_seed, repr_seed = self.seed_node(x_spacer)

            if "non_seed" in intervention:
                s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32)
                _, repr_nonseed = self.nonseed_node(x_spacer)
            else:
                s_nonseed, repr_nonseed = self.nonseed_node(x_spacer)

        elif self.architecture in ["linear_bypass", "mini_mlp"]:
            # Run 8 & 9: Entrambe usano la conta o il vettore esatto dei mismatch
            mismatches_batch = []
            for sg, ot in zip(sgrnas, off_targets):
                mm = [1.0 if sg[i] != ot[i] else 0.0 for i in range(20)]
                mismatches_batch.append(mm)
                
            # Tensore [B, 20]
            s_mismatch = torch.tensor(mismatches_batch, dtype=torch.float32, device=device)
            
            # Slicing allineato a models/deep/modules.py (0:8, 8:16, 16:20)
            mm_nonseed = s_mismatch[:, 0:8]
            mm_seed = s_mismatch[:, 8:16]
            mm_prox = s_mismatch[:, 16:20]

            if self.architecture == "linear_bypass":
                # Run 8: Somma bruta
                s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32) if "proximal" in intervention else mm_prox.sum(dim=1, keepdim=True)
                s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32) if "seed" in intervention else mm_seed.sum(dim=1, keepdim=True)
                s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32) if "non_seed" in intervention else mm_nonseed.sum(dim=1, keepdim=True)
            
            elif self.architecture == "mini_mlp":
                # Run 9: Mini-reti non-lineari
                assert self.proximal_node is not None and self.seed_node is not None and self.nonseed_node is not None
                s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32) if "proximal" in intervention else self.proximal_node(mm_prox)
                s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32) if "seed" in intervention else self.seed_node(mm_seed)
                s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32) if "non_seed" in intervention else self.nonseed_node(mm_nonseed)

        elif self.architecture == "typed_mlp":
            # Run 10: Estrazione One-Hot (Match, Wobble, Transition, Transversion)
            typed_batch = []
            
            # Helper per la classificazione
            def get_mismatch_type(sg_char, ot_char):
                if sg_char == ot_char:
                    return [1.0, 0.0, 0.0, 0.0] # 0: Match
                
                # Wobble (G-T o T-G mima l'RNA G-U)
                pair = {sg_char, ot_char}
                if pair == {'G', 'T'}:
                    return [0.0, 1.0, 0.0, 0.0] # 1: Wobble
                
                # Transitions (A<->G, C<->T)
                if pair in [{'A', 'G'}, {'C', 'T'}]:
                    return [0.0, 0.0, 1.0, 0.0] # 2: Transition
                
                # Tutto il resto è Transversion (Cambiamento della struttura chimica)
                return [0.0, 0.0, 0.0, 1.0]     # 3: Transversion

            for sg, ot in zip(sgrnas, off_targets):
                seq_encoding = [get_mismatch_type(sg[i], ot[i]) for i in range(20)]
                typed_batch.append(seq_encoding)
                
            # Tensore [Batch, 20, 4]
            s_typed = torch.tensor(typed_batch, dtype=torch.float32, device=device)
            
            # Slicing
            mm_nonseed = s_typed[:, 0:8, :]   # [B, 8, 4]
            mm_seed = s_typed[:, 8:16, :]     # [B, 8, 4]
            mm_prox = s_typed[:, 16:20, :]    # [B, 4, 4]

            s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32) if "proximal" in intervention else self.proximal_node(mm_prox)
            s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32) if "seed" in intervention else self.seed_node(mm_seed)
            s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32) if "non_seed" in intervention else self.nonseed_node(mm_nonseed)
        
        elif self.architecture == "learned_mlp":
            # Run 12: Estrazione Appresa
            # L'encoder restituisce direttamente un tensore [B, 20, 4] già sul device corretto
            s_learned = self.pairwise_encoder.encode(sgrnas, off_targets)
            
            # Slicing posizionale (identico alla Run 10)
            mm_nonseed = s_learned[:, 0:8, :]   # [B, 8, 4]
            mm_seed = s_learned[:, 8:16, :]     # [B, 8, 4]
            mm_prox = s_learned[:, 16:20, :]    # [B, 4, 4]

            # Passaggio nei nodi causali
            s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32) if "proximal" in intervention else self.proximal_node(mm_prox)
            s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32) if "seed" in intervention else self.seed_node(mm_seed)
            s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32) if "non_seed" in intervention else self.nonseed_node(mm_nonseed)

        elif self.architecture == "context_aware_mlp":
            # Run 13: Estrazione Context-Aware (Mismatch + Identità Base sgRNA)
            # L'encoder restituisce direttamente un tensore [B, 20, 8]
            s_context = self.encoder.encode(sgrnas, off_targets)
            
            # Slicing posizionale (identico alla Run 10, ma l'ultima dimensione è 8)
            mm_nonseed = s_context[:, 0:8, :]   # [B, 8, 8]
            mm_seed = s_context[:, 8:16, :]     # [B, 8, 8]
            mm_prox = s_context[:, 16:20, :]    # [B, 4, 8]

            # Passaggio nei nodi causali (con supporto per gli interventi del Do-Calculus)
            s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32) if "proximal" in intervention else self.proximal_node(mm_prox)
            s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32) if "seed" in intervention else self.seed_node(mm_seed)
            s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32) if "non_seed" in intervention else self.nonseed_node(mm_nonseed)
        
        elif self.architecture == "positional_mlp":
            typed_batch = []
            
            def get_mismatch_type(sg_char, ot_char):
                if sg_char == ot_char: return [1.0, 0.0, 0.0, 0.0]
                pair = {sg_char, ot_char}
                if pair == {'G', 'T'}: return [0.0, 1.0, 0.0, 0.0]
                if pair in [{'A', 'G'}, {'C', 'T'}]: return [0.0, 0.0, 1.0, 0.0]
                return [0.0, 0.0, 0.0, 1.0]

            for sg, ot in zip(sgrnas, off_targets):
                seq_encoding = [get_mismatch_type(sg[i], ot[i]) for i in range(20)]
                typed_batch.append(seq_encoding)
                
            # Tensore [Batch, 20, 4]
            s_typed = torch.tensor(typed_batch, dtype=torch.float32, device=device)
            
            # Passiamo tutte le posizioni attraverso la MLP condivisa
            # Output: [Batch, 20, 1] -> squeeze -> [Batch, 20]
            pos_penalties = self.pos_node(s_typed).squeeze(-1)
            pos_penalties = F.relu(pos_penalties) # Normalizza output negativi
            
            # Salviamo nei vecchi scalari la somma delle zone solo per retrocompatibilità coi log
            s_nonseed = pos_penalties[:, 0:8].sum(dim=1, keepdim=True)
            s_seed = pos_penalties[:, 8:16].sum(dim=1, keepdim=True)
            s_prox = pos_penalties[:, 16:20].sum(dim=1, keepdim=True)
        
        else:
            raise ValueError(f"Architettura non riconosciuta: {self.architecture}")

        # =====================================================================
        # --- 3. PARTE COMUNE: NORMALIZZAZIONE E HARD PRIOR ---
        # =====================================================================
        pam_gate = torch.sigmoid(pam_gate)
        
        # Le ReLU normalizzano output anomali
        s_prox = F.relu(s_prox)
        s_seed = F.relu(s_seed)
        s_nonseed = F.relu(s_nonseed)

        # Hard Prior: Seed Dominance (|w_seed_eff| >= |w_nonseed_eff|)
        w_nonseed_base = F.softplus(self.w_nonseed)
        w_nonseed_eff = -w_nonseed_base
        
        w_seed_extra = F.softplus(self.w_seed)
        w_seed_eff = -(w_nonseed_base + w_seed_extra)
        
        w_prox_eff = -F.softplus(self.w_proximal)

        bias_eff = torch.clamp(self.bias, min=-4.0, max=3.0)

        if self.architecture == "positional_mlp":
        # HARD PRIOR POSIZIONALE: Tutti i 20 pesi devono essere <= 0
            w_pos_eff = -F.softplus(self.w_pos) # [20]
            # Moltiplica ogni penalità per il suo peso e somma (Prodotto scalare)
            thermo_logit = torch.sum(pos_penalties * w_pos_eff, dim=1, keepdim=True) + bias_eff
        else:
            # Logit Termodinamico Puro (V)
            thermo_logit = (s_prox * w_prox_eff) + (s_seed * w_seed_eff) + (s_nonseed * w_nonseed_eff) + bias_eff

        # --- IBRIDAZIONE CAUSALE (Iniezione di U) ---  <--- NUOVO BLOCCO
        final_logit = thermo_logit
        context_logit = torch.zeros_like(thermo_logit)
        
        if self.context_dim > 0 and context_features is not None:
            context_logit = self.context_net(context_features)
            final_logit = thermo_logit + context_logit # Somma additiva: Fisica + Ambiente
        
        activity_prob = pam_gate * torch.sigmoid(final_logit)

        return {
            "pam_gate": pam_gate,
            "proximal_scalar": s_prox,
            "seed_scalar": s_seed,
            "nonseed_scalar": s_nonseed,
            "thermo_logit": thermo_logit,
            "context_logit": context_logit,
            "logit": final_logit,
            "activity_probability": activity_prob,
            "repr_pam": repr_pam,
            "repr_proximal": repr_prox,
            "repr_seed": repr_seed,
            "repr_nonseed": repr_nonseed
        }

    def forward(self, sgrnas: list[str] | str, off_targets: list[str] | str, context_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if isinstance(sgrnas, str): sgrnas = [sgrnas]
        if isinstance(off_targets, str): off_targets = [off_targets]
        return self._base_forward(sgrnas, off_targets, context_features=context_features)

    def do(self, sgrnas: list[str] | str, off_targets: list[str] | str, intervention: dict[str, float], context_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if isinstance(sgrnas, str): sgrnas = [sgrnas]
        if isinstance(off_targets, str): off_targets = [off_targets]
        return self._base_forward(sgrnas, off_targets, context_features=context_features, intervention=intervention)

    def predict_proba_batch(self, sgrnas: list[str], off_targets: list[str], context_features: torch.Tensor | None = None) -> torch.Tensor:
        out = self._base_forward(sgrnas, off_targets, context_features=context_features)
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