from __future__ import annotations

import torch
import torch.nn as nn


class PAMModule(nn.Module):
    """
    Processa l'embedding del trinucleotide PAM (posizioni 20-22).
    Input: Tensor(B, 3, E)
    Output: Tuple[pam_gate Tensor(B, 1), representation Tensor(B, hidden_dim)]
    """
    def __init__(self, embed_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_pam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_pam.size(0)
        x_flat = x_pam.view(B, -1)
        
        representation = self.mlp(x_flat) 
        pam_gate = torch.sigmoid(self.head(representation))
        
        return pam_gate, representation


class SpacerRegionModule(nn.Module):
    """
    Modulo generico per processare sotto-regioni dello spacer (direzione 5' -> 3').
    Restituisce (scalar_output, representation).
    """
    def __init__(
        self, 
        start_idx: int, 
        end_idx: int, 
        embed_dim: int = 16, 
        use_transformer: bool = True,
        n_heads: int = 4,
        hidden_dim: int = 32
    ):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.seq_len = end_idx - start_idx
        self.use_transformer = use_transformer

        if self.use_transformer:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=n_heads, 
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=0.3
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.head = nn.Linear(embed_dim, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.seq_len * embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_spacer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Slicing naturale 5' -> 3'
        x_slice = x_spacer[:, self.start_idx:self.end_idx, :]
        B = x_slice.size(0)

        if self.use_transformer:
            x_slice = x_slice + self.pos_embedding
            encoded = self.transformer(x_slice)
            representation = encoded.mean(dim=1)  # GAP per rappresentazione globale
        else:
            x_flat = x_slice.contiguous().view(B, -1)
            representation = self.mlp(x_flat)

        scalar_out = self.head(representation)
        return scalar_out, representation


# --- Implementazioni Concrete con indici Biologici 5' -> 3' ---

class NonSeedModule(SpacerRegionModule):
    """Elabora le posizioni distali (5') dallo slot 0 al 7."""
    def __init__(self, embed_dim: int = 16):
        super().__init__(start_idx=0, end_idx=8, embed_dim=embed_dim, use_transformer=True)


class SeedExtensionModule(SpacerRegionModule):
    """Elabora la regione centrale dello spacer, slot 8 al 15."""
    def __init__(self, embed_dim: int = 16):
        super().__init__(start_idx=8, end_idx=16, embed_dim=embed_dim, use_transformer=True)


class ProximalModule(SpacerRegionModule):
    """Elabora le posizioni vicine al PAM (3'), slot 16 al 19."""
    def __init__(self, embed_dim: int = 16):
        # Usiamo MLP perché il Transformer su soli 4 token è inefficiente
        super().__init__(start_idx=16, end_idx=20, embed_dim=embed_dim, use_transformer=False)


class MismatchVectorModule(nn.Module):
    """
    Riceve il vettore binario di mismatch di una specifica regione.
    Applica una trasformazione non lineare per catturare l'epistasi posizionale.
    """
    def __init__(self, region_size: int, hidden_dim: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(region_size, hidden_dim),
            nn.LeakyReLU(0.1),  # Evita la morte dei neuroni nascosti
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),      # Forza output > 0 dolcemente: il gradiente non muore MAI
        )

    def forward(self, mm_vector_region: torch.Tensor) -> torch.Tensor:
        return self.fc(mm_vector_region)

class TypedMismatchModule(nn.Module):
    """
    Riceve il tensore one-hot dei tipi di mismatch (Match, Wobble, Transition, Transversion).
    Input shape: (Batch, region_size, 4)
    """
    def __init__(self, region_size: int, hidden_dim: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(), # Trasforma [B, region_size, 4] in [B, region_size * 4]
            nn.Linear(region_size * 4, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Vincolo biologico morbido (>=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)