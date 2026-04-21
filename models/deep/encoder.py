from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TOKEN_MAP = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
PAD_TOKEN = 0


def encode_sequence(sequence: str, max_len: int = 23) -> list[int]:
    encoded = [TOKEN_MAP.get(base.upper(), TOKEN_MAP["N"]) for base in sequence[:max_len]]
    if len(encoded) < max_len:
        encoded.extend([PAD_TOKEN] * (max_len - len(encoded)))
    return encoded


def encode_pair_batch(
    guides: Iterable[str],
    targets: Iterable[str],
    max_len: int = 23,
) -> tuple[np.ndarray, np.ndarray]:
    guide_tokens = np.asarray([encode_sequence(s, max_len=max_len) for s in guides], dtype=np.int64)
    target_tokens = np.asarray([encode_sequence(s, max_len=max_len) for s in targets], dtype=np.int64)
    return guide_tokens, target_tokens


class PairwiseTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int = 6, d_model: int = 32, nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, guide_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        guide_emb = self.embedding(guide_tokens)
        target_emb = self.embedding(target_tokens)
        pair_emb = guide_emb + target_emb
        hidden = self.encoder(pair_emb)
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def fit(
        self,
        x: tuple[np.ndarray, np.ndarray],
        y: np.ndarray,
        epochs: int = 8,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> "PairwiseTransformerClassifier":
        self.train()
        guide_tokens, target_tokens = x
        dataset = TensorDataset(
            torch.tensor(guide_tokens, dtype=torch.long),
            torch.tensor(target_tokens, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            for guide_batch, target_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self(guide_batch, target_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        self.eval()
        guide_tokens, target_tokens = x
        with torch.no_grad():
            logits = self(
                torch.tensor(guide_tokens, dtype=torch.long),
                torch.tensor(target_tokens, dtype=torch.long),
            )
        return torch.sigmoid(logits).cpu().numpy()

    def explain(self, x: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        self.eval()
        guide_tokens, target_tokens = x
        with torch.no_grad():
            guide_emb = self.embedding(torch.tensor(guide_tokens, dtype=torch.long))
            target_emb = self.embedding(torch.tensor(target_tokens, dtype=torch.long))
            token_importance = (guide_emb - target_emb).abs().mean(dim=-1)
        return token_importance.cpu().numpy()
