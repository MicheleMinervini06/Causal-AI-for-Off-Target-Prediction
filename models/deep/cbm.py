from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ConceptBottleneckModel(nn.Module):
    def __init__(self, input_dim: int, concept_dim: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        self.concept_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, concept_dim),
            nn.Sigmoid(),
        )
        self.task_head = nn.Linear(concept_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concepts = self.concept_head(x)
        logits = self.task_head(concepts).squeeze(-1)
        return logits, concepts


class CBMClassifier:
    def __init__(self, input_dim: int, concept_dim: int = 8, hidden_dim: int = 32) -> None:
        self.model = ConceptBottleneckModel(
            input_dim=input_dim,
            concept_dim=concept_dim,
            hidden_dim=hidden_dim,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> "CBMClassifier":
        self.model.train()
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                logits, _ = self.model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(x, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()

    def explain(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            _, concepts = self.model(torch.tensor(x, dtype=torch.float32))
        return concepts.cpu().numpy()
