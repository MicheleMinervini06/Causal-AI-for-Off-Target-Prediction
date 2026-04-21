from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_bio_loss(attention_map: torch.Tensor, bio_prior: torch.Tensor) -> torch.Tensor:
    """KL divergence between normalized attention and empirical biological prior."""
    attention = attention_map / (attention_map.sum(dim=-1, keepdim=True) + 1e-8)
    prior = bio_prior / (bio_prior.sum(dim=-1, keepdim=True) + 1e-8)
    return F.kl_div((attention + 1e-8).log(), prior, reduction="batchmean")


def contrastive_explanation_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Triplet-style loss on explanation vectors."""
    pos = F.pairwise_distance(anchor, positive)
    neg = F.pairwise_distance(anchor, negative)
    return torch.relu(pos - neg + margin).mean()


L_attention_bio = attention_bio_loss
L_contrastive_explanation = contrastive_explanation_loss
