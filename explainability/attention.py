from __future__ import annotations

import numpy as np


def aggregate_heads(attention: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Aggregate heads from shape (heads, tokens, tokens)."""
    attention = np.asarray(attention, dtype=float)
    if attention.ndim != 3:
        raise ValueError("Expected attention shape: (heads, tokens, tokens)")

    if mode == "mean":
        return attention.mean(axis=0)
    if mode == "max":
        return attention.max(axis=0)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def attention_rollout(attention_stack: np.ndarray, head_mode: str = "mean") -> np.ndarray:
    """Rollout attention through layers. Expected shape: (layers, heads, tokens, tokens)."""
    attention_stack = np.asarray(attention_stack, dtype=float)
    if attention_stack.ndim != 4:
        raise ValueError("Expected attention stack shape: (layers, heads, tokens, tokens)")

    n_layers, _, n_tokens, _ = attention_stack.shape
    rollout = np.eye(n_tokens, dtype=float)

    for layer in range(n_layers):
        layer_attn = aggregate_heads(attention_stack[layer], mode=head_mode)
        layer_attn = layer_attn + np.eye(n_tokens, dtype=float)
        layer_attn = layer_attn / layer_attn.sum(axis=-1, keepdims=True)
        rollout = layer_attn @ rollout
    return rollout
