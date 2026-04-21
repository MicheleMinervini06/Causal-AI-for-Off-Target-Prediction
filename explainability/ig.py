from __future__ import annotations

import numpy as np
import torch

try:
    from captum.attr import IntegratedGradients
except Exception:  # pragma: no cover
    IntegratedGradients = None


def integrated_gradients(
    model: torch.nn.Module,
    inputs: np.ndarray,
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
    target: int | None = None,
) -> np.ndarray:
    """Integrated Gradients attribution for differentiable models."""
    if IntegratedGradients is None:
        raise ImportError("captum is not installed. Run: uv sync")

    model.eval()
    input_tensor = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
    if baseline is None:
        baseline_tensor = torch.zeros_like(input_tensor)
    else:
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32)

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    ig = IntegratedGradients(forward_fn)
    attr = ig.attribute(input_tensor, baselines=baseline_tensor, target=target, n_steps=n_steps)
    return attr.detach().cpu().numpy()
