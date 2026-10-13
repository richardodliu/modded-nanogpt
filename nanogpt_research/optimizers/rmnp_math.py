"""Small, testable helpers for the RMNP training specialization."""

from __future__ import annotations

import math

import torch.nn.functional as F
from torch import Tensor


def rmnp_momentum_direction(
    grad: Tensor,
    momentum_buffer: Tensor,
    momentum: float,
    *,
    nesterov: bool = True,
) -> Tensor:
    """Apply the official RMNP momentum update and return its preconditioner input."""
    momentum_buffer.mul_(momentum).add_(grad)
    if nesterov:
        return grad.add(momentum_buffer, alpha=momentum)
    return momentum_buffer


def row_normalize_native(direction: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize native (output, input) rows without Muon's tall-matrix transpose."""
    return F.normalize(direction, p=2, dim=-1, eps=eps)


def rmnp_aspect_scale(direction: Tensor) -> float:
    """Return the official Muon-compatible max(1, output/input)^0.5 scale."""
    if direction.ndim != 2:
        raise ValueError(
            f"RMNP expects a 2D matrix, got shape={tuple(direction.shape)}"
        )
    return max(1.0, direction.size(-2) / direction.size(-1)) ** 0.5


def cosine_warmup_eta(
    step: int,
    train_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Return a warmup+cosine multiplier for the update at zero-based ``step``."""
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if not 0 <= step < train_steps:
        raise ValueError(f"step {step} is outside [0, {train_steps})")
    if not 0 <= warmup_steps < train_steps:
        raise ValueError("warmup_steps must lie in [0, train_steps)")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must lie in [0, 1]")
    completed = step + 1
    if warmup_steps > 0 and completed <= warmup_steps:
        return completed / warmup_steps
    progress = (completed - warmup_steps) / (train_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
