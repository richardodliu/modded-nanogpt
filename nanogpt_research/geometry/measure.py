"""Measured angle of the actual stored weights."""

from __future__ import annotations
import torch
from torch import Tensor


def stable_weight_step_angle(before: Tensor, after: Tensor) -> float:
    """Match TBMetrics' stable atan2 angle on the actually stored tensor dtype."""
    before_value = before.detach().float()
    after_value = after.detach().float()
    before_sq = before_value.square().sum(dtype=torch.float64)
    after_sq = after_value.square().sum(dtype=torch.float64)
    if before_sq.item() <= 0 or after_sq.item() <= 0:
        raise RuntimeError("cannot measure a weight-step angle for a zero-norm tensor")
    dot = (before_value * after_value).sum(dtype=torch.float64)
    radial_coefficient = dot / before_sq
    tangential = after_value - radial_coefficient.to(before_value.dtype) * before_value
    tangential_norm = tangential.square().sum(dtype=torch.float64).clamp_min(0).sqrt()
    radial_component = dot / before_sq.sqrt()
    return torch.atan2(tangential_norm, radial_component).item()
