"""Radial coupling without experiment or logging dependencies."""

from __future__ import annotations
import math
import torch


def before_geometry(named):
    weights, geometry = [], []
    for name, parameter in named:
        if parameter.ndim != 2 or parameter.grad is None:
            raise ValueError(f"Expected a matrix with a raw gradient: {name}")
        weight = parameter.detach().clone()
        w = weight.double()
        g = parameter.grad.detach().double()
        weights.append(weight)
        geometry.append(torch.stack(((g * w).sum(), w.square().sum())))
    result = torch.stack(geometry)
    if not torch.isfinite(result).all() or not (result[:, 1] > 0).all():
        raise FloatingPointError(
            "Undefined radial metric: nonfinite gradient/weight or zero weight norm"
        )
    return weights, result


def after_geometry(named, before, geometry):
    # Subtract in FP64 to measure the difference of the stored W_{t+1}, W_t
    # exactly, rather than substituting an optimizer direction or eta*direction.
    inner = torch.stack(
        [
            ((p.detach().double() - w.double()) * w.double()).sum()
            for (_, p), w in zip(named, before, strict=True)
        ]
    )
    values = geometry[:, 0] * inner / geometry[:, 1]
    result = torch.stack((geometry[:, 0], inner, geometry[:, 1], values), dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite actual-update radial metric")
    return result.cpu().tolist()


def aggregate(names, rows):
    blocks = {}
    for name, row in zip(names, rows, strict=True):
        block = name.removeprefix("_orig_mod.").removeprefix("blocks.").split(".")[0]
        blocks.setdefault(block, []).append(row[3])
    return math.fsum(r[3] for r in rows), {k: math.fsum(v) for k, v in blocks.items()}
