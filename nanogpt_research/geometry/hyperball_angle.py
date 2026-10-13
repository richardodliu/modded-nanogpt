"""Pure fixed-angle hyperball geometry; independent of the Megatron adapter."""

import math
import torch


def tensor_norm(tensor: torch.Tensor) -> float:
    # A float64 accumulation improves scalar control accuracy without FP64
    # matrix multiplication or full-size double copies of model parameters.
    return math.sqrt(float(tensor.square().sum(dtype=torch.float64)))


def angle_between(a: torch.Tensor, b: torch.Tensor) -> float:
    """Stable finite angle, including tiny angles where acos loses precision."""
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    ar, br = tensor_norm(a), tensor_norm(b)
    if ar == 0 or br == 0:
        raise ValueError("Angle has a zero endpoint")
    ua, ub = a / ar, b / br
    return 2.0 * math.atan2(tensor_norm(ua - ub), tensor_norm(ua + ub))


def fixed_angle_target(
    weight: torch.Tensor, direction: torch.Tensor, radius: float, theta: float
):
    """Return geodesic target and audit scalars, preserving the given direction.

    Zero direction / numerically radial direction: explicitly skip the angle.
    Zero theta: exact identity.  Positive but unreachable Hyperball inverse:
    raise, with no negative step, clipping or alternative direction.
    """
    if weight.shape != direction.shape or weight.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("Control requires matching FP32/FP64 weight and direction")
    if not (
        math.isfinite(radius)
        and radius > 0
        and math.isfinite(theta)
        and 0 <= theta < math.pi / 2
    ):
        raise ValueError((radius, theta))
    r = tensor_norm(weight)
    p = tensor_norm(direction)
    if not (math.isfinite(r) and r > 0 and math.isfinite(p)):
        raise ValueError(f"Nonfinite/zero weight or direction norm: {r}, {p}")
    scalars = {
        "weight_norm_pre": r,
        "direction_norm": p,
        "radius_target": radius,
        "theta_target": theta,
        "status": "applied",
        "inverse_h": None,
        "c": None,
        "tangent_norm": 0.0,
        "tangent_ratio": 0.0,
    }
    if theta == 0:
        scalars["status"] = "zero_angle"
        return weight.clone(), scalars
    if p == 0:
        scalars["status"] = "zero_direction_skip"
        return weight.clone(), scalars
    dot = float((weight * direction).sum(dtype=torch.float64))
    c = dot / (r * p)
    tangent = direction - weight * (dot / (r * r))
    # Remove the tiny FP32 residual parallel component before normalization.
    correction = float((weight * tangent).sum(dtype=torch.float64)) / (r * r)
    tangent.add_(weight, alpha=-correction)
    d = tensor_norm(tangent)
    q = d / p
    scalars.update(c=c, tangent_norm=d, tangent_ratio=q)
    if q <= 1e-7:
        scalars["status"] = "degenerate_tangent_skip"
        return weight.clone(), scalars
    denominator = q * math.cos(theta) + c * math.sin(theta)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            f"Positive Hyperball inverse unreachable: theta={theta}, c={c}, q={q}, denominator={denominator}"
        )
    scalars["inverse_h"] = math.sin(theta) / denominator
    target = weight * (radius * math.cos(theta) / r)
    target.add_(tangent, alpha=-radius * math.sin(theta) / d)
    # Roundoff-only radius correction; no controller or target-angle clipping.
    target.mul_(radius / tensor_norm(target))
    return target, scalars
