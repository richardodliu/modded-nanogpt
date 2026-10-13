"""Exact per-matrix angular learning-rate maps for MuonWD and MuonH.

The solvers operate on scalar Frobenius geometry measured immediately before an
optimizer update.  They intentionally do not clamp an invalid solution: the
alignment experiment must expose singular or non-finite geometry instead of
silently changing the requested target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AlignmentSolve:
    target_angle_rad: float
    tangent: float
    numerator: float
    denominator: float
    applied_lr: float
    radial_component: float


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _target_tangent(target_angle_rad: float) -> tuple[float, float]:
    angle = _finite("target_angle_rad", target_angle_rad)
    if angle < 0 or angle >= math.pi / 2:
        raise ValueError(f"target angle must lie in [0, pi/2), got {angle}")
    return angle, math.tan(angle)


def _cosine_and_perpendicular(cosine: float) -> tuple[float, float]:
    cosine = _finite("cosine", cosine)
    tolerance = 1e-12
    if cosine < -1 - tolerance or cosine > 1 + tolerance:
        raise ValueError(f"cosine outside [-1, 1]: {cosine}")
    cosine = max(-1.0, min(1.0, cosine))
    return cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))


def solve_muonh_lr(
    target_angle_rad: float, cosine: float, eps: float = 1e-12
) -> AlignmentSolve:
    """Solve the dimensionless MuonH LR that realizes ``target_angle_rad``.

    MuonH forms ``Z = W - h U ||W||/||U||`` and then renormalizes ``Z`` to
    ``||W||``.  Renormalization preserves the angle, so

        tan(phi) = h sqrt(1-c^2) / (1-h c).
    """

    angle, tangent = _target_tangent(target_angle_rad)
    cosine, perpendicular = _cosine_and_perpendicular(cosine)
    if tangent == 0:
        return AlignmentSolve(angle, tangent, 0.0, perpendicular, 0.0, 1.0)
    numerator = tangent
    denominator = perpendicular + tangent * cosine
    if not math.isfinite(denominator) or denominator <= eps:
        raise ValueError(f"invalid MuonH inverse denominator: {denominator}")
    lr = numerator / denominator
    radial = 1.0 - lr * cosine
    if not math.isfinite(lr) or lr < 0:
        raise ValueError(f"invalid MuonH inverse LR: {lr}")
    if not math.isfinite(radial) or radial <= eps:
        raise ValueError(
            f"MuonH inverse step crosses the angular hemisphere: radial={radial}"
        )
    return AlignmentSolve(angle, tangent, numerator, denominator, lr, radial)


def solve_muonwd_lr(
    target_angle_rad: float,
    weight_norm: float,
    update_norm: float,
    inner: float,
    weight_decay: float,
    eps: float = 1e-12,
) -> AlignmentSolve:
    """Solve the MuonWD LR, including LR-coupled decoupled weight decay.

    For ``W+ = (1-eta*lambda)W - eta*U``, this solves

        tan(phi) = eta ||U_perp|| /
                   ((1-eta*lambda)||W|| - eta ||U|| c).

    The returned LR must be used for both the decay multiplier and the additive
    Muon update to match the implementation under test.
    """

    angle, tangent = _target_tangent(target_angle_rad)
    weight_norm = _finite("weight_norm", weight_norm)
    update_norm = _finite("update_norm", update_norm)
    inner = _finite("inner", inner)
    weight_decay = _finite("weight_decay", weight_decay)
    if weight_norm <= 0 or update_norm <= 0:
        raise ValueError(
            f"weight_norm and update_norm must be positive, got {weight_norm}, {update_norm}"
        )
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be nonnegative, got {weight_decay}")
    cosine, perpendicular = _cosine_and_perpendicular(
        inner / (weight_norm * update_norm)
    )
    if tangent == 0:
        return AlignmentSolve(
            angle, tangent, 0.0, update_norm * perpendicular, 0.0, weight_norm
        )
    numerator = tangent * weight_norm
    denominator = update_norm * perpendicular + tangent * (
        weight_decay * weight_norm + update_norm * cosine
    )
    scale = max(
        update_norm,
        abs(tangent) * (weight_decay * weight_norm + update_norm),
        1.0,
    )
    if not math.isfinite(denominator) or denominator <= eps * scale:
        raise ValueError(f"invalid MuonWD inverse denominator: {denominator}")
    lr = numerator / denominator
    radial = (1.0 - lr * weight_decay) * weight_norm - lr * update_norm * cosine
    if not math.isfinite(lr) or lr < 0:
        raise ValueError(f"invalid MuonWD inverse LR: {lr}")
    if not math.isfinite(radial) or radial <= eps * weight_norm:
        raise ValueError(
            f"MuonWD inverse step crosses the angular hemisphere: radial={radial}"
        )
    return AlignmentSolve(angle, tangent, numerator, denominator, lr, radial)


def reconstructed_angle_muonh(lr: float, cosine: float) -> float:
    cosine, perpendicular = _cosine_and_perpendicular(cosine)
    return math.atan2(lr * perpendicular, 1.0 - lr * cosine)


def reconstructed_angle_muonwd(
    lr: float,
    weight_norm: float,
    update_norm: float,
    inner: float,
    weight_decay: float,
) -> float:
    cosine, perpendicular = _cosine_and_perpendicular(
        inner / (weight_norm * update_norm)
    )
    tangential = lr * update_norm * perpendicular
    radial = (1.0 - lr * weight_decay) * weight_norm - lr * update_norm * cosine
    return math.atan2(tangential, radial)
