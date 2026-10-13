from __future__ import annotations
import math
import torch

from .sgd_angular_reference import (
    inverse,
    stored_angle,
    tolerance,
    solve_update as legacy_solve,
)


def round_once_h_candidate(before, direction, lr):
    if lr == 0:
        return before.clone()
    w = before.double()
    u = direction.double()
    z = w - lr * u * before.norm().item() / direction.norm().clamp_min(1e-10).item()
    return (z * (w.norm() / z.norm())).to(before.dtype)


@torch.no_grad()
def solve_update(before, direction, target, family, weight_decay=0.0):
    try:
        lr, candidate, row = legacy_solve(
            before, direction, target, family, weight_decay
        )
        radius_error = (
            abs(candidate.double().norm().item() / before.double().norm().item() - 1.0)
            if family == "sgdh"
            else 0.0
        )
        row.update(
            angular_legacy_fallback=0,
            angular_fp64_retraction=0,
            angular_radius_relative_error=radius_error,
        )
        return lr, candidate, row
    except ValueError as exc:
        if not str(exc).startswith("Stored-weight angular error "):
            raise
        original_error = str(exc)
    w = before.double()
    u = direction.double()
    radius = w.norm()
    q = w / radius
    radial = (q * u).sum().item()
    perpendicular = (u - radial * q).norm().item()
    eta = inverse(
        target,
        radius.item(),
        radial,
        perpendicular,
        weight_decay if family == "sgd" else 0.0,
    )
    r32 = before.norm()
    n32 = direction.norm().clamp_min(1e-10)
    initial = eta if family == "sgd" else eta * n32.item() / r32.item()
    allowed = tolerance(target)
    trials = 0
    best = None

    def evaluate(lr, round_once):
        nonlocal trials, best
        if (
            not math.isfinite(lr)
            or lr < 0
            or (family == "sgd" and 1 - lr * weight_decay <= 0)
        ):
            return None
        if lr == 0:
            candidate = before.clone()
        elif family == "sgd":
            candidate = (
                before.clone().mul_(1 - lr * weight_decay).add_(direction, alpha=-lr)
            )
        elif round_once:
            candidate = round_once_h_candidate(before, direction, lr)
        else:
            z = before - lr * direction * r32 / n32
            candidate = z / z.norm() * r32
        actual = stored_angle(before, candidate)
        trials += 1
        error = abs(actual - target)
        row = (error, lr, actual, candidate, int(round_once))
        if best is None or error < best[0]:
            best = row
        return row

    # Quantized scalar curves need not be monotone. Do not limit the search to
    # [0.5,1.5] of the analytic inverse, or discard candidates by monotonicity.
    for mode in [False, True] if family == "sgdh" else [False]:
        for factor in [1.0, 0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0] + [
            i / 32 for i in range(1, 65)
        ]:
            evaluate(initial * factor, mode)
            if best[0] <= allowed:
                break
        if best[0] <= allowed:
            break
    if best[0] > allowed:
        raise ValueError(
            original_error + "; numeric v2 also failed, best_error=" + str(best[0])
        )
    error, lr, actual, candidate, round_once = best
    radius_error = (
        abs(candidate.double().norm().item() / radius.item() - 1.0)
        if family == "sgdh"
        else 0.0
    )
    # A round-once FP32 spherical projection cannot preserve radius bit exactly.
    # This bound verifies it is a rounding error rather than a scale update.
    if family == "sgdh" and radius_error > 5e-7:
        raise ValueError("H radius changed beyond FP32 rounding")
    row = dict(
        angle=actual,
        angular_error=actual - target,
        angular_tolerance=allowed,
        angular_tolerance_ratio=error / allowed,
        angular_solver_trials=trials + 25,
        angular_radial_component=radial,
        angular_perpendicular_norm=perpendicular,
        angular_analytic_lr=initial,
        angular_zero_lr_for_nonzero_target=int(lr == 0 and target > 0),
        angular_legacy_fallback=1,
        angular_fp64_retraction=round_once,
        angular_radius_relative_error=radius_error,
    )
    return lr, candidate, row
