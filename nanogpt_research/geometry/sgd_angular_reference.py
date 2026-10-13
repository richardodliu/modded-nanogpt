"""Exact angular inverse, followed by verification on the actual FP32 update.

The target is the reference's measured stored-weight angle, in rad/update.
Neither the target nor the update direction is clipped or replaced.
"""

import math
import torch

ANGLE_ATOL = 2e-8
ANGLE_RTOL = 2e-5


def tolerance(target):
    return max(ANGLE_ATOL, ANGLE_RTOL * abs(target))


def stored_angle(before, after):
    # Deliberately identical to the completed reference's evaluation.angle.
    a, b = before.detach().float(), after.detach().float()
    aa = a.square().sum(dtype=torch.float64)
    dot = (a * b).sum(dtype=torch.float64)
    if aa <= 0 or b.square().sum(dtype=torch.float64) <= 0:
        raise ValueError("zero-norm weight in angle measurement")
    tangent = b - (dot / aa).to(a.dtype) * a
    return torch.atan2(
        tangent.square().sum(dtype=torch.float64).sqrt(), dot / aa.sqrt()
    ).item()


def inverse(target, radius, radial, perpendicular, weight_decay=0.0):
    """Solve the additive SGD LR, also used before the H radius retraction."""
    if not all(
        math.isfinite(v) for v in (target, radius, radial, perpendicular, weight_decay)
    ):
        raise ValueError("Nonfinite angular geometry")
    if (
        radius <= 0
        or perpendicular < 0
        or weight_decay < 0
        or not 0 <= target < math.pi
    ):
        raise ValueError("Invalid angular geometry")
    if target == 0:
        return 0.0
    if perpendicular == 0:
        raise ValueError("Nonzero target angle is unreachable for a collinear update")
    sine, cosine = math.sin(target), math.cos(target)
    denominator = perpendicular * cosine + (radial + weight_decay * radius) * sine
    if denominator <= 0:
        raise ValueError(
            "Target angle is unreachable at finite nonnegative learning rate"
        )
    eta = radius * sine / denominator
    if not math.isfinite(eta) or eta < 0 or 1 - eta * weight_decay <= 0:
        raise ValueError("Angular solution violates the positive-decay update branch")
    return eta


@torch.no_grad()
def solve_update(before, direction, target, family, weight_decay=0.0):
    if family not in ("sgd", "sgdh"):
        raise ValueError(family)
    if not math.isfinite(target) or not 0 <= target < math.pi:
        raise ValueError("Invalid angular target")
    w, u = before.double(), direction.double()
    radius = w.norm().item()
    if not math.isfinite(radius) or radius <= 0 or not torch.isfinite(u).all():
        raise ValueError("Invalid angular parameter or update")
    q = w / radius
    radial = (q * u).sum().item()
    perpendicular = (u - radial * q).norm().item()
    additive = inverse(
        target, radius, radial, perpendicular, weight_decay if family == "sgd" else 0.0
    )
    radius32 = before.norm()
    direction_norm32 = direction.norm().clamp_min(1e-10)
    # The H implementation uses FP32 reductions. Convert to its exact LR units.
    initial_lr = (
        additive
        if family == "sgd"
        else additive * direction_norm32.item() / radius32.item()
    )
    del w, u, q

    def evaluate(lr):
        if (
            lr < 0
            or not math.isfinite(lr)
            or family == "sgd"
            and 1 - lr * weight_decay <= 0
        ):
            return None
        if lr == 0:
            candidate = before.clone()
        elif family == "sgd":
            candidate = (
                before.clone().mul_(1 - lr * weight_decay).add_(direction, alpha=-lr)
            )
        else:
            candidate = before - lr * direction * radius32 / direction_norm32
            norm = candidate.norm()
            if not torch.isfinite(norm) or norm <= 0:
                return None
            candidate = candidate / norm * radius32
        angle = stored_angle(before, candidate)
        return (abs(angle - target), lr, angle, candidate)

    best = evaluate(initial_lr)
    if best is None:
        raise ValueError("Nonfinite angular candidate")
    trials = 1
    allowed = tolerance(target)
    if best[0] > allowed and target <= allowed:
        # Below the predeclared storage-resolution tolerance, an unchanged FP32
        # matrix can be closer than a positive LR plus rounded H retraction.
        # Keep the nonzero target and residual in the log; never label it exact.
        zero = evaluate(0.0)
        trials += 1
        if zero[0] < best[0]:
            best = zero
    if best[0] > allowed:
        # Quantization can perturb the analytic inverse. Search its local scalar
        # LR neighborhood deterministically, retaining the unchanged target/U.
        low, high = initial_lr * 0.5, initial_lr * 1.5
        for _ in range(24):
            lr = (low + high) / 2
            row = evaluate(lr)
            trials += 1
            if row is None:
                high = lr
                continue
            if row[0] < best[0]:
                best = row
            if best[0] <= allowed:
                break
            if row[2] < target:
                low = lr
            else:
                high = lr
    error, lr, actual, candidate = best
    if error > allowed:
        raise ValueError(
            f"Stored-weight angular error {error} exceeds {allowed}; target={target}, actual={actual}, family={family}"
        )
    return (
        lr,
        candidate,
        dict(
            angle=actual,
            angular_error=actual - target,
            angular_tolerance=allowed,
            angular_tolerance_ratio=error / allowed,
            angular_solver_trials=trials,
            angular_radial_component=radial,
            angular_zero_lr_for_nonzero_target=int(lr == 0 and target > 0),
            angular_perpendicular_norm=perpendicular,
            angular_analytic_lr=initial_lr,
        ),
    )
