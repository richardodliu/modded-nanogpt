"""Frozen Muon update rules extracted from the 20k alignment trainer."""

import math, json
from pathlib import Path
import torch
from torch import Tensor
import torch.distributed as dist
from nanogpt_research.geometry.angular import solve_muonh_lr, solve_muonwd_lr
from hashlib import sha256
from nanogpt_research.checkpoint import rank_world


def sha256_text(text):
    return sha256(text.encode()).hexdigest()


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        rank, world_size = rank_world()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                world_size - len(params) % world_size
            )
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    state["_tb_pre_projection_u"] = update.detach()
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )


def scale_invariant_update_(
    param: Tensor, update: Tensor, lr: float, eps: float = 1e-10
) -> None:
    """Hyperball-constrained step: take a Muon-orthogonalised update of size lr * ||param||,
    then renormalise back onto the Frobenius sphere of the parameter's initial radius. Preserves
    ||param|| exactly across training; the invariant lets us drop weight decay on hidden
    matrices entirely (the constraint already prevents norm growth)."""
    # Preserve the BF16 Newton-Schulz direction, but apply its scalar geometry
    # in the FP32 parameter dtype. BF16 intermediate scaling quantizes h and
    # makes some measured WD target angles unreachable by scalar refinement.
    update = update.to(dtype=param.dtype)
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)


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


def quantized_step_candidate(
    parameter: Tensor,
    update: Tensor,
    lr: float,
    mode: str,
    weight_decay: float,
) -> tuple[Tensor, float]:
    """Simulate the exact BF16 in-place update and return its realised angle."""
    candidate = parameter.detach().clone()
    if mode in ("muonh_matches_muonwd", "muonh_matches_muon"):
        scale_invariant_update_(candidate, update, lr)
    elif mode in (
        "muonwd_matches_muonh",
        "muon_matches_muonwd",
        "muon_matches_muonh",
        "muonwd_matches_muon",
    ):
        candidate.mul_(1 - lr * weight_decay)
        candidate.add_(update, alpha=-lr)
    else:
        raise ValueError(f"unsupported quantized alignment mode: {mode}")
    return candidate, stable_weight_step_angle(parameter, candidate)


def refine_quantized_lr(
    parameter: Tensor,
    update: Tensor,
    target_angle: float,
    analytical_lr: float,
    mode: str,
    weight_decay: float,
    iterations: int,
) -> tuple[float, Tensor, float, float, float]:
    """Refine the exact real-valued LR against the angle realised by BF16 storage."""
    if iterations <= 0:
        candidate, angle = quantized_step_candidate(
            parameter, update, analytical_lr, mode, weight_decay
        )
        return analytical_lr, candidate, angle, angle, angle
    if not math.isfinite(analytical_lr) or analytical_lr <= 0:
        raise RuntimeError(f"invalid analytical alignment LR: {analytical_lr}")

    evaluated: list[tuple[float, float, Tensor]] = []

    def evaluate(lr: float) -> float:
        candidate, angle = quantized_step_candidate(
            parameter, update, lr, mode, weight_decay
        )
        if not math.isfinite(angle):
            raise RuntimeError(f"non-finite quantized angle at lr={lr}")
        evaluated.append((abs(angle - target_angle), lr, candidate))
        return angle

    low = analytical_lr * 0.95
    high = analytical_lr * 1.05
    low_angle = evaluate(low)
    high_angle = evaluate(high)
    for _ in range(8):
        if low_angle <= target_angle <= high_angle:
            break
        if target_angle < low_angle:
            high, high_angle = low, low_angle
            low *= 0.5
            low_angle = evaluate(low)
        else:
            low, low_angle = high, high_angle
            high *= 2.0
            high_angle = evaluate(high)
    else:
        raise RuntimeError(
            "could not bracket target BF16 angle: "
            f"target={target_angle} low=({low},{low_angle}) high=({high},{high_angle})"
        )

    if high_angle < low_angle:
        raise RuntimeError(
            f"non-monotone BF16 angle bracket: low={low_angle} high={high_angle}"
        )
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        midpoint_angle = evaluate(midpoint)
        if midpoint_angle < target_angle:
            low, low_angle = midpoint, midpoint_angle
        else:
            high, high_angle = midpoint, midpoint_angle

    _, best_lr, best_candidate = min(evaluated, key=lambda item: item[0])
    best_angle = stable_weight_step_angle(parameter, best_candidate)
    return best_lr, best_candidate, best_angle, low_angle, high_angle


class MuonH(torch.optim.Optimizer):
    """MuonH: same Newton-Schulz orthogonalised direction as Muon, applied via a Frobenius-
    norm-preserving hyperball projection. Used here for ALL hidden 2D weight matrices —
    q, k, v, mlp.fc, attn.proj, mlp.proj — under non-zero (Kaiming-derived) init."""

    def __init__(self, params, lr=0.014, mu=0.95):
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        rank, world_size = rank_world()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                world_size - len(params) % world_size
            )
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    state["_tb_pre_projection_u"] = update.detach()
                    scale_invariant_update_(p, update, group["lr"])
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )


def load_alignment_targets(
    path: str, expected_names: list[str], train_steps: int
) -> dict:
    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError(f"alignment target file does not exist: {target_path}")
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported target schema: {payload.get('schema_version')!r}"
        )
    names = payload.get("parameter_names")
    rows = payload.get("target_radians")
    if not isinstance(names, list) or not isinstance(rows, list):
        raise ValueError(
            "target file must contain parameter_names and target_radians lists"
        )
    if set(names) != set(expected_names):
        missing = sorted(set(expected_names) - set(names))
        extra = sorted(set(names) - set(expected_names))
        raise ValueError(f"target parameter mismatch: missing={missing}, extra={extra}")
    if len(rows) != train_steps:
        raise ValueError(f"expected {train_steps} target rows, found {len(rows)}")
    name_to_column = {name: index for index, name in enumerate(names)}
    reordered_rows = []
    for step, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(names):
            raise ValueError(f"invalid target row {step}: expected {len(names)} values")
        reordered = [float(row[name_to_column[name]]) for name in expected_names]
        if not all(
            math.isfinite(value) and 0 <= value < math.pi / 2 for value in reordered
        ):
            raise ValueError(f"target row {step} contains invalid angles")
        reordered_rows.append(reordered)
    return {
        "path": str(target_path),
        "sha256": sha256_text(target_path.read_text(encoding="utf-8")),
        "parameter_names": list(expected_names),
        "target_radians": reordered_rows,
        "source": payload.get("source", {}),
    }


class LayerwiseAlignedMuon(torch.optim.Optimizer):
    """Muon direction with an exact per-step, per-matrix angular LR solver."""

    def __init__(
        self,
        named_params,
        target_payload,
        mode: str,
        weight_decay=0.0,
        mu=0.95,
        quantization_aware=False,
        refinement_iterations=12,
    ):
        if mode not in (
            "muonh_matches_muonwd",
            "muonwd_matches_muonh",
            "muon_matches_muonwd",
            "muon_matches_muonh",
            "muonwd_matches_muon",
            "muonh_matches_muon",
        ):
            raise ValueError(f"unknown layerwise alignment mode: {mode}")
        named_params = sorted(
            list(named_params), key=lambda item: item[1].size(), reverse=True
        )
        if not named_params:
            raise ValueError("layerwise alignment requires at least one matrix")
        params = [param for _, param in named_params]
        defaults = dict(lr=0.0, weight_decay=float(weight_decay), mu=float(mu))
        super().__init__(params, defaults)
        self.mode = mode
        self.name_by_id = {id(param): name for name, param in named_params}
        target_names = target_payload["parameter_names"]
        self.target_column = {name: index for index, name in enumerate(target_names)}
        self.target_radians = target_payload["target_radians"]
        self.quantization_aware = bool(quantization_aware)
        self.refinement_iterations = int(refinement_iterations)
        if set(self.name_by_id.values()) != set(self.target_column):
            raise ValueError("optimizer matrices do not match alignment target columns")
        self.target_step = None

    def set_target_step(self, step: int) -> None:
        if not 0 <= int(step) < len(self.target_radians):
            raise IndexError(f"alignment target step out of range: {step}")
        self.target_step = int(step)

    @torch.no_grad()
    def step(self):
        if self.target_step is None:
            raise RuntimeError(
                "set_target_step(step) must be called before every aligned update"
            )
        rank, world_size = rank_world()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                world_size - len(params) % world_size
            )
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    name = self.name_by_id[id(p)]
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    state["_tb_pre_projection_u"] = update.detach()

                    parameter = p.detach().float()
                    direction = update.detach().float()
                    weight_norm = parameter.norm().item()
                    update_norm = direction.norm().item()
                    inner = (parameter * direction).sum(dtype=torch.float64).item()
                    cosine = inner / max(weight_norm * update_norm, 1e-30)
                    target = self.target_radians[self.target_step][
                        self.target_column[name]
                    ]

                    if self.mode in ("muonh_matches_muonwd", "muonh_matches_muon"):
                        solved = solve_muonh_lr(target, cosine)
                    else:
                        solved = solve_muonwd_lr(
                            target,
                            weight_norm,
                            update_norm,
                            inner,
                            group["weight_decay"],
                        )

                    analytical_lr = float(solved.applied_lr)
                    if self.quantization_aware:
                        (
                            applied_lr,
                            candidate,
                            predicted_actual_angle,
                            bracket_low_angle,
                            bracket_high_angle,
                        ) = refine_quantized_lr(
                            p,
                            update,
                            target,
                            analytical_lr,
                            self.mode,
                            float(group["weight_decay"]),
                            self.refinement_iterations,
                        )
                        p.copy_(candidate)
                    else:
                        applied_lr = analytical_lr
                        if self.mode in ("muonh_matches_muonwd", "muonh_matches_muon"):
                            scale_invariant_update_(p, update, applied_lr)
                        else:
                            p.mul_(1 - applied_lr * group["weight_decay"])
                            p.add_(update, alpha=-applied_lr)
                        predicted_actual_angle = stable_weight_step_angle(parameter, p)
                        bracket_low_angle = predicted_actual_angle
                        bracket_high_angle = predicted_actual_angle

                    # Rank-sharded scalar evidence; TBMetrics reconstructs the 72 rows.
                    state["_tb_alignment_meta"] = (
                        float(target),
                        float(solved.tangent),
                        float(solved.numerator),
                        float(solved.denominator),
                        analytical_lr,
                        float(applied_lr),
                        float(solved.radial_component),
                        float(weight_norm),
                        float(update_norm),
                        float(inner),
                        float(cosine),
                        float(group["weight_decay"]),
                        float(predicted_actual_angle),
                        float(abs(predicted_actual_angle - target)),
                        float(
                            self.refinement_iterations if self.quantization_aware else 0
                        ),
                        float(bracket_low_angle),
                        float(bracket_high_angle),
                        float(bracket_high_angle - bracket_low_angle),
                    )
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )
        self.target_step = None
