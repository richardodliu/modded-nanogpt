"""Sharded AdamW/AdamH with measured per-matrix angle or norm alignment.

The Adam direction is bias-corrected m / (sqrt(v) + eps), with v accumulated
from raw gradients. Both optimizers use exactly this direction and FP32 state.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from nanogpt_research.geometry.angular import solve_muonh_lr, solve_muonwd_lr


def load_targets(path, names, steps, kind):
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    field = "target_radians" if kind == "angle" else "target_norm_eff"
    assert payload["schema_version"] == (1 if kind == "angle" else 2)
    assert len(payload["parameter_names"]) == len(names)
    assert set(payload["parameter_names"]) == set(names)
    assert payload["train_steps"] == steps and len(payload[field]) == steps
    upper = math.pi / 2 if kind == "angle" else math.inf
    assert all(
        len(row) == len(names) and all(math.isfinite(x) and 0 <= x < upper for x in row)
        for row in payload[field]
    )
    payload.update(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    return payload


def angle(before, after):
    # Match the training logger's stable stored-weight angle exactly.
    a, b = before.detach().float(), after.detach().float()
    aa = a.square().sum(dtype=torch.float64)
    dot = (a * b).sum(dtype=torch.float64)
    if aa <= 0 or b.square().sum(dtype=torch.float64) <= 0:
        raise ValueError("zero-norm weight in angle measurement")
    tangent = b - (dot / aa).to(a.dtype) * a
    return torch.atan2(
        tangent.square().sum(dtype=torch.float64).sqrt(), dot / aa.sqrt()
    ).item()


def apply_update(p, u, lr, hyperball, wd):
    if hyperball:
        wnorm, unorm = p.norm(), u.norm()
        candidate = p - lr * u * wnorm / unorm.clamp_min(1e-10)
        p.copy_(candidate / candidate.norm().clamp_min(1e-10) * wnorm)
    else:
        p.mul_(1 - lr * wd)
        p.add_(u, alpha=-lr)


def refine(p, u, target, initial, hyperball, wd, iterations):
    if target == 0:
        return 0.0, p.clone(), 0.0, 0.0, 0.0
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError(f"invalid analytical LR: {initial}")
    best = None

    def evaluate(lr):
        nonlocal best
        candidate = p.clone()
        apply_update(candidate, u, lr, hyperball, wd)
        value = angle(p, candidate)
        if not math.isfinite(value):
            raise ValueError("nonfinite stored-weight angle")
        error = abs(value - target)
        if best is None or error < best[0]:
            best = (error, lr, candidate, value)
        return value

    evaluate(initial)
    lo, hi = initial * 0.95, initial * 1.05
    alo, ahi = evaluate(lo), evaluate(hi)
    for _ in range(12):
        if alo <= target <= ahi:
            break
        if target < alo:
            hi, ahi = lo, alo
            lo *= 0.5
            alo = evaluate(lo)
        else:
            lo, alo = hi, ahi
            hi *= 2
            ahi = evaluate(hi)
    else:
        raise ValueError(f"cannot bracket angle {target}: {alo}, {ahi}")
    for _ in range(iterations):
        mid = (lo + hi) / 2
        amid = evaluate(mid)
        if amid < target:
            lo, alo = mid, amid
        else:
            hi, ahi = mid, amid
    if best[0] > max(2e-7, 2e-4 * target):
        raise ValueError(
            f"stored-weight angle error exceeds gate: {best[0]} vs {target}"
        )
    return best[1], best[2], best[3], alo, ahi


class MatrixAdam(torch.optim.Optimizer):
    state_sharded = True

    def __init__(
        self,
        named_params,
        mode,
        lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
        eps=1e-8,
        target_payload=None,
        refinement_iterations=20,
    ):
        named_params = sorted(
            list(named_params), key=lambda x: x[1].size(), reverse=True
        )
        self.kind = "norm" if mode.startswith("norm_") else "angle"
        self.reference = mode.startswith("reference_")
        family = (
            mode.removeprefix("norm_").removeprefix("reference_").split("_matches_")[0]
        )
        if family not in ("adamw", "adamh"):
            raise ValueError(mode)
        self.hyperball = family == "adamh"
        self.mode = mode
        self.refinement_iterations = refinement_iterations
        self.names = {id(p): n for n, p in named_params}
        self.target_step = None
        self.target_payload = target_payload
        if not self.reference:
            if target_payload is None or set(target_payload["parameter_names"]) != set(
                self.names.values()
            ):
                raise ValueError("target matrix names do not match")
            self.columns = {
                n: i for i, n in enumerate(target_payload["parameter_names"])
            }
        super().__init__(
            [p for _, p in named_params],
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=0.0 if self.hyperball else weight_decay,
            ),
        )

    def set_target_step(self, step):
        field = "target_norm_eff" if self.kind == "norm" else "target_radians"
        if not 0 <= step < len(self.target_payload[field]):
            raise IndexError(step)
        self.target_step = step

    @torch.no_grad()
    def step(self):
        if not self.reference and self.target_step is None:
            raise ValueError("set_target_step is required")
        distributed = dist.is_available() and dist.is_initialized()
        rank, world = (
            (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
        )
        for group in self.param_groups:
            params = group["params"]
            beta1, beta2 = group["betas"]
            for index in range(0, len(params), world):
                if index + rank < len(params):
                    p = params[index + rank]
                    if p.dtype != torch.float32:
                        raise ValueError("matrix weights and moments must be FP32")
                    state = self.state[p]
                    if "step" not in state:
                        state.update(
                            step=0,
                            exp_avg=torch.zeros_like(p),
                            exp_avg_sq=torch.zeros_like(p),
                        )
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError(
                            f"missing/nonfinite gradient: {self.names[id(p)]}"
                        )
                    state["step"] += 1
                    m, v, t = state["exp_avg"], state["exp_avg_sq"], state["step"]
                    m.lerp_(p.grad, 1 - beta1)
                    v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    u = (m / (1 - beta1**t)) / (
                        (v / (1 - beta2**t)).sqrt() + group["eps"]
                    )
                    state["_tb_pre_projection_u"] = u.detach()
                    wn = p.square().sum(dtype=torch.float64).sqrt().item()
                    un = u.square().sum(dtype=torch.float64).sqrt().item()
                    inner = (p * u).sum(dtype=torch.float64).item()
                    if (
                        not all(math.isfinite(x) for x in (wn, un, inner))
                        or wn <= 0
                        or un <= 0
                    ):
                        raise ValueError(f"invalid live geometry: {self.names[id(p)]}")
                    cosine = max(-1.0, min(1.0, inner / (wn * un)))
                    lr = float(group["lr"])
                    target = 0.0
                    if not self.reference:
                        field = (
                            "target_norm_eff"
                            if self.kind == "norm"
                            else "target_radians"
                        )
                        target = self.target_payload[field][self.target_step][
                            self.columns[self.names[id(p)]]
                        ]
                        if self.kind == "norm":
                            lr = target * (un if self.hyperball else wn)
                        else:
                            solved = (
                                solve_muonh_lr(target, cosine)
                                if self.hyperball
                                else solve_muonwd_lr(
                                    target, wn, un, inner, group["weight_decay"]
                                )
                            )
                            analytical = solved.applied_lr
                            lr, candidate, actual, lo, hi = refine(
                                p,
                                u,
                                target,
                                analytical,
                                self.hyperball,
                                group["weight_decay"],
                                self.refinement_iterations,
                            )
                            state["_tb_alignment_meta"] = (
                                target,
                                solved.tangent,
                                solved.numerator,
                                solved.denominator,
                                analytical,
                                lr,
                                solved.radial_component,
                                wn,
                                un,
                                inner,
                                cosine,
                                group["weight_decay"],
                                actual,
                                abs(actual - target),
                                self.refinement_iterations,
                                lo,
                                hi,
                                hi - lo,
                            )
                    if not math.isfinite(lr) or lr < 0:
                        raise ValueError(f"invalid matrix learning rate: {lr}")
                    if not self.reference and self.kind == "angle":
                        p.copy_(candidate)
                        realized = actual
                    else:
                        apply_update(p, u, lr, self.hyperball, group["weight_decay"])
                        realized = lr / (un if self.hyperball else wn)
                    if not torch.isfinite(p).all() or p.norm() <= 0:
                        raise ValueError(
                            f"nonfinite/zero updated matrix: {self.names[id(p)]}"
                        )
                    if self.reference:
                        target = realized
                    state["last_applied_lr"] = lr
                    state["_tb_adam_meta"] = (
                        lr,
                        target,
                        realized,
                        abs(realized - target),
                        wn,
                        un,
                    )
                if distributed:
                    # All ranks store identical parameters; only the owner stores Adam moments.
                    block = params[index : index + world]
                    padded = block + [
                        torch.empty_like(params[-1]) for _ in range(world - len(block))
                    ]
                    # Matrices in each block have identical shapes in the 72-matrix model.
                    if len({tuple(x.shape) for x in padded}) == 1:
                        dist.all_gather(padded, padded[rank])
                    else:
                        for owner, param in enumerate(block):
                            dist.broadcast(param, owner)
        self.target_step = None
