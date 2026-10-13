"""SGD variants with an exact measured-angle alignment mode and zero WD."""

import math
import torch
import torch.distributed as dist
from nanogpt_research.geometry.sgd_angular import solve_update


def world_rank():
    return (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)


@torch.no_grad()
def hyperball_update_(p, u, lr):
    if lr == 0:
        return
    den = u.norm()
    if den.item() == 0:
        return
    radius = p.norm()
    candidate = p - lr * u * radius / den.clamp_min(1e-10)
    norm = candidate.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise FloatingPointError("Invalid H retraction")
    p.copy_(candidate / norm * radius)


class NormHSGD(torch.optim.Optimizer):
    state_sharded = True

    def __init__(
        self,
        named,
        family="sgd",
        *,
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        dampening=0.0,
        weight_decay=0.0,
        aligned=False,
        alignment_mode="angle"
    ):
        if family not in ("sgd", "sgdh") or alignment_mode not in ("h", "elr", "angle"):
            raise ValueError((family, alignment_mode))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum and zero dampening")
        if lr < 0 or momentum < 0 or dampening < 0 or weight_decay < 0:
            raise ValueError("optimizer hyperparameters must be nonnegative")
        named = sorted(named, key=lambda x: x[1].size(), reverse=True)
        super().__init__(
            [p for _, p in named],
            dict(
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov,
                weight_decay=weight_decay if family == "sgd" else 0.0,
            ),
        )
        self.names = {id(p): n for n, p in named}
        world, rank = world_rank()
        self.owned_param_ids = {
            id(p) for i, (_, p) in enumerate(named) if i % world == rank
        }
        self.family = family
        self.aligned = aligned
        self.alignment_mode = alignment_mode
        self.targets = None
        self.last_rows = []
        self.raw_gradient = momentum == 0

    @torch.no_grad()
    def step(self):
        self.last_rows = []
        world, rank = world_rank()
        if self.aligned:
            if self.targets is None or set(self.targets) != set(self.names.values()):
                raise ValueError("Missing matrix targets")
            if any(not math.isfinite(v) or v < 0 for v in self.targets.values()):
                raise ValueError("Invalid alignment target")
        for group in self.param_groups:
            momentum, dampening, nesterov = (
                group["momentum"],
                group["dampening"],
                group["nesterov"],
            )
            wd = group["weight_decay"]
            params = group["params"]
            pad = params + [torch.empty_like(params[-1]) for _ in range(world)]
            for base in range(0, len(params), world):
                if base + rank < len(params):
                    p = params[base + rank]
                    name = self.names[id(p)]
                    if p.grad is None:
                        raise ValueError("Missing gradient: " + name)
                    g = p.grad.detach()
                    r = p.norm().item()
                    gn = g.norm().item()
                    if not math.isfinite(r) or not math.isfinite(gn) or r <= 0:
                        raise ValueError("Invalid geometry: " + name)
                    state = self.state[p]
                    if momentum:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = g.clone()
                        else:
                            state["momentum_buffer"].mul_(momentum).add_(
                                g, alpha=1 - dampening
                            )
                        m = state["momentum_buffer"]
                        u = g.add(m, alpha=momentum) if nesterov else m
                        mn = m.norm().item()
                    else:
                        u = g
                        mn = 0.0
                    un = u.norm().item()
                    den = max(un, 1e-10)
                    if not math.isfinite(un):
                        raise ValueError("Nonfinite update direction")
                    state["_tb_pre_projection_u"] = u
                    target = float(self.targets[name]) if self.aligned else None
                    angle_row = {}
                    candidate = None
                    factor = r if self.alignment_mode == "h" else r * r
                    if self.aligned and self.alignment_mode == "angle":
                        lr, candidate, angle_row = solve_update(
                            p, u, target, self.family, wd
                        )
                        c = 1 - lr * wd
                        eta = lr if self.family == "sgd" else lr * r / den
                        h = eta / (c * r)
                        elr = h / r
                    elif self.family == "sgd":
                        # Solve t = eta / ((1-eta*wd)*factor), not a Taylor approximation.
                        lr = (
                            target * factor / (1 + wd * target * factor)
                            if self.aligned
                            else group["lr"]
                        )
                        c = 1 - lr * wd
                        if not math.isfinite(c) or c <= 0:
                            raise ValueError("Nonpositive decay factor")
                        eta = lr
                        h = eta / (c * r)
                        elr = h / r
                    else:
                        lr = (
                            target * den * (r if self.alignment_mode == "elr" else 1)
                            if self.aligned
                            else group["lr"]
                        )
                        c = 1.0
                        h = lr / den
                        elr = h / r
                        eta = lr * r / den
                    if not all(math.isfinite(v) for v in (lr, h, elr, eta)):
                        raise ValueError("Nonfinite solved LR")
                    actual = h if self.alignment_mode == "h" else elr
                    if (
                        target is not None
                        and self.alignment_mode != "angle"
                        and abs(actual - target) > max(1e-15, 2e-12 * target)
                    ):
                        raise ValueError("Alignment mismatch")
                    if candidate is not None:
                        p.copy_(candidate)
                    elif self.family == "sgd":
                        p.mul_(c).add_(u, alpha=-lr)
                    else:
                        hyperball_update_(p, u, lr)
                    self.last_rows.append(
                        dict(
                            name=name,
                            h=h,
                            elr=elr,
                            target_alignment=target,
                            alignment_mode=self.alignment_mode,
                            applied_lr=lr,
                            additive_eta=eta,
                            weight_norm_before=r,
                            update_norm=un,
                            raw_gradient_norm=gn,
                            momentum_buffer_norm=mn,
                            momentum=momentum,
                            nesterov=nesterov,
                            dampening=dampening,
                            weight_decay=wd,
                            decay_factor=c,
                            sgd_scale_invariant_lr=elr,
                            normalized_step_magnitude=h * un,
                            **angle_row
                        )
                    )
                if world > 1:
                    dist.all_gather(pad[base : base + world], pad[base + rank])
        self.targets = None

    def gathered_rows(self):
        world, _ = world_rank()
        parts = [None] * world
        if world > 1:
            dist.all_gather_object(parts, self.last_rows)
        else:
            parts = [self.last_rows]
        rows = [r for part in parts for r in part]
        assert len(rows) == len(self.names) == len({r["name"] for r in rows})
        return {r["name"]: r for r in rows}
