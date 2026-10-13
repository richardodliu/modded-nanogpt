"""Track-3 optimizer implementation: normuon_record10."""

from __future__ import annotations
import torch
import torch.distributed as dist
from nanogpt_research.checkpoint import rank_world
from .common import zeropower_via_newtonschulz5


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
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )


def normuon_update(grad, momentum, second_momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5

    norm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = (
        torch.mean(update * update, dim=-1, keepdim=True)
        if grad.size(-2) >= grad.size(-1)
        else torch.mean(update * update, dim=-2, keepdim=True)
    )
    second_momentum.lerp_(v_mean.float(), 1 - 0.95)
    step_size = second_momentum.clamp_min(1e-10).rsqrt().bfloat16()

    update.mul_(step_size)
    norm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(norm / (norm_new.clamp_min(1e-10)))

    return update


class NorMuon(torch.optim.Optimizer):
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
                        state["second_momentum"] = (
                            torch.zeros_like(p.data[..., 0:1])
                            if p.size(-2) >= p.size(-1)
                            else torch.zeros_like(p.data[0:1, ...])
                        )

                    update = normuon_update(
                        p.grad,
                        state["momentum"],
                        state["second_momentum"],
                        mu=group["mu"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )
