"""Track-3 optimizer implementation: muonh_record5."""

from __future__ import annotations
import torch
from torch import Tensor
import torch.distributed as dist
from nanogpt_research.checkpoint import rank_world
from .common import zeropower_via_newtonschulz5


def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


@torch.no_grad()
def scale_invariant_update_(
    param: Tensor, update: Tensor, lr: float, eps: float = 1e-10
) -> None:
    """Hyperball-constrained step: take a Muon-orthogonalised update of size lr * ||param||,
    then renormalise back onto the Frobenius sphere of the parameter's initial radius. Preserves
    ||param|| exactly across training; the invariant lets us drop weight decay on hidden
    matrices entirely (the constraint already prevents norm growth)."""
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)


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
                    scale_invariant_update_(p, update, group["lr"])
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )
