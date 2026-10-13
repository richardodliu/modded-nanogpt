"""Track-3 optimizer implementation: controlled_normuonh_riemannian_tangent."""

from __future__ import annotations
import torch
import torch.distributed as dist
from nanogpt_research.checkpoint import rank_world
from .common import zeropower_via_newtonschulz5
from .common import scale_invariant_update_


def project_tangent(update, param, eps=1e-10):
    p_norm_sq = torch.clamp(param.float().square().sum(), min=eps)
    radial = (update.float() * param.float()).sum() / p_norm_sq
    return update - param * radial.to(dtype=param.dtype)


def normuon_update(
    grad, param, momentum, v_buf, mu=0.95, beta2=0.95, eps=1e-10, nesterov=True
):
    """NorMuon direction: NS-orthogonalised gradient followed by Adafactor-style variance
    preconditioning along the SHORT axis (per https://arxiv.org/pdf/2510.05491). The
    variance buffer `v_buf` is an EMA of squared post-NS values along the short axis,
    persistent across optimizer steps."""
    # Riemannian variant: remove radial components before momentum/NS so the nonlinear
    # preconditioner does not mix off-manifold gradient mass back into the tangent direction.
    grad_tangent = project_tangent(grad, param, eps)
    momentum.lerp_(grad_tangent, 1 - mu)
    momentum.copy_(project_tangent(momentum, param, eps))
    update = grad_tangent.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    # Variance per row (if rows >= cols) or per column (otherwise) — Adafactor-style.
    if grad.size(-2) >= grad.size(-1):
        v_new = update.square().mean(dim=-1, keepdim=True)
    else:
        v_new = update.square().mean(dim=-2, keepdim=True)
    # `update` is bf16 after NS5; cast `v_new` to `v_buf`'s dtype (float32) so lerp_ matches.
    v_buf.lerp_(v_new.to(v_buf.dtype), 1 - beta2)
    update = update * v_buf.clamp_min(eps).rsqrt()
    return update


class NorMuonH(torch.optim.Optimizer):
    """NorMuonH: NS-orthogonalised gradient + Adafactor-style row/column variance
    preconditioning (NorMuon, https://arxiv.org/pdf/2510.05491) + hyperball Frobenius-
    norm-preserving step. Used here for ALL hidden 2D weight matrices — q, k, v, mlp.fc,
    attn.proj, mlp.proj — under non-zero (Kaiming-derived) init."""

    def __init__(self, params, lr=0.018, mu=0.95, beta2=0.95, eps=1e-10):
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, mu=mu, beta2=beta2, eps=eps)
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
                        # variance buffer shape: keepdim along the short axis
                        v_shape = list(p.shape)
                        if p.size(-2) >= p.size(-1):
                            v_shape[-1] = 1
                        else:
                            v_shape[-2] = 1
                        state["v"] = torch.zeros(
                            v_shape, dtype=p.dtype, device=p.device
                        )
                    update = normuon_update(
                        p.grad,
                        p,
                        state["momentum"],
                        state["v"],
                        mu=group["mu"],
                        beta2=group["beta2"],
                        eps=group["eps"],
                    )
                    scale_invariant_update_(p, update, group["lr"])
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )
