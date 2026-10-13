"""RMNP adapted from Dominator-Index/RMNP (Apache-2.0), commit a662ebc."""

import torch
import torch.distributed as dist
from nanogpt_research.checkpoint import rank_world
from .rmnp_math import rmnp_momentum_direction, row_normalize_native, rmnp_aspect_scale


class RMNP(torch.optim.Optimizer):
    """Row-normalized momentum with native output-row geometry."""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        weight_decay: float,
        eps: float,
        nesterov: bool = True,
    ):
        params = sorted(list(params), key=lambda value: value.size(), reverse=True)
        if not params:
            raise ValueError("RMNP requires at least one matrix parameter")
        super().__init__(
            params,
            dict(
                lr=float(lr),
                momentum=float(momentum),
                nesterov=bool(nesterov),
                weight_decay=float(weight_decay),
                eps=float(eps),
            ),
        )

    @torch.no_grad()
    def step(self):
        rank, world_size = rank_world()
        for group in self.param_groups:
            params = group["params"]
            padding = (world_size - len(params) % world_size) % world_size
            params_pad = params + [torch.empty_like(params[-1])] * padding
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    parameter = params[base_i + rank]
                    raw_grad = parameter.grad
                    if raw_grad is None:
                        raise RuntimeError("RMNP matrix gradient is missing")
                    if raw_grad.ndim != 2:
                        raise RuntimeError(
                            f"RMNP requires 2D gradients, got {raw_grad.shape}"
                        )
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    pre_row_normalization = rmnp_momentum_direction(
                        raw_grad,
                        state["momentum_buffer"],
                        group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    post_row_normalization = row_normalize_native(
                        pre_row_normalization,
                        group["eps"],
                    )
                    post_row_normalization *= rmnp_aspect_scale(raw_grad)

                    # Existing full diagnostics consume the exact pre-LR RMNP U.
                    state["_tb_pre_projection_u"] = post_row_normalization.detach()
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(
                        post_row_normalization,
                        alpha=-group["lr"],
                    )
                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )
