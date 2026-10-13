"""Adam preconditioning with a Frobenius-norm-preserving update."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def scale_invariant_update_(
    parameter: Tensor,
    update: Tensor,
    learning_rate: float,
    eps: float = 1e-10,
) -> None:
    """Apply an update on the parameter's initial Frobenius hyperball."""
    parameter_norm = parameter.norm()
    update_norm = update.norm()
    candidate = parameter - learning_rate * update * parameter_norm / torch.clamp(
        update_norm, min=eps
    )
    candidate_norm = torch.clamp(candidate.norm(), min=eps)
    parameter.copy_(candidate / candidate_norm * parameter_norm)


class AdamH(torch.optim.Optimizer):
    """Accepted AdamH: Adam preconditioning plus hyperball projection."""

    def __init__(self, params, lr=0.018, betas=(0.9, 0.95), eps=1e-8):
        parameters = list(params)
        if not parameters:
            raise ValueError("AdamH requires at least one parameter")
        super().__init__(parameters, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(parameter.grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    parameter.grad, parameter.grad, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                update = (exp_avg / bias_correction1) / (
                    (exp_avg_sq / bias_correction2).sqrt() + eps
                )
                state["_tb_pre_projection_u"] = update.detach()
                scale_invariant_update_(parameter, update, group["lr"], eps)
        return loss


def set_initial_lrs(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = float(group["lr"])


def apply_lr_factor(optimizers: list[torch.optim.Optimizer], factor: float) -> None:
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"invalid LR factor {factor}")
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor


def optimizer_train(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        train = getattr(optimizer, "train", None)
        if train is not None:
            train()


def optimizer_eval(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        evaluate = getattr(optimizer, "eval", None)
        if evaluate is not None:
            evaluate()
