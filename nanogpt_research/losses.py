"""Summed token loss; optional diagnostics never change its gradients."""

import torch
import torch.nn.functional as F


def cross_entropy_with_centered_logit_regularization(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_reg_weight: float,
    ignore_index: int = -1,
    *,
    measure_logit_variance: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed token loss plus detached CE/regularizer/count statistics."""
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    valid_mask = flat_targets.ne(ignore_index)

    ce_loss_sum = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=ignore_index,
        reduction="sum",
    )
    if not measure_logit_variance and float(logit_reg_weight) == 0.0:
        # The centered-logit statistic was only used for diagnostics.  Omitting
        # its full-vocabulary variance is a material speedup for figure-only runs.
        valid_count = valid_mask.sum().to(dtype=torch.float32)
        metrics = torch.stack(
            (
                ce_loss_sum.detach().float(),
                torch.zeros_like(ce_loss_sum.detach().float()),
                valid_count.detach(),
            )
        )
        return ce_loss_sum, metrics
    # correction=0 is the vocabulary mean squared centered logit. For the standard
    # CE arm this is detached diagnostics only, so its backward graph is exactly CE.
    if float(logit_reg_weight) == 0.0:
        with torch.no_grad():
            centered_mse_per_token = torch.var(
                flat_logits.float(), dim=-1, correction=0
            )
            centered_logit_sum = centered_mse_per_token[valid_mask].sum()
    else:
        centered_mse_per_token = torch.var(flat_logits.float(), dim=-1, correction=0)
        centered_logit_sum = centered_mse_per_token[valid_mask].sum()
    valid_count = valid_mask.sum().to(dtype=torch.float32)
    total_loss_sum = (
        ce_loss_sum
        if float(logit_reg_weight) == 0.0
        else ce_loss_sum + float(logit_reg_weight) * centered_logit_sum
    )
    metrics = torch.stack(
        (
            ce_loss_sum.detach().float(),
            centered_logit_sum.detach().float(),
            valid_count.detach(),
        )
    )
    return total_loss_sum, metrics
