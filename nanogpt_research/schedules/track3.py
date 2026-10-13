"""Track-3 schedules with explicit endpoints (steps are zero-based)."""


def power_decay_lr(
    step,
    peak_lr,
    *,
    floor_lr=1e-5,
    warmup_steps=100,
    hold_until=199,
    decay_until=3200,
    power=0.4
):
    """Warm up including both endpoints, hold, then apply a power decay."""
    if (
        step < 0
        or warmup_steps < 2
        or hold_until < warmup_steps - 1
        or decay_until <= hold_until
    ):
        raise ValueError("invalid schedule endpoints")
    if not 0 <= floor_lr <= peak_lr or power <= 0:
        raise ValueError("invalid learning rates or power")
    if step < warmup_steps:
        return floor_lr + (peak_lr - floor_lr) * step / (warmup_steps - 1)
    if step <= hold_until:
        return peak_lr
    if step <= decay_until:
        progress = (step - hold_until) / (decay_until - hold_until)
        return floor_lr + (peak_lr - floor_lr) * (1 - progress**power)
    return floor_lr


def cooldown_factor(step, train_steps, cooldown_fraction=0.85):
    if (
        not 0 <= step <= train_steps
        or train_steps <= 0
        or not 0 < cooldown_fraction <= 1
    ):
        raise ValueError("invalid cooldown parameters")
    progress = step / train_steps
    return (
        1.0 if progress < 1 - cooldown_fraction else (1 - progress) / cooldown_fraction
    )
