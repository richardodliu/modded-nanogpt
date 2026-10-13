"""Causal stable-then-decay controller reused for the 10B AdamH arm.

This is the accepted gradient-norm detector from
``muonh_gradnorm_triggered_linear_20260804`` with experiment-local names.  It
only uses gradients observed at or before the current update.  Once stability
is persistent, the LR decays linearly to zero at the common 10B horizon.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any


@dataclass(frozen=True)
class StabilityConfig:
    horizon: int
    window: int = 1000
    check_interval: int = 100
    min_step: int = 2000
    mean_shift_threshold: float = 0.01
    slope_span_threshold: float = 0.01
    persistence_checks: int = 5
    force_switch_step: int | None = None

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least two updates")
        if self.window < 2 or self.check_interval < 1:
            raise ValueError("window and check_interval must be positive")
        if self.min_step < 2 * self.window:
            raise ValueError("min_step must cover two adjacent windows")
        if self.mean_shift_threshold <= 0 or self.slope_span_threshold <= 0:
            raise ValueError("stability thresholds must be positive")
        if self.persistence_checks < 1:
            raise ValueError("persistence_checks must be positive")
        if self.force_switch_step is not None and not (
            self.min_step <= self.force_switch_step < self.horizon - 1
        ):
            raise ValueError("force_switch_step must be inside the training horizon")


class GradientStabilityController:
    """Detect a flat log-gradient-norm regime without looking into the future."""

    def __init__(self, config: StabilityConfig):
        self.config = config
        self._log_norms: deque[float] = deque(maxlen=2 * config.window)
        self.stable_streak = 0
        self.switch_step: int | None = None
        self.forced_switch = False
        self.last_step = -1
        self.last_diagnostics: dict[str, float | int | bool] = {
            "checked": False,
            "mean_shift": float("nan"),
            "slope_span": float("nan"),
            "recent_log_mad": float("nan"),
            "stable_now": False,
            "stable_streak": 0,
        }

    @staticmethod
    def _slope_span(values: list[float]) -> float:
        count = len(values)
        center = 0.5 * (count - 1)
        denominator = sum((index - center) ** 2 for index in range(count))
        mean = sum(values) / count
        slope = (
            sum((index - center) * (value - mean) for index, value in enumerate(values))
            / denominator
        )
        return abs(slope) * (count - 1)

    def observe(
        self, update_step: int, grad_norm: float
    ) -> dict[str, float | int | bool]:
        if update_step != self.last_step + 1:
            raise ValueError(
                "gradient observations must be dense and ordered: "
                f"last={self.last_step}, new={update_step}"
            )
        if not math.isfinite(grad_norm) or grad_norm < 0:
            raise ValueError(f"invalid gradient norm: {grad_norm}")
        self.last_step = update_step
        self._log_norms.append(math.log(max(grad_norm, 1e-300)))

        checked = (
            update_step + 1 >= self.config.min_step
            and (update_step + 1) % self.config.check_interval == 0
            and len(self._log_norms) == 2 * self.config.window
        )
        diagnostics: dict[str, float | int | bool] = {
            "checked": checked,
            "mean_shift": float("nan"),
            "slope_span": float("nan"),
            "recent_log_mad": float("nan"),
            "stable_now": False,
            "stable_streak": self.stable_streak,
        }
        if checked and self.switch_step is None:
            values = list(self._log_norms)
            previous = values[: self.config.window]
            recent = values[self.config.window :]
            previous_median = statistics.median(previous)
            recent_median = statistics.median(recent)
            mean_shift = abs(math.expm1(recent_median - previous_median))
            slope_span = self._slope_span(recent)
            recent_mad = 1.4826 * statistics.median(
                abs(value - recent_median) for value in recent
            )
            stable_now = (
                mean_shift <= self.config.mean_shift_threshold
                and slope_span <= self.config.slope_span_threshold
            )
            self.stable_streak = self.stable_streak + 1 if stable_now else 0
            diagnostics.update(
                mean_shift=mean_shift,
                slope_span=slope_span,
                recent_log_mad=recent_mad,
                stable_now=stable_now,
                stable_streak=self.stable_streak,
            )
            if self.stable_streak >= self.config.persistence_checks:
                self.switch_step = update_step

        if (
            self.switch_step is None
            and self.config.force_switch_step is not None
            and update_step >= self.config.force_switch_step
        ):
            self.switch_step = update_step
            self.forced_switch = True

        self.last_diagnostics = diagnostics
        return dict(diagnostics)

    def lr_factor(self, update_step: int) -> float:
        if not 0 <= update_step < self.config.horizon:
            raise ValueError(
                f"step {update_step} is outside horizon {self.config.horizon}"
            )
        if self.switch_step is None or update_step <= self.switch_step:
            return 1.0
        denominator = (self.config.horizon - 1) - self.switch_step
        return max(0.0, 1.0 - (update_step - self.switch_step) / denominator)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "log_norms": list(self._log_norms),
            "stable_streak": self.stable_streak,
            "switch_step": self.switch_step,
            "forced_switch": self.forced_switch,
            "last_step": self.last_step,
            "last_diagnostics": self.last_diagnostics,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("unsupported controller checkpoint schema")
        if state.get("config") != asdict(self.config):
            raise ValueError("controller configuration mismatch")
        values = [float(value) for value in state["log_norms"]]
        if len(values) > 2 * self.config.window or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError("invalid checkpointed gradient history")
        self._log_norms.clear()
        self._log_norms.extend(values)
        self.stable_streak = int(state["stable_streak"])
        switch_step = state["switch_step"]
        self.switch_step = None if switch_step is None else int(switch_step)
        self.forced_switch = bool(state["forced_switch"])
        self.last_step = int(state["last_step"])
        self.last_diagnostics = dict(state["last_diagnostics"])


def wsd_factor(
    step: int, horizon: int, warmup_steps: int, decay_fraction: float
) -> float:
    """Linear warmup, constant plateau, then a linear decay to zero."""
    if not 0 <= step < horizon:
        raise ValueError(f"step {step} is outside horizon {horizon}")
    if not 0 <= warmup_steps < horizon:
        raise ValueError("warmup_steps must be inside the horizon")
    if not 0 < decay_fraction < 1:
        raise ValueError("decay_fraction must be between zero and one")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_start = max(warmup_steps, int(round(horizon * (1.0 - decay_fraction))))
    if step < decay_start:
        return 1.0
    denominator = max((horizon - 1) - decay_start, 1)
    return max(0.0, 1.0 - (step - decay_start) / denominator)


def linear_warmup_factor(step: int, warmup_steps: int) -> float:
    if step < 0 or warmup_steps < 0:
        raise ValueError("step and warmup_steps must be nonnegative")
    return min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0
