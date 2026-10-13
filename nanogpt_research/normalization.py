"""Derf and Internal-Taper with explicit, checkpointed calibration.

Derf: official affine module defaults, commit recorded in plan.json.
Taper: paper Eq3/Eq4; fixed-g continuation is our ablation, not cosine taper.
All arithmetic inside normalization is FP32; output follows input dtype.
"""

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

EPS = torch.finfo(torch.float32).eps
GATES = {
    "taper_g1": 1.0,
    "taper_g0p75": 0.75,
    "taper_g0p5": 0.5,
    "taper_g0p25": 0.25,
    "taper_g0": 0.0,
}


def derf_reference(x, alpha, shift, gains, bias):
    return (gains * torch.erf(alpha * x.float() + shift) + bias).to(x.dtype)


def taper_reference(x, gains, linear_gains, c, gate):
    f = x.float()
    # Zero terms deliberately keep both parameter gradients defined.
    if gate == 0.0:
        return (f * (c * linear_gains) + gains * 0.0).to(x.dtype)
    z = f * (f.square().mean(-1, keepdim=True) + EPS).rsqrt()
    if gate == 1.0:
        return (z * gains + linear_gains * 0.0).to(x.dtype)
    return (gate * z * gains + (1.0 - gate) * c * f * linear_gains).to(x.dtype)


class Derf(nn.Module):
    leaves = ("alpha", "shift", "gains", "bias")
    gain_leaves = ("gains",)
    kind = "derf"

    def __init__(self, dim, *, device=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.5, device=device))
        self.shift = nn.Parameter(torch.zeros(1, device=device))
        self.gains = nn.Parameter(torch.ones(dim, device=device))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
        self.backend = "reference"

    def forward(self, x):
        if self.backend != "reference":
            from .normalization_kernels import derf_fused

            return derf_fused(x, self.alpha, self.shift, self.gains, self.bias)
        return derf_reference(x, self.alpha, self.shift, self.gains, self.bias)


class TaperNorm(nn.Module):
    leaves = ("gains", "linear_gains")
    gain_leaves = leaves
    kind = "taper"

    def __init__(self, dim, *, device=None, final=False):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim, device=device))
        self.linear_gains = nn.Parameter(torch.ones(dim, device=device))
        for name, value in [
            ("c", 1.0),
            ("ema_num", 0.0),
            ("ema_den", 0.0),
            ("last_num", 0.0),
            ("last_den", 0.0),
        ]:
            self.register_buffer(name, torch.tensor(value, device=device))
        self.register_buffer(
            "calibration_steps", torch.tensor(0, device=device, dtype=torch.int64)
        )
        self.register_buffer("calibrated", torch.tensor(False, device=device))
        self.final = final
        self.gate = 1.0
        self.collect = False
        self.backend = "reference"

    def forward(self, x):
        if self.collect and not self.final:
            with torch.no_grad():
                f = x.float()
                weighted = (f * self.gains).square().sum(-1)
                inv = (f.square().mean(-1) + EPS).rsqrt()
                self.last_num.copy_((weighted * inv).mean())
                self.last_den.copy_(weighted.mean())
        if self.backend != "reference":
            from .normalization_kernels import taper_fused

            return taper_fused(x, self.gains, self.linear_gains, self.c, self.gate)
        return taper_reference(x, self.gains, self.linear_gains, self.c, self.gate)


def sites(model):
    return [
        (n, m) for n, m in model.named_modules() if isinstance(m, (Derf, TaperNorm))
    ]


def make_norm(dim, device, final=False, case="derf"):
    return (
        Derf(dim, device=device)
        if case == "derf"
        else TaperNorm(dim, device=device, final=final)
    )


class NormAttention(nn.Module):
    def __init__(self, old, case="derf"):
        super().__init__()
        self.num_heads, self.head_dim = old.num_heads, old.head_dim
        self.q, self.k, self.v = old.q, old.k, old.v
        self.proj, self.rotary = old.proj, old.rotary
        self.q_norm = make_norm(self.head_dim, old.q.weight.device, case=case)
        self.k_norm = make_norm(self.head_dim, old.k.weight.device, case=case)

    def forward(self, x):
        b, t = x.shape[:2]
        q = self.q(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k(x).view(b, t, self.num_heads, self.head_dim)
        v = self.v(x).view(b, t, self.num_heads, self.head_dim)
        q, k = self.rotary(self.q_norm(q)), self.rotary(self.k_norm(k))
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=0.12,
            is_causal=True,
        ).transpose(1, 2)
        return self.proj(y.contiguous().view(b, t, self.num_heads * self.head_dim))


@torch.no_grad()
def replace_normalization(model, case="derf"):
    if case != "derf" and case not in GATES:
        raise ValueError(case)
    for name, old in list(model.named_modules()):
        if old.__class__.__name__ != "RMSNorm":
            continue
        new = make_norm(
            old.gains.numel(), old.gains.device, final=name == "norm2", case=case
        )
        new.gains.copy_(old.gains)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, new)
    for block in model.blocks:
        block.attn = NormAttention(block.attn, case)
    assert len(sites(model)) == 4 * len(model.blocks) + 2
    return model


@torch.no_grad()
def prepare_step(model, update, warmup=40, case="taper_g0p5"):
    """Freeze calibration and set g BEFORE measuring optimizer-induced changes."""
    for _, m in sites(model):
        if not isinstance(m, TaperNorm):
            continue
        if update > warmup and not m.final and not bool(m.calibrated.item()):
            assert m.calibration_steps.item() == warmup
            assert torch.isfinite(m.ema_num) and m.ema_num > 0 and m.ema_den > 0
            # Bias corrections cancel in this ratio.
            m.c.copy_(m.ema_num / m.ema_den.clamp_min(1e-30))
            m.linear_gains.copy_(m.gains)
            m.calibrated.fill_(True)
        m.gate = 1.0 if m.final or update <= warmup else GATES[case]
        m.collect = update <= warmup and not m.final


@torch.no_grad()
def finish_step(model):
    pending = [m for _, m in sites(model) if isinstance(m, TaperNorm) and m.collect]
    if pending:
        values = torch.stack([torch.stack((m.last_num, m.last_den)) for m in pending])
        if dist.is_initialized():
            dist.all_reduce(values)
            values.div_(dist.get_world_size())
        for m, v in zip(pending, values):
            m.ema_num.mul_(0.99).add_(v[0], alpha=0.01)
            m.ema_den.mul_(0.99).add_(v[1], alpha=0.01)
            m.calibration_steps.add_(1)
            m.collect = False


@torch.no_grad()
def set_backend(model, backend):
    for _, m in sites(model):
        m.backend = backend
