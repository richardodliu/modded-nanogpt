"""Deterministic fused kernels. Promotion is gated by GPU measurements.

Exact libdevice erf; FP32 intermediate arithmetic and fixed two-stage
parameter-gradient reductions. No approximate erf and no atomic gradients.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.library import triton_op, wrap_triton
from .normalization import EPS, derf_reference, taper_reference


@triton.jit
def _derf_f(X, A, S, G, B, Y, N: tl.constexpr, D: tl.constexpr, K: tl.constexpr):
    i = tl.program_id(0) * K + tl.arange(0, K)
    x = tl.load(X + i, i < N * D, other=0).to(tl.float32)
    a = tl.load(A)
    s = tl.load(S)
    g = tl.load(G + i % D)
    b = tl.load(B + i % D)
    tl.store(Y + i, g * libdevice.erf(a * x + s) + b, i < N * D)


@triton.jit
def _derf_b(
    X,
    A,
    S,
    G,
    DY,
    DX,
    PG,
    PB,
    PA,
    PS,
    N: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    p = tl.program_id(0)
    r = p * R + tl.arange(0, R)
    c = tl.arange(0, C)
    mask = (r[:, None] < N) & (c[None, :] < D)
    i = r[:, None] * D + c[None, :]
    x = tl.load(X + i, mask, other=0).to(tl.float32)
    dy = tl.load(DY + i, mask, other=0).to(tl.float32)
    a = tl.load(A)
    s = tl.load(S)
    g = tl.load(G + c, c < D, other=0)
    z = a * x + s
    t = libdevice.erf(z)
    dz = (dy * g[None, :]) * (1.1283791670955126 * libdevice.exp(-z * z))
    tl.store(DX + i, dz * a, mask)
    tl.store(PG + p * D + c, tl.sum(dy * t, 0), c < D)
    tl.store(PB + p * D + c, tl.sum(dy, 0), c < D)
    tl.store(PA + p, tl.sum(tl.sum(dz * x, 0), 0))
    tl.store(PS + p, tl.sum(tl.sum(dz, 0), 0))


@triton_op("norm20261003::derf_forward", mutates_args={})
def derf_op(
    x: torch.Tensor, a: torch.Tensor, s: torch.Tensor, g: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    n, d = x.numel() // x.shape[-1], x.shape[-1]
    y = torch.empty_like(x)
    wrap_triton(_derf_f)[(triton.cdiv(x.numel(), 1024),)](
        x, a, s, g, b, y, n, d, 1024, num_warps=4, enable_fp_fusion=False
    )
    return y


@triton_op("norm20261003::derf_backward", mutates_args={})
def derf_back(
    x: torch.Tensor, a: torch.Tensor, s: torch.Tensor, g: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, d = x.numel() // x.shape[-1], x.shape[-1]
    r = 16 if d >= 512 else 32
    p = triton.cdiv(n, r)
    dx = torch.empty_like(x)
    pg = torch.empty((p, d), device=x.device)
    pb = torch.empty_like(pg)
    pa = torch.empty((p,), device=x.device)
    ps = torch.empty_like(pa)
    wrap_triton(_derf_b)[(p,)](
        x,
        a,
        s,
        g,
        dy.contiguous(),
        dx,
        pg,
        pb,
        pa,
        ps,
        n,
        d,
        r,
        triton.next_power_of_2(d),
        num_warps=8 if d >= 512 else 4,
        enable_fp_fusion=False,
    )
    return dx, pa.sum().reshape_as(a), ps.sum().reshape_as(s), pg.sum(0), pb.sum(0)


def _derf_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:4])


def _derf_grad(ctx, dy):
    return derf_back(*ctx.saved_tensors, dy)


derf_op.register_autograd(_derf_grad, setup_context=_derf_setup)


@triton.jit
def _taper_f(
    X,
    G,
    H,
    CV,
    Y,
    N: tl.constexpr,
    D: tl.constexpr,
    GATE: tl.constexpr,
    EPSILON: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    r = tl.program_id(0) * R + tl.arange(0, R)
    c = tl.arange(0, C)
    i = r[:, None] * D + c[None, :]
    mask = (r[:, None] < N) & (c[None, :] < D)
    x = tl.load(X + i, mask, other=0).to(tl.float32)
    g = tl.load(G + c, c < D, other=0)
    h = tl.load(H + c, c < D, other=0)
    cv = tl.load(CV)
    if GATE == 0.0:
        y = x * (cv * h[None, :])
    else:
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPSILON)
        z = x * inv[:, None]
        if GATE == 1.0:
            y = z * g[None, :]
        else:
            y = GATE * z * g[None, :] + (1.0 - GATE) * cv * x * h[None, :]
    tl.store(Y + i, y, mask)


@triton.jit
def _taper_b(
    X,
    G,
    H,
    CV,
    DY,
    DX,
    PG,
    PH,
    N: tl.constexpr,
    D: tl.constexpr,
    GATE: tl.constexpr,
    EPSILON: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    p = tl.program_id(0)
    r = p * R + tl.arange(0, R)
    c = tl.arange(0, C)
    i = r[:, None] * D + c[None, :]
    mask = (r[:, None] < N) & (c[None, :] < D)
    x = tl.load(X + i, mask, other=0).to(tl.float32)
    dy = tl.load(DY + i, mask, other=0).to(tl.float32)
    g = tl.load(G + c, c < D, other=0)
    h = tl.load(H + c, c < D, other=0)
    cv = tl.load(CV)
    if GATE == 0.0:
        dx = dy * (cv * h[None, :])
        pg = tl.full((C,), 0, tl.float32)
    else:
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPSILON)
        v = dy * g[None, :]
        inner = tl.sum(v * x, 1) / D
        dx = GATE * (v * inv[:, None] - x * (inner * inv * inv * inv)[:, None])
        pg = tl.sum(dy * (x * inv[:, None]) * GATE, 0)
        if GATE != 1.0:
            dx = dx + (1.0 - GATE) * cv * dy * h[None, :]
    ph = tl.sum((1.0 - GATE) * cv * dy * x, 0)
    tl.store(DX + i, dx, mask)
    tl.store(PG + p * D + c, pg, c < D)
    tl.store(PH + p * D + c, ph, c < D)


@triton_op("norm20261003::taper_forward", mutates_args={})
def taper_op(
    x: torch.Tensor, g: torch.Tensor, h: torch.Tensor, c: torch.Tensor, gate: float
) -> torch.Tensor:
    n, d = x.numel() // x.shape[-1], x.shape[-1]
    y = torch.empty_like(x)
    r = 4
    wrap_triton(_taper_f)[(triton.cdiv(n, r),)](
        x,
        g,
        h,
        c,
        y,
        n,
        d,
        gate,
        EPS,
        r,
        triton.next_power_of_2(d),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y


@triton_op("norm20261003::taper_backward", mutates_args={})
def taper_back(
    x: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    dy: torch.Tensor,
    gate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, d = x.numel() // x.shape[-1], x.shape[-1]
    r = 16 if d >= 512 else 32
    p = triton.cdiv(n, r)
    dx = torch.empty_like(x)
    pg = torch.empty((p, d), device=x.device)
    ph = torch.empty_like(pg)
    wrap_triton(_taper_b)[(p,)](
        x,
        g,
        h,
        c,
        dy.contiguous(),
        dx,
        pg,
        ph,
        n,
        d,
        gate,
        EPS,
        r,
        triton.next_power_of_2(d),
        num_warps=8 if d >= 512 else 4,
        enable_fp_fusion=False,
    )
    return dx, pg.sum(0), ph.sum(0)


def _taper_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:4])
    ctx.gate = inputs[4]


def _taper_grad(ctx, dy):
    dx, dg, dh = taper_back(*ctx.saved_tensors, dy, ctx.gate)
    return dx, dg, dh, None, None


taper_op.register_autograd(_taper_grad, setup_context=_taper_setup)


def derf_fused(x, a, s, g, b):
    if not x.is_cuda or not x.is_contiguous():
        return derf_reference(x, a, s, g, b)
    return derf_op(x, a, s, g, b)


def taper_fused(x, g, h, c, gate):
    if not x.is_cuda or not x.is_contiguous():
        return taper_reference(x, g, h, c, gate)
    return taper_op(x, g, h, c, float(gate))
