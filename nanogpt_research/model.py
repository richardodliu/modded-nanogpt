"""Track-3 decoder. Default 12 x 768, untied embeddings: 162,354,816 parameters."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .losses import cross_entropy_with_centered_logit_regularization


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return (norm(x.float()) * self.gains).type_as(x)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_frequency = (1 / 1024) ** torch.linspace(
            0, 1, steps=dim // 4, dtype=torch.float32
        )
        self.register_buffer(
            "angular_freq",
            torch.cat([angular_frequency, angular_frequency.new_zeros(dim // 4)]),
        )

    def forward(self, x: Tensor) -> Tensor:
        position = torch.arange(x.size(1), dtype=torch.float32, device=x.device)
        theta = torch.outer(position, self.angular_freq)[None, :, None, :]
        cosine, sine = theta.cos(), theta.sin()
        x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
        return torch.cat(
            (x1 * cosine + x2 * sine, x1 * (-sine) + x2 * cosine), 3
        ).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hidden = self.num_heads * head_dim
        self.q = Linear(dim, hidden)
        self.k = Linear(dim, hidden)
        self.v = Linear(dim, hidden)
        self.proj = Linear(hidden, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, sequence = x.shape[:2]
        q = self.q(x).view(batch, sequence, self.num_heads, self.head_dim)
        k = self.k(x).view(batch, sequence, self.num_heads, self.head_dim)
        v = self.v(x).view(batch, sequence, self.num_heads, self.head_dim)
        q, k = self.rotary(norm(q)), self.rotary(norm(k))
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=0.12,
            is_causal=True,
        ).transpose(1, 2)
        return self.proj(output.contiguous().view(batch, sequence, -1))


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = Linear(dim, 4 * dim)
        self.proj = Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.fc(x).relu().square())


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50304,
        num_layers: int = 12,
        model_dim: int = 768,
        centered_logit_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.centered_logit_reg_weight = float(centered_logit_reg_weight)
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return cross_entropy_with_centered_logit_regularization(
            logits, targets, self.centered_logit_reg_weight
        )
