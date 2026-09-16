from __future__ import annotations

import torch
from torch import nn


class MockAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)


class MockMLP(nn.Module):
    def __init__(self, dim: int, ratio: int = 4) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, ratio * dim)
        self.lin2 = nn.Linear(ratio * dim, dim)


class MockBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = MockAttention(dim)
        self.mlp = MockMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Small differentiable proxy, not a full attention implementation.
        q, k, v = self.attn.qkv(x).chunk(3, dim=-1)
        x = x + self.attn.proj((q + k + v) / 3.0)
        x = x + self.mlp.lin2(torch.nn.functional.gelu(self.mlp.lin1(x)))
        return x


class MockImageEncoder(nn.Module):
    def __init__(self, dim: int = 16, depth: int = 12) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MockBlock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class MockSAM(nn.Module):
    def __init__(self, dim: int = 16, depth: int = 12) -> None:
        super().__init__()
        self.image_encoder = MockImageEncoder(dim, depth)
        self.classification_head = nn.Linear(dim, 9)
        self.quality_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        z = self.image_encoder(x)
        pooled = z.mean(dim=1)
        return self.classification_head(pooled), self.quality_head(pooled)
