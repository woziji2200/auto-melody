"""Small causal Transformer block shared by chart-sequence models."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as torch_functional


class CausalTransformerBlock(nn.Module):
    """Pre-norm causal self-attention with an explicit one-step cache."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("model dimension must divide evenly across heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.attention_norm = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.attention_output = nn.Linear(dimension, dimension)
        self.feedforward_norm = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(dropout),
        )

    def _split_qkv(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = values.shape
        qkv = self.qkv(values).view(
            batch, length, 3, self.heads, self.head_dimension
        )
        query, key, value = qkv.unbind(dim=2)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def forward(
        self, values: torch.Tensor, sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        query, key, value = self._split_qkv(self.attention_norm(values))
        attended = torch_functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).reshape(values.shape)
        values = values + self.attention_output(attended)
        values = values + self.feedforward(self.feedforward_norm(values))
        return values * sequence_mask.unsqueeze(-1).to(values.dtype)

    def decode_step(
        self,
        values: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self._split_qkv(self.attention_norm(values))
        if cache is not None:
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)
        scale = 1 / math.sqrt(self.head_dimension)
        weights = torch.softmax(query @ key.transpose(-2, -1) * scale, dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(values.shape)
        values = values + self.attention_output(attended)
        values = values + self.feedforward(self.feedforward_norm(values))
        return values, (key, value)
