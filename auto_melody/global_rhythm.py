"""Shared beat-grid modules for the rhythm mainline."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from auto_melody.dataset import BEAT_GRID_TICKS_PER_MEASURE


@dataclass
class RhythmPlanOutput:
    placement_scores: torch.Tensor
    measure_log_counts: torch.Tensor
    type_logits: torch.Tensor
    duration_logs: torch.Tensor
    ticks: torch.Tensor
    candidate_mask: torch.Tensor
    measure_indices: torch.Tensor
    measure_mask: torch.Tensor


class TemporalBlock(nn.Module):
    """One depthwise residual block over adjacent beat-grid candidates."""

    def __init__(self, dimension: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=dimension,
        )
        self.pointwise = nn.Conv1d(dimension, dimension * 2, kernel_size=1)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(values.transpose(1, 2))
        hidden = torch.nn.functional.glu(self.pointwise(hidden), dim=1)
        values = self.norm(values + self.dropout(hidden.transpose(1, 2)))
        return values * mask.unsqueeze(-1).to(values.dtype)


def _state_dict_without(
    module: nn.Module, *prefixes: str
) -> dict[str, torch.Tensor]:
    state = module.state_dict()
    if not prefixes:
        return dict(state)
    return {
        name: value
        for name, value in state.items()
        if not any(name.startswith(prefix) for prefix in prefixes)
    }


def _load_state_dict_without(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...],
    label: str,
) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)
    invalid_missing = [
        name for name in missing if not any(name.startswith(prefix) for prefix in prefixes)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint mismatch: missing={invalid_missing}, "
            f"unexpected={unexpected}"
        )


def _sequence_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    positions = torch.arange(width, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _measure_positions(
    width: int, dimension: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    positions = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    result = torch.zeros((width, dimension), device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies)
    return result.to(dtype)


def _measure_context(
    values: torch.Tensor,
    ticks: torch.Tensor,
    mask: torch.Tensor,
    *,
    measure_encoder: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, dimension = values.shape
    measure_indices = torch.zeros_like(ticks)
    summaries = []
    lengths = []
    for index in range(batch):
        valid = mask[index]
        song_ticks = ticks[index, valid]
        absolute = torch.div(
            song_ticks, BEAT_GRID_TICKS_PER_MEASURE, rounding_mode="floor"
        )
        relative = absolute - absolute.min()
        count = int(relative.max()) + 1
        measure_indices[index, valid] = relative
        selected = values[index, valid]
        summed = values.new_zeros((count, dimension))
        totals = values.new_zeros((count, 1))
        summed.index_add_(0, relative, selected)
        totals.index_add_(0, relative, values.new_ones((relative.numel(), 1)))
        maximum = values.new_full((count, dimension), -torch.inf)
        maximum.scatter_reduce_(
            0,
            relative.unsqueeze(-1).expand(-1, dimension),
            selected,
            reduce="amax",
            include_self=True,
        )
        summaries.append((summed / totals.clamp_min(1) + maximum) * 0.5)
        lengths.append(count)
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=values.device)
    padded = pad_sequence(summaries, batch_first=True)
    measure_mask = _sequence_mask(length_tensor, padded.shape[1])
    contextual = measure_encoder(
        padded
        + _measure_positions(
            padded.shape[1], dimension, device=values.device, dtype=values.dtype
        ).unsqueeze(0),
        src_key_padding_mask=~measure_mask,
    )
    contextual = contextual * measure_mask.unsqueeze(-1).to(contextual.dtype)
    broadcast = torch.stack(
        [contextual[index][measure_indices[index]] for index in range(batch)]
    )
    return (
        broadcast * mask.unsqueeze(-1).to(broadcast.dtype),
        contextual,
        measure_indices,
        measure_mask,
    )
