"""Natural four-lane arrangement on a fixed, lane-invariant rhythm skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from auto_melody.actions import ACTION_BOS_TOKEN, ACTION_VOCAB_SIZE, action_classes_to_lane_states
from auto_melody.dataset import BEAT_GRID_TICKS_PER_MEASURE, read_manifest
from auto_melody.malody import load_chart
from auto_melody.consensus_index import extract_chart_events
from auto_melody.causal_block import CausalTransformerBlock


LANE_ARRANGER_SCHEMA = "malody_4k_lane_arranger_v2"


@dataclass
class LaneActionOutput:
    logits: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor


class LaneArranger(nn.Module):
    """Generate lane actions while preserving press/Hold cardinality."""

    def __init__(
        self,
        *,
        decoder_dim: int = 384,
        decoder_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if decoder_dim < 128 or decoder_dim % 8 or decoder_layers <= 0:
            raise ValueError("decoder dimensions must be valid")
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.action_embedding = nn.Embedding(ACTION_VOCAB_SIZE + 1, decoder_dim)
        self.phase_embedding = nn.Embedding(BEAT_GRID_TICKS_PER_MEASURE, decoder_dim)
        self.type_embedding = nn.Embedding(25, decoder_dim)
        self.gap_projection = nn.Linear(2, decoder_dim)
        self.difficulty_projection = nn.Sequential(
            nn.Linear(1, decoder_dim // 4),
            nn.SiLU(),
            nn.Linear(decoder_dim // 4, decoder_dim),
        )
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(decoder_dim, heads=8, dropout=dropout)
            for _ in range(decoder_layers)
        )
        self.output_norm = nn.LayerNorm(decoder_dim)
        self.output = nn.Linear(decoder_dim, ACTION_VOCAB_SIZE)
        states = action_classes_to_lane_states(torch.arange(ACTION_VOCAB_SIZE))
        self.register_buffer("action_lane_states", states, persistent=False)
        self.register_buffer(
            "action_press_counts", (states > 0).sum(dim=-1), persistent=False
        )
        self.register_buffer(
            "action_hold_counts", (states == 2).sum(dim=-1), persistent=False
        )

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
        positions = torch.arange(width, device=lengths.device).unsqueeze(0)
        return positions < lengths.unsqueeze(1)

    @staticmethod
    def _sinusoidal_positions(width: int, dimension: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / dimension)
        )
        result = torch.zeros((width, dimension), device=device, dtype=torch.float32)
        result[:, 0::2] = torch.sin(position * frequency)
        result[:, 1::2] = torch.cos(position * frequency)
        return result.to(dtype)

    def _build_event_embeddings(
        self,
        ticks: torch.Tensor,
        previous_actions: torch.Tensor,
        press_counts: torch.Tensor,
        hold_counts: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        if ticks.ndim != 2 or previous_actions.shape != ticks.shape:
            raise ValueError("ticks and previous actions must have matching [song, event] shape")
        if difficulty.shape != (ticks.shape[0],):
            raise ValueError("difficulty must contain one value per song")
        previous_gap = torch.zeros_like(ticks)
        previous_gap[:, 1:] = ticks[:, 1:] - ticks[:, :-1]
        next_gap = torch.zeros_like(ticks)
        next_gap[:, :-1] = previous_gap[:, 1:]
        gaps = torch.stack((previous_gap, next_gap), dim=-1).to(
            self.action_embedding.weight.dtype
        )
        type_ids = press_counts.clamp(0, 4) * 5 + hold_counts.clamp(0, 4)
        return (
            self.action_embedding(previous_actions)
            + self.phase_embedding(ticks.remainder(BEAT_GRID_TICKS_PER_MEASURE))
            + self.type_embedding(type_ids)
            + self.gap_projection(torch.log1p(gaps.clamp_min(0)))
            + self.difficulty_projection(difficulty.unsqueeze(-1)).unsqueeze(1)
            + self._sinusoidal_positions(
                ticks.shape[1], self.decoder_dim,
                device=ticks.device,
                dtype=self.action_embedding.weight.dtype,
            ).unsqueeze(0)
        )

    def _mask_illegal_actions(
        self,
        logits: torch.Tensor,
        press_counts: torch.Tensor,
        hold_counts: torch.Tensor,
        blocked_lanes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        legal = (
            self.action_press_counts.view(*((1,) * press_counts.ndim), -1)
            == press_counts.unsqueeze(-1)
        ) & (
            self.action_hold_counts.view(*((1,) * hold_counts.ndim), -1)
            == hold_counts.unsqueeze(-1)
        )
        if blocked_lanes is not None:
            lane_states = self.action_lane_states.view(
                *((1,) * press_counts.ndim), ACTION_VOCAB_SIZE, 4
            )
            conflicts = ((lane_states > 0) & blocked_lanes.unsqueeze(-2)).any(dim=-1)
            legal = legal & ~conflicts
        if bool((~legal.any(dim=-1)).any()):
            raise ValueError("skeleton event has no legal playable action")
        return logits.masked_fill(~legal, -torch.inf)

    def teacher_forced(
        self,
        ticks: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> LaneActionOutput:
        mask = self._sequence_mask(lengths, ticks.shape[1])
        safe_targets = targets.clamp_min(0)
        target_states = self.action_lane_states[safe_targets]
        press_counts = (target_states > 0).sum(dim=-1)
        hold_counts = (target_states == 2).sum(dim=-1)
        previous = torch.full_like(safe_targets, ACTION_BOS_TOKEN)
        previous[:, 1:] = safe_targets[:, :-1]
        values = self._build_event_embeddings(ticks, previous, press_counts, hold_counts, difficulty)
        for block in self.blocks:
            values = block(values, mask)
        logits = self._mask_illegal_actions(
            self.output(self.output_norm(values)), press_counts, hold_counts
        )
        return LaneActionOutput(logits, targets, mask)

    @staticmethod
    def _sample_action(logits: torch.Tensor, *, temperature: float, top_p: float, generator: torch.Generator) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax().reshape(1)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        keep = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities < top_p
        filtered = sorted_probabilities * keep
        sampled = torch.multinomial(
            filtered / filtered.sum().clamp_min(1e-12), 1, generator=generator
        )
        return sorted_indices.gather(0, sampled)

    @torch.inference_mode()
    def sample_actions(
        self,
        ticks: torch.Tensor,
        press_counts: torch.Tensor,
        hold_counts: torch.Tensor,
        *,
        difficulty: float,
        hold_duration_ticks: torch.Tensor | None = None,
        release_gap_ticks: int = BEAT_GRID_TICKS_PER_MEASURE // 16,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 42,
    ) -> torch.Tensor:
        if ticks.ndim != 1 or press_counts.shape != ticks.shape or hold_counts.shape != ticks.shape:
            raise ValueError("ticks and type counts must be one aligned event sequence")
        if hold_duration_ticks is not None:
            if hold_duration_ticks.shape != (ticks.numel(), 4) or bool((hold_duration_ticks < 0).any()):
                raise ValueError("Hold durations must have shape [event, 4] and be nonnegative")
            duration_counts = (hold_duration_ticks > 0).sum(dim=-1).to(hold_counts.device)
            if not torch.equal(duration_counts, hold_counts):
                raise ValueError("Hold durations must match the skeleton Hold counts")
        if not ticks.numel():
            return torch.empty(0, dtype=torch.long)
        if not 0 < top_p <= 1 or release_gap_ticks < 0:
            raise ValueError("sampling parameters are invalid")
        device = ticks.device
        generator = torch.Generator(device=device).manual_seed(seed)
        difficulty_tensor = torch.tensor([difficulty], device=device)
        positions = self._sinusoidal_positions(
            ticks.numel(), self.decoder_dim, device=device,
            dtype=self.action_embedding.weight.dtype,
        )
        previous = torch.full((1,), ACTION_BOS_TOKEN, dtype=torch.long, device=device)
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.blocks)
        active_until = torch.full((4,), -1, dtype=ticks.dtype, device=device)
        durations = hold_duration_ticks.to(device=device, dtype=ticks.dtype) if hold_duration_ticks is not None else None
        decoded = []
        for index in range(ticks.numel()):
            previous_gap = ticks[index] - ticks[index - 1] if index else ticks.new_zeros(())
            next_gap = ticks[index + 1] - ticks[index] if index + 1 < ticks.numel() else ticks.new_zeros(())
            gaps = torch.stack((previous_gap, next_gap)).to(self.action_embedding.weight.dtype)
            type_ids = press_counts[index].clamp(0, 4) * 5 + hold_counts[index].clamp(0, 4)
            values = (
                self.action_embedding(previous).squeeze(0)
                + self.phase_embedding(ticks[index].remainder(BEAT_GRID_TICKS_PER_MEASURE))
                + self.type_embedding(type_ids)
                + self.gap_projection(torch.log1p(gaps.clamp_min(0)))
                + self.difficulty_projection(difficulty_tensor.unsqueeze(-1)).squeeze(0)
                + positions[index]
            ).reshape(1, 1, -1)
            for block_index, block in enumerate(self.blocks):
                values, caches[block_index] = block.decode_step(values, caches[block_index])
            logits = self.output(self.output_norm(values[0, 0]))
            logits = self._mask_illegal_actions(
                logits, press_counts[index], hold_counts[index], active_until > ticks[index]
            )
            action = self._sample_action(logits, temperature=temperature, top_p=top_p, generator=generator)
            if durations is not None and int(hold_counts[index]) > 0:
                held_lanes = torch.nonzero(self.action_lane_states[action.item()] == 2).flatten()
                event_durations = durations[index][durations[index] > 0].sort().values
                active_until[held_lanes] = ticks[index] + event_durations + release_gap_ticks
            previous = action
            decoded.append(action)
        return torch.cat(decoded).cpu()


class LaneActionDataset(Dataset[dict[str, Any]]):
    """Load complete human action sequences for fixed-skeleton training."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_dir: str | Path,
        *,
        mirror_probability: float = 0.5,
    ) -> None:
        if not 0 <= mirror_probability <= 1:
            raise ValueError("mirror_probability must be in [0, 1]")
        root = Path(dataset_dir)
        self.mirror_probability = mirror_probability
        self.items: list[dict[str, Any]] = []
        for row in read_manifest(manifest_path):
            chart = load_chart(root / str(row["archive"]), str(row["chart_member"]))
            events = extract_chart_events(chart)
            if not events:
                continue
            self.items.append({
                "ticks": torch.tensor([event[0] for event in events], dtype=torch.long),
                "actions": torch.tensor([event[1] for event in events], dtype=torch.long),
                "difficulty": float(row["difficulty"]),
            })
        if not self.items:
            raise ValueError("lane arranger dataset is empty")
        self.lengths = [int(item["ticks"].numel()) for item in self.items]

    @staticmethod
    def _mirror(actions: torch.Tensor) -> torch.Tensor:
        states = action_classes_to_lane_states(actions).flip(dims=(-1,))
        powers = states.new_tensor((1, 3, 9, 27))
        return (states * powers).sum(dim=-1) - 1

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        actions = item["actions"]
        if self.mirror_probability and torch.rand(()) < self.mirror_probability:
            actions = self._mirror(actions)
        return {"ticks": item["ticks"], "actions": actions, "difficulty": item["difficulty"]}


def collate_lane_actions(items: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([int(item["ticks"].numel()) for item in items], dtype=torch.long)
    return {
        "ticks": pad_sequence([item["ticks"] for item in items], batch_first=True),
        "actions": pad_sequence([item["actions"] for item in items], batch_first=True, padding_value=-1),
        "lengths": lengths,
        "difficulty": torch.tensor([item["difficulty"] for item in items], dtype=torch.float32),
    }


class EventCountBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: Sequence[int], *, max_events: int, shuffle: bool, generator: torch.Generator | None = None) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.lengths = tuple(int(value) for value in lengths)
        self.max_events = max_events
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[list[int]]:
        indices = torch.randperm(len(self.lengths), generator=self.generator).tolist() if self.shuffle else list(range(len(self.lengths)))
        batch: list[int] = []
        used = 0
        for index in indices:
            length = self.lengths[index]
            if batch and used + length > self.max_events:
                yield batch
                batch, used = [], 0
            batch.append(index)
            used += length
        if batch:
            yield batch

    def __len__(self) -> int:
        return max(1, math.ceil(sum(self.lengths) / self.max_events))
