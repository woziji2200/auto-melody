"""Shared training helpers for the consensus rhythm stages."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from auto_melody.actions import action_classes_from_pattern_and_holds, action_to_rhythm_type
from auto_melody.dataset import SongWindowBatchSampler


def build_song_batch_sampler(
    window_counts,
    *,
    max_windows: int,
    shuffle: bool,
    seed: int,
):
    sampler = SongWindowBatchSampler(
        window_counts,
        max_windows=max_windows,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )
    return sampler


def build_data_loader(dataset, *, batch_sampler, workers: int, collate_fn):
    options: dict[str, Any] = {
        "batch_sampler": batch_sampler,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }
    if workers:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def cycle_data_batches(loader):
    while True:
        seen = False
        for batch in loader:
            seen = True
            yield batch
        if not seen:
            raise RuntimeError("empty training loader")


def pad_flattened_song_values(values: torch.Tensor, lengths: torch.Tensor, width: int, *, fill: float = 0.0) -> torch.Tensor:
    result = values.new_full((lengths.numel(), width), fill)
    for index, part in enumerate(values.split([int(value) for value in lengths.tolist()])):
        result[index, : part.numel()] = part
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_training_device() -> tuple[torch.device, torch.dtype | None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, torch.bfloat16 if device.type == "cuda" else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def move_batch_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def build_rhythm_targets(batch: dict[str, torch.Tensor], rhythm_output) -> tuple[torch.Tensor, torch.Tensor]:
    actions = action_classes_from_pattern_and_holds(
        batch["beat_pattern_targets"], batch["beat_hold_targets"]
    )
    types = action_to_rhythm_type(actions)
    parts = list(types.split([int(value) for value in batch["beat_song_lengths"].tolist()]))
    duration_parts = list(
        batch["beat_hold_duration_ticks"].split(
            [int(value) for value in batch["beat_song_lengths"].tolist()]
        )
    )
    target_types = torch.full_like(rhythm_output.ticks, -1)
    target_durations = torch.zeros(
        (*rhythm_output.ticks.shape, 4), dtype=torch.float32, device=rhythm_output.ticks.device
    )
    for index, (part, durations) in enumerate(zip(parts, duration_parts)):
        width = part.numel()
        target_types[index, :width] = part
        for position in torch.nonzero(part > 0).flatten().tolist():
            values = durations[position][durations[position] > 0].sort().values
            target_durations[index, position, : values.numel()] = torch.log1p(values)
    return target_types, target_durations


def count_events_per_measure(target_types: torch.Tensor, rhythm_output) -> torch.Tensor:
    counts = torch.zeros_like(rhythm_output.measure_log_counts)
    for song in range(target_types.shape[0]):
        valid = rhythm_output.candidate_mask[song]
        indices = rhythm_output.measure_indices[song, valid]
        values = (target_types[song, valid] > 0).to(counts.dtype)
        counts[song].index_add_(0, indices, values)
    return counts


def within_measure_ranking_loss(scores: torch.Tensor, targets: torch.Tensor, rhythm_output) -> torch.Tensor:
    losses = []
    for song in range(scores.shape[0]):
        for measure in torch.nonzero(rhythm_output.measure_mask[song]).flatten().tolist():
            valid = rhythm_output.candidate_mask[song] & (rhythm_output.measure_indices[song] == measure)
            positive = valid & (targets[song] > 0)
            if bool(positive.any()):
                log_probs = functional.log_softmax(scores[song, valid], dim=0)
                local_positive = positive[valid]
                losses.append(-log_probs[local_positive].mean())
    return torch.stack(losses).mean() if losses else scores.new_zeros(())


def budgeted_selection_f1(scores: torch.Tensor, targets: torch.Tensor, counts: torch.Tensor, rhythm_output) -> torch.Tensor:
    predicted = torch.zeros_like(rhythm_output.candidate_mask)
    for song in range(scores.shape[0]):
        for measure in torch.nonzero(rhythm_output.measure_mask[song]).flatten().tolist():
            local = rhythm_output.candidate_mask[song] & (rhythm_output.measure_indices[song] == measure)
            count = min(int(torch.expm1(counts[song, measure]).round().clamp_min(0)), int(local.sum()))
            if count:
                positions = torch.nonzero(local).flatten()
                selected = scores[song, positions].topk(count).indices
                predicted[song, positions[selected]] = True
    positive = targets > 0
    valid = rhythm_output.candidate_mask
    tp = (predicted & positive & valid).sum().float()
    fp = (predicted & ~positive & valid).sum().float()
    fn = (~predicted & positive & valid).sum().float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def rhythm_loss_metrics(
    rhythm_output,
    batch: dict[str, torch.Tensor],
    *,
    count_weight: float = 1.0,
    contour_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets, durations = build_rhythm_targets(batch, rhythm_output)
    valid = rhythm_output.candidate_mask
    positive = valid & (targets > 0)
    counts = count_events_per_measure(targets, rhythm_output)
    count_loss = functional.smooth_l1_loss(rhythm_output.measure_log_counts[rhythm_output.measure_mask], torch.log1p(counts[rhythm_output.measure_mask]))
    contour_terms = []
    predicted_counts = torch.expm1(rhythm_output.measure_log_counts).clamp_min(0)
    for song in range(counts.shape[0]):
        valid_measures = rhythm_output.measure_mask[song]
        target_curve = counts[song, valid_measures]
        predicted_curve = predicted_counts[song, valid_measures]
        contour_terms.append(
            functional.smooth_l1_loss(
                predicted_curve / predicted_curve.mean().clamp_min(1),
                target_curve / target_curve.mean().clamp_min(1),
            )
        )
    contour_loss = torch.stack(contour_terms).mean() if contour_terms else count_loss.new_zeros(())
    rank_loss = within_measure_ranking_loss(rhythm_output.placement_scores, targets, rhythm_output)
    type_loss = functional.cross_entropy(rhythm_output.type_logits[positive], targets[positive] - 1) if bool(positive.any()) else count_loss.new_zeros(())
    duration_mask = positive.unsqueeze(-1) & (durations > 0)
    duration_loss = functional.smooth_l1_loss(rhythm_output.duration_logs[duration_mask], durations[duration_mask]) if bool(duration_mask.any()) else count_loss.new_zeros(())
    total = (
        count_weight * count_loss
        + contour_weight * contour_loss
        + rank_loss
        + type_loss
        + 0.1 * duration_loss
    )
    f1 = budgeted_selection_f1(rhythm_output.placement_scores.detach(), targets, rhythm_output.measure_log_counts.detach(), rhythm_output)
    type_accuracy = (rhythm_output.type_logits[positive].argmax(dim=-1) == targets[positive] - 1).float().mean() if bool(positive.any()) else total.new_zeros(())
    return total, {
        "loss": float(total.detach()),
        "count_loss": float(count_loss.detach()),
        "count_weight": float(count_weight),
        "contour_loss": float(contour_loss.detach()),
        "contour_weight": float(contour_weight),
        "rank_loss": float(rank_loss.detach()),
        "type_loss": float(type_loss.detach()),
        "duration_loss": float(duration_loss.detach()),
        "selection_f1": float(f1),
        "type_accuracy": float(type_accuracy),
    }
