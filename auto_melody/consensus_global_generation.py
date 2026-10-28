"""Shared full-song beat-grid decoding for the consensus rhythm planner."""

from __future__ import annotations

from fractions import Fraction
import math

import torch

from auto_melody.actions import rhythm_type_counts
from auto_melody.chart_generation import PredictedNote
from auto_melody.dataset import (
    BEAT_GRID_TICKS_PER_BEAT,
    MERT_SAMPLE_RATE,
    encode_beat_grid_window,
    mert_frame_count,
)
from auto_melody.malody import fraction_to_beat


def build_inference_batch(source_chart, audio_waveform: torch.Tensor, target_level: float) -> dict[str, torch.Tensor]:
    window_samples = 5 * MERT_SAMPLE_RATE
    window_count = max(1, math.ceil(audio_waveform.numel() / window_samples))
    audio_windows = torch.zeros((window_count, window_samples), dtype=torch.float32)
    attention_mask = torch.zeros_like(audio_windows, dtype=torch.long)
    frame_indices = []
    candidate_ticks = []
    candidate_window_indices = []
    for window in range(window_count):
        start_sample = window * window_samples
        valid_samples = min(window_samples, max(0, audio_waveform.numel() - start_sample))
        if valid_samples:
            audio_windows[window, :valid_samples] = audio_waveform[start_sample : start_sample + valid_samples]
            attention_mask[window, :valid_samples] = 1
        beat_grid = encode_beat_grid_window(
            source_chart,
            window_start_seconds=window * 5.0,
            window_seconds=valid_samples / MERT_SAMPLE_RATE,
            num_frames=max(1, mert_frame_count(max(1, valid_samples))),
        )
        if beat_grid.ticks.numel():
            frame_indices.append(beat_grid.frame_indices)
            candidate_ticks.append(beat_grid.ticks)
            candidate_window_indices.append(
                torch.full_like(beat_grid.ticks, window, dtype=torch.long)
            )
    if not candidate_ticks:
        raise RuntimeError("audio contains no valid beat-grid candidates")
    return {
        "input_values": audio_windows,
        "attention_mask": attention_mask,
        "difficulty": torch.full((window_count,), max(0.0, min(1.0, target_level / 40.0))),
        "beat_window_indices": torch.cat(candidate_window_indices),
        "beat_frame_indices": torch.cat(frame_indices),
        "beat_ticks": torch.cat(candidate_ticks),
        "beat_song_lengths": torch.tensor([sum(item.numel() for item in candidate_ticks)]),
    }


@torch.inference_mode()
def decode_rhythm_plan(
    rhythm_model,
    inference_batch: dict[str, torch.Tensor],
    source_chart,
    device: torch.device,
    temperature: float,
    *,
    seed: int = 42,
) -> list[PredictedNote]:
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        rhythm_output = rhythm_model(
            inference_batch["input_values"].to(device), inference_batch["attention_mask"].to(device),
            inference_batch["difficulty"].to(device), inference_batch["beat_window_indices"].to(device),
            inference_batch["beat_frame_indices"].to(device), inference_batch["beat_ticks"].to(device),
            inference_batch["beat_song_lengths"].to(device),
        )
    candidate_ticks = rhythm_output.ticks[0]
    candidate_mask = rhythm_output.candidate_mask[0]
    predicted_notes: list[PredictedNote] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for measure_index in torch.nonzero(rhythm_output.measure_mask[0]).flatten().tolist():
        valid_candidates = candidate_mask & (rhythm_output.measure_indices[0] == measure_index)
        candidate_positions = torch.nonzero(valid_candidates).flatten()
        event_count = int(torch.expm1(rhythm_output.measure_log_counts[0, measure_index].float()).round().clamp_min(0))
        event_count = min(event_count, int(candidate_positions.numel()))
        if event_count <= 0:
            continue
        selected_positions = candidate_positions[rhythm_output.placement_scores[0, candidate_positions].float().topk(event_count).indices]
        for position in selected_positions.sort().values.tolist():
            type_logits = rhythm_output.type_logits[0, position].float()
            if temperature <= 0:
                type_index = int(type_logits.argmax())
            else:
                probabilities = torch.softmax(type_logits / temperature, dim=-1)
                type_index = int(torch.multinomial(probabilities, 1, generator=generator))
            type_id = type_index + 1
            press_count, hold_count = rhythm_type_counts(torch.tensor(type_id))
            tick = int(candidate_ticks[position])
            start = source_chart.timeline.beat_to_seconds(
                fraction_to_beat(Fraction(tick, BEAT_GRID_TICKS_PER_BEAT))
            ) - source_chart.audio_start_chart_seconds
            duration_ticks = torch.expm1(rhythm_output.duration_logs[0, position].float()).round().clamp_min(1).to(torch.long)
            hold_durations = sorted(int(value) for value in duration_ticks.tolist() if value > 0)[: int(hold_count)]
            for lane in range(int(press_count)):
                end = None
                if lane < len(hold_durations):
                    end_tick = tick + hold_durations[lane]
                    end = source_chart.timeline.beat_to_seconds(
                        fraction_to_beat(Fraction(end_tick, BEAT_GRID_TICKS_PER_BEAT))
                    ) - source_chart.audio_start_chart_seconds
                predicted_notes.append(PredictedNote(start, lane, end))
    return sorted(predicted_notes, key=lambda note: (note.start_seconds, note.column))
