"""Audio decoding, Malody chart construction, and generation diagnostics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import PurePosixPath
from typing import Any, Sequence
import zipfile

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample

from .dataset import MERT_SAMPLE_RATE
from .malody import LoadedChart, TimedNote, beat_fraction, fraction_to_beat


@dataclass(frozen=True)
class PredictedNote:
    """A generated note in the input audio's time coordinate system."""

    start_seconds: float
    column: int
    end_seconds: float | None = None
    confidence: float = 0.0

    @property
    def is_hold(self) -> bool:
        return self.end_seconds is not None


def read_chart_audio(
    chart: LoadedChart, *, sample_rate: int = MERT_SAMPLE_RATE
) -> torch.Tensor:
    """Decode a ready chart's full-track audio to mono float32."""

    member = chart.primary_audio_event.member
    assert member is not None
    with zipfile.ZipFile(chart.archive_path) as archive:
        with archive.open(member) as compressed:
            with sf.SoundFile(compressed) as audio_file:
                source_rate = audio_file.samplerate
                array = audio_file.read(dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.asarray(array)).mean(dim=1)
    if source_rate != sample_rate and waveform.numel():
        waveform = resample(waveform, source_rate, sample_rate)
    return waveform.to(dtype=torch.float32).contiguous()


def _quantized_chart_beat(
    chart: LoadedChart, audio_seconds: float, *, resolution: int
) -> tuple[int, int, int]:
    chart_seconds = audio_seconds + chart.audio_start_chart_seconds
    raw = beat_fraction(chart.timeline.seconds_to_beat(chart_seconds))
    return fraction_to_beat(Fraction(round(float(raw) * resolution), resolution))


def build_generated_chart(
    source: LoadedChart,
    notes: Sequence[PredictedNote],
    *,
    difficulty_level: float,
    creator: str = "Auto-Melody",
    background: str = "",
    beat_resolution: int = 960,
) -> dict[str, Any]:
    """Build an MC object while retaining the source song's timing map."""

    if beat_resolution <= 0:
        raise ValueError("beat_resolution must be positive")
    if background and PurePosixPath(background).name != background:
        raise ValueError("background must be a plain filename")
    audio_event = source.primary_audio_event
    audio_filename = PurePosixPath(audio_event.sound).name
    meta = deepcopy(source.data["meta"])
    meta.update(
        creator=creator,
        background=background,
        version=f"4K AI Lv.{difficulty_level:g}",
        id=0,
        mode=0,
    )
    mode_ext = dict(meta.get("mode_ext") or {})
    mode_ext["column"] = 4
    meta["mode_ext"] = mode_ext

    events: list[dict[str, Any]] = []
    for note in notes:
        event: dict[str, Any] = {
            "beat": list(
                _quantized_chart_beat(
                    source, note.start_seconds, resolution=beat_resolution
                )
            ),
            "column": note.column,
        }
        if note.end_seconds is not None:
            end_beat = _quantized_chart_beat(
                source, note.end_seconds, resolution=beat_resolution
            )
            if beat_fraction(end_beat) > beat_fraction(event["beat"]):
                event["endbeat"] = list(end_beat)
        events.append(event)

    # Malody auto requires room between an LN release and the next same-lane press.
    by_lane: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    for event in events:
        by_lane[int(event["column"])].append(event)
    minimum_release = Fraction(1, 4)
    for lane_events in by_lane:
        lane_events.sort(key=lambda value: beat_fraction(value["beat"]))
        for current, following in zip(lane_events, lane_events[1:]):
            if "endbeat" not in current:
                continue
            latest_end = beat_fraction(following["beat"]) - minimum_release
            start = beat_fraction(current["beat"])
            if latest_end <= start:
                current.pop("endbeat", None)
            elif beat_fraction(current["endbeat"]) > latest_end:
                current["endbeat"] = list(fraction_to_beat(latest_end))
    events.append(
        {
            "beat": list(audio_event.beat),
            "sound": audio_filename,
            "vol": audio_event.volume if audio_event.volume is not None else 100,
            "offset": audio_event.offset_ms,
            "type": 1,
        }
    )
    return {
        "meta": meta,
        "time": deepcopy(source.data["time"]),
        "effect": [],
        "note": events,
    }


def _pattern_summary(patterns: Sequence[int]) -> dict[str, Any]:
    four_grams = [
        tuple(patterns[index : index + 4])
        for index in range(max(0, len(patterns) - 3))
    ]
    counts: dict[tuple[int, ...], int] = {}
    for value in four_grams:
        counts[value] = counts.get(value, 0) + 1
    longest_alternation = 0
    for start in range(len(patterns)):
        end = min(len(patterns), start + 2)
        while (
            end < len(patterns)
            and patterns[end] == patterns[end - 2]
            and patterns[end] != patterns[end - 1]
        ):
            end += 1
        longest_alternation = max(longest_alternation, end - start)
    longest_repeat = 0
    run = 0
    previous = None
    for pattern in patterns:
        run = run + 1 if pattern == previous else 1
        longest_repeat = max(longest_repeat, run)
        previous = pattern
    return {
        "unique_4gram_ratio": len(counts) / max(1, len(four_grams)),
        "dominant_4gram_ratio": max(counts.values(), default=0)
        / max(1, len(four_grams)),
        "longest_abab_run": longest_alternation,
        "longest_same_mask_run": longest_repeat,
    }


def chart_quality_summary(
    source: LoadedChart,
    notes: Sequence[PredictedNote | TimedNote],
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    """Summarize timing, density, chord, lane, pattern, and hold properties."""

    lane_counts = [0, 0, 0, 0]
    lane_notes: list[list[PredictedNote | TimedNote]] = [[] for _ in range(4)]
    holds = []
    starts = []
    grid_errors_ms = []
    for note in notes:
        start = note.start_seconds
        starts.append((start, note.column))
        lane_counts[note.column] += 1
        lane_notes[note.column].append(note)
        if note.end_seconds is not None:
            holds.append(float(note.end_seconds) - start)
        chart_seconds = start + source.audio_start_chart_seconds
        position = float(beat_fraction(source.timeline.seconds_to_beat(chart_seconds)))
        point = min(
            source.timeline.points,
            key=lambda item: abs(item.seconds - chart_seconds),
        )
        straight_error = abs(position - round(position * 8) / 8)
        triplet_error = abs(position - round(position * 6) / 6)
        grid_errors_ms.append(min(straight_error, triplet_error) * 60_000 / point.bpm)

    starts.sort()
    chord_sizes = []
    onset_times = []
    patterns = []
    index = 0
    while index < len(starts):
        group_end = index + 1
        while (
            group_end < len(starts)
            and starts[group_end][0] - starts[index][0] <= 0.025
        ):
            group_end += 1
        onset_times.append(starts[index][0])
        pattern = 0
        for _, column in starts[index:group_end]:
            pattern |= 1 << column
        patterns.append(pattern)
        chord_sizes.append(group_end - index)
        index = group_end

    density_window_count = max(1, math.ceil(duration_seconds / 5.0))
    onset_density = [0] * density_window_count
    for start in onset_times:
        window = min(density_window_count - 1, max(0, int(start / 5.0)))
        onset_density[window] += 1

    release_gaps = []
    for values in lane_notes:
        values.sort(key=lambda note: note.start_seconds)
        for current, following in zip(values, values[1:]):
            if current.end_seconds is None:
                continue
            end_seconds = current.end_seconds + source.audio_start_chart_seconds
            next_seconds = following.start_seconds + source.audio_start_chart_seconds
            end_beat = beat_fraction(source.timeline.seconds_to_beat(end_seconds))
            next_beat = beat_fraction(source.timeline.seconds_to_beat(next_seconds))
            release_gaps.append(float(next_beat - end_beat))

    repeated_lane = sum(
        starts[index][1] == starts[index - 1][1]
        for index in range(1, len(starts))
    )
    count = len(notes)
    sorted_errors = sorted(grid_errors_ms)
    median_error = sorted_errors[len(sorted_errors) // 2] if sorted_errors else 0.0
    return {
        "note_count": count,
        "notes_per_second": count / max(duration_seconds, 1e-9),
        "hold_count": len(holds),
        "hold_ratio": len(holds) / max(count, 1),
        "mean_hold_seconds": sum(holds) / max(len(holds), 1),
        "lane_counts": lane_counts,
        "lane_ratios": [value / max(count, 1) for value in lane_counts],
        "inner_lane_ratio": (lane_counts[1] + lane_counts[2]) / max(count, 1),
        "same_lane_transition_ratio": repeated_lane / max(len(starts) - 1, 1),
        "chord_counts": {
            str(size): chord_sizes.count(size) for size in sorted(set(chord_sizes))
        },
        "onset_density_5s": onset_density,
        "empty_5s_windows": onset_density.count(0),
        "pattern_sequence": _pattern_summary(patterns),
        "hold_release": {
            "minimum_gap_beats": min(release_gaps, default=0.0),
            "under_quarter_beat": sum(value < 0.25 - 1e-6 for value in release_gaps),
        },
        "beat_grid": {
            "definition": "nearest straight-32nd or triplet position",
            "within_10ms": sum(error <= 10 for error in grid_errors_ms) / max(count, 1),
            "within_20ms": sum(error <= 20 for error in grid_errors_ms) / max(count, 1),
            "median_error_ms": median_error,
        },
    }


def _report_chart_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    charts = report.get("charts")
    if isinstance(charts, list):
        return charts
    return [
        {
            "level": report.get("level"),
            "member": report.get("member"),
            "retrieved_measures": report.get("retrieval", {}).get(
                "retrieved_measures"
            ),
            "total_measures": report.get("retrieval", {}).get(
                "total_measures"
            ),
            "similar_measure_consistency": report.get(
                "similar_measure_consistency", {}
            ),
            "skeleton": report.get("skeleton", {}),
            "quality": report.get("quality", {}),
        }
    ]


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render(row: list[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            if index == 0:
                cells.append(value.ljust(widths[index]))
            else:
                cells.append(value.rjust(widths[index]))
        return "  ".join(cells)

    return [
        render(headers),
        "  ".join("-" * width for width in widths),
        *(render(row) for row in rows),
    ]


def format_generation_report(
    report: dict[str, Any],
) -> str:
    """Render the machine-readable generation report as a compact terminal summary."""

    chart_items = _report_chart_items(report)
    rows = []
    for item in chart_items:
        quality = item.get("quality", {})
        chord_counts = quality.get("chord_counts", {})
        onsets = sum(int(value) for value in chord_counts.values())
        chords = sum(
            int(value)
            for size, value in chord_counts.items()
            if int(size) > 1
        )
        holds = int(quality.get("hold_count", 0))
        note_count = int(quality.get("note_count", 0))
        retrieved = item.get("retrieved_measures")
        total_measures = item.get("total_measures")
        consistency = item.get("similar_measure_consistency", {}).get(
            "phrase_retrieval", {}
        )
        similar = consistency.get("lane_state_consistency")
        level = item.get("level")
        level_text = "-" if level is None else f"{float(level):g}"
        rows.append(
            [
                level_text,
                f"{note_count:,}",
                f"{float(quality.get('notes_per_second', 0.0)):.2f}",
                f"{holds:,} ({float(quality.get('hold_ratio', 0.0)):.1%})",
                f"{chords:,} ({chords / max(onsets, 1):.1%})",
                "-"
                if retrieved is None or total_measures is None
                else (
                    f"{int(retrieved):,}/{int(total_measures):,} "
                    f"({int(retrieved) / max(int(total_measures), 1):.1%})"
                ),
                "-" if similar is None else f"{float(similar):.1%}",
            ]
        )

    lines = ["", "GENERATION COMPLETE", "=" * 88, "", "Charts"]
    lines.extend(
        _format_table(
            [
                "Lv",
                "Note",
                "Note/s",
                "Hold Note",
                "Chords",
                "Retrieved/Total",
                "Similar",
            ],
            rows,
        )
    )
    lines.append("=" * 88)
    return "\n".join(lines)
