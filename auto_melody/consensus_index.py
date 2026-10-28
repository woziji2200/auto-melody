"""Reusable chart event index for consensus rhythm training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from auto_melody.dataset import (
    annotate_difficulties,
    nearest_beat_grid_tick,
    read_manifest,
    select_training_charts,
)
from auto_melody.malody import LoadedChart, beat_fraction, load_chart


SEQUENCE_INDEX_SCHEMA = "malody_4k_rhythm_sequence_index_v3"


def extract_chart_events(chart: LoadedChart) -> tuple[tuple[int, int, tuple[int, int, int, int]], ...]:
    """Merge notes at one beat-grid tick into a 4K action and Hold durations."""

    grouped: dict[int, tuple[list[int], list[int]]] = {}
    for note in chart.notes:
        tick = nearest_beat_grid_tick(beat_fraction(note.beat))
        if tick < 0:
            continue
        states, durations = grouped.setdefault(tick, ([0, 0, 0, 0], [0, 0, 0, 0]))
        if note.end_beat is None:
            states[note.column] = max(states[note.column], 1)
            continue
        end_tick = nearest_beat_grid_tick(beat_fraction(note.end_beat))
        states[note.column] = 2
        durations[note.column] = max(1, end_tick - tick)
    events = []
    for tick, (states, durations) in sorted(grouped.items()):
        action = sum(state * (3**lane) for lane, state in enumerate(states)) - 1
        events.append((tick, action, tuple(durations)))
    return tuple(events)


def build_sequence_index(
    manifest_path: str | Path,
    dataset_dir: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """Persist chart events for salience support and phrase retrieval."""

    rows = annotate_difficulties(select_training_charts(read_manifest(manifest_path)))
    root = Path(dataset_dir)
    records: list[dict[str, Any]] = []
    for row in rows:
        chart = load_chart(root / str(row["archive"]), str(row["chart_member"]))
        events = extract_chart_events(chart)
        encoded = torch.tensor(
            [[tick, action, *durations] for tick, action, durations in events],
            dtype=torch.int32,
        )
        if not events:
            encoded = torch.empty((0, 6), dtype=torch.int32)
        records.append({"row": row, "events": encoded})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SEQUENCE_INDEX_SCHEMA,
            "manifest": str(Path(manifest_path)),
            "dataset_dir": str(root),
            "records": records,
        },
        output,
    )
    return {"charts": len(records)}
