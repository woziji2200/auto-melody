"""Difficulty-aware same-song onset agreement for the rhythm mainline."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from auto_melody.dataset import MalodySongDataset, WindowReference


def _chart_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["archive"]), str(row["chart_member"])


def build_difficulty_weighted_support(
    index_path: str | Path,
    *,
    scale_levels: float,
) -> dict[tuple[str, str], dict[int, float]]:
    """Return per-chart onset agreement, weighted by nearby chart difficulties."""

    if scale_levels <= 0:
        raise ValueError("scale_levels must be positive")
    index = torch.load(index_path, map_location="cpu", weights_only=False)
    records = index.get("records")
    if not isinstance(records, list):
        raise ValueError(f"rhythm index has no records: {index_path}")
    by_audio: dict[str, list[tuple[tuple[str, str], float, set[int]]]] = defaultdict(list)
    for record in records:
        row = record["row"]
        events = record["events"]
        ticks = {int(value[0]) for value in events.tolist()}
        by_audio[str(row["audio_fingerprint"])].append(
            (_chart_key(row), float(row["difficulty_level"]), ticks)
        )
    result: dict[tuple[str, str], dict[int, float]] = {}
    for charts in by_audio.values():
        for target_key, target_level, _ in charts:
            weights = [math.exp(-abs(level - target_level) / scale_levels) for _, level, _ in charts]
            denominator = sum(weights)
            agreement: dict[int, float] = defaultdict(float)
            for weight, (_, _, ticks) in zip(weights, charts):
                for tick in ticks:
                    agreement[tick] += weight / denominator
            result[target_key] = dict(agreement)
    return result


class OnsetSupportSongDataset(MalodySongDataset):
    """A full-song dataset carrying an onset-agreement scalar for each tick."""

    def __init__(
        self,
        *args: Any,
        rhythm_index: str | Path,
        support_scale_levels: float = 4.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.support = build_difficulty_weighted_support(
            rhythm_index, scale_levels=support_scale_levels
        )
        missing = [_chart_key(row) for row in self.rows if _chart_key(row) not in self.support]
        if missing:
            raise ValueError(f"{len(missing)} charts are missing from onset support index")

    def _load_reference(
        self, reference: WindowReference, *, mirror: bool | None = None
    ) -> dict[str, Any]:
        result = super()._load_reference(reference, mirror=mirror)
        row = self.rows[reference.row_index]
        support = self.support[_chart_key(row)]
        result["beat_support_targets"] = torch.tensor(
            [float(support.get(int(tick), 0.0)) for tick in result["beat_ticks"].tolist()],
            dtype=torch.float32,
        )
        return result
