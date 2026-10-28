"""Windowed Malody data aligned to the local MERT-v1-330M encoder."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence
import zipfile

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as torch_functional
from torch.utils.data import Dataset
from torchaudio.functional import resample

from .audio_cache import AudioCache
from .malody import LoadedChart, beat_fraction, fraction_to_beat, load_chart


MERT_SAMPLE_RATE = 24_000
MERT_CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
MERT_CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)
MERT_HOP_SAMPLES = math.prod(MERT_CONV_STRIDE)
MERT_RECEPTIVE_FIELD_SAMPLES = 400
MERT_FIRST_FRAME_CENTER = (MERT_RECEPTIVE_FIELD_SAMPLES - 1) / 2

MAX_DIFFICULTY_LEVEL = 40.0
BEAT_GRID_TICKS_PER_BEAT = 24
BEAT_GRID_TICKS_PER_MEASURE = BEAT_GRID_TICKS_PER_BEAT * 4
BEAT_GRID_OFFSETS = tuple(
    tick
    for tick in range(BEAT_GRID_TICKS_PER_BEAT)
    if tick % 3 == 0 or tick % 4 == 0
)
_LEVEL_PATTERN = re.compile(r"(?i)\blv\.?\s*([0-9]+(?:\.[0-9]+)?)")


def mert_frame_count(num_samples: int) -> int:
    """Return MERT's convolutional output length for an input sample count."""

    length = num_samples
    for kernel, stride in zip(MERT_CONV_KERNEL, MERT_CONV_STRIDE):
        length = (length - kernel) // stride + 1
        if length <= 0:
            return 0
    return length


def _nearest_mert_frame(local_seconds: float, num_frames: int, sample_rate: int) -> int:
    continuous = (
        local_seconds * sample_rate - MERT_FIRST_FRAME_CENTER
    ) / MERT_HOP_SAMPLES
    return min(num_frames - 1, max(0, math.floor(continuous + 0.5)))


@dataclass(frozen=True)
class BeatGridWindow:
    """Beat-synchronous chart targets mapped to their nearest MERT frames."""

    frame_indices: torch.Tensor
    ticks: torch.Tensor
    pattern_targets: torch.Tensor
    hold_targets: torch.Tensor
    hold_duration_ticks: torch.Tensor


def _nearest_integer(value: Fraction) -> int:
    """Round a fraction to the nearest integer, with halves away from zero."""

    if value >= 0:
        return math.floor(value + Fraction(1, 2))
    return math.ceil(value - Fraction(1, 2))


def nearest_beat_grid_tick(
    position: Fraction, *, ticks_per_beat: int = BEAT_GRID_TICKS_PER_BEAT
) -> int:
    """Snap a beat to the union of straight 32nd and triplet positions."""

    if ticks_per_beat != BEAT_GRID_TICKS_PER_BEAT:
        return _nearest_integer(position * ticks_per_beat)
    scaled = position * ticks_per_beat
    beat_start = math.floor(scaled / ticks_per_beat) * ticks_per_beat
    candidates = [beat_start + offset for offset in BEAT_GRID_OFFSETS]
    candidates.extend((beat_start - 3, beat_start + ticks_per_beat))
    return min(candidates, key=lambda tick: (abs(scaled - tick), tick))


def encode_beat_grid_window(
    chart: LoadedChart,
    *,
    window_start_seconds: float,
    window_seconds: float,
    num_frames: int,
    sample_rate: int = MERT_SAMPLE_RATE,
    ticks_per_beat: int = BEAT_GRID_TICKS_PER_BEAT,
) -> BeatGridWindow:
    """Build exact beat-grid candidates and targets for one audio window."""

    if window_seconds < 0 or num_frames <= 0 or ticks_per_beat <= 0:
        raise ValueError("window length cannot be negative and grid dimensions must be positive")
    if window_seconds == 0:
        return BeatGridWindow(
            frame_indices=torch.empty(0, dtype=torch.long),
            ticks=torch.empty(0, dtype=torch.long),
            pattern_targets=torch.empty(0, dtype=torch.long),
            hold_targets=torch.empty((0, 4), dtype=torch.bool),
            hold_duration_ticks=torch.empty((0, 4), dtype=torch.float32),
        )
    origin = chart.audio_start_chart_seconds
    chart_start = origin + window_start_seconds
    chart_end = chart_start + window_seconds
    start_position = beat_fraction(chart.timeline.seconds_to_beat(chart_start))
    end_position = beat_fraction(chart.timeline.seconds_to_beat(chart_end))
    first_tick = math.floor(start_position * ticks_per_beat) - 1
    final_tick = math.ceil(end_position * ticks_per_beat) + 1

    ticks: list[int] = []
    frame_indices: list[int] = []
    for tick in range(first_tick, final_tick + 1):
        if (
            ticks_per_beat == BEAT_GRID_TICKS_PER_BEAT
            and tick % ticks_per_beat not in BEAT_GRID_OFFSETS
        ):
            continue
        position = Fraction(tick, ticks_per_beat)
        audio_seconds = chart.timeline.beat_to_seconds(
            fraction_to_beat(position)
        ) - origin
        if not window_start_seconds <= audio_seconds < chart_end - origin:
            continue
        local_seconds = audio_seconds - window_start_seconds
        ticks.append(tick)
        frame_indices.append(
            _nearest_mert_frame(local_seconds, num_frames, sample_rate)
        )

    count = len(ticks)
    pattern_masks = torch.zeros(count, dtype=torch.long)
    hold_targets = torch.zeros((count, 4), dtype=torch.bool)
    hold_duration_ticks = torch.zeros((count, 4), dtype=torch.float32)
    tick_to_index = {tick: index for index, tick in enumerate(ticks)}
    window_end_seconds = window_start_seconds + window_seconds
    for note in chart.notes:
        start_seconds = chart.timeline.beat_to_seconds(note.beat) - origin
        if not window_start_seconds <= start_seconds < window_end_seconds:
            continue
        tick = nearest_beat_grid_tick(
            beat_fraction(note.beat), ticks_per_beat=ticks_per_beat
        )
        index = tick_to_index.get(tick)
        if index is None:
            continue
        pattern_masks[index] |= 1 << note.column
        if note.end_beat is not None:
            hold_targets[index, note.column] = True
            duration = max(
                1,
                _nearest_integer(
                    (beat_fraction(note.end_beat) - beat_fraction(note.beat))
                    * ticks_per_beat
                ),
            )
            hold_duration_ticks[index, note.column] = float(duration)

    return BeatGridWindow(
        frame_indices=torch.tensor(frame_indices, dtype=torch.long),
        ticks=torch.tensor(ticks, dtype=torch.long),
        pattern_targets=pattern_masks - 1,
        hold_targets=hold_targets,
        hold_duration_ticks=hold_duration_ticks,
    )


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            rows.append(value)
    return rows


def write_manifest_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def parse_difficulty_level(version: object) -> float | None:
    """Parse a Malody ``Lv.xx`` value without mistaking ``4K`` for a level."""

    if not isinstance(version, str):
        return None
    match = _LEVEL_PATTERN.search(version)
    if match is None:
        return None
    level = float(match.group(1))
    return level if 0 < level <= MAX_DIFFICULTY_LEVEL else None


def _note_density(row: Mapping[str, Any]) -> float:
    start = float(row.get("first_note_seconds") or 0.0)
    end = float(row.get("last_note_seconds") or 0.0)
    duration = max(1.0, end - start)
    return int(row.get("note_count") or 0) / duration


def annotate_difficulties(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the normalized difficulty control used by the models.

    Explicit Malody levels are preferred.  Missing levels are estimated with a
    least-squares fit from log note density to known levels.
    """

    result = [dict(row) for row in rows]
    if not result:
        return result
    if all("difficulty" in row and "difficulty_level" in row for row in result):
        return result

    known_x = []
    known_y = []
    for row in result:
        level = parse_difficulty_level(row.get("version"))
        if level is not None:
            known_x.append(math.log1p(_note_density(row)))
            known_y.append(level)
    if len(known_x) >= 2 and max(known_x) > min(known_x):
        slope, intercept = np.polyfit(known_x, known_y, 1)
    else:
        slope, intercept = 0.0, 20.0

    for row in result:
        density = _note_density(row)
        parsed_level = parse_difficulty_level(row.get("version"))
        if parsed_level is None:
            level = min(
                MAX_DIFFICULTY_LEVEL,
                max(1.0, float(slope * math.log1p(density) + intercept)),
            )
            source = "density_estimate"
        else:
            level = parsed_level
            source = "version"
        row["note_density"] = density
        row["difficulty_level"] = level
        row["difficulty"] = level / MAX_DIFFICULTY_LEVEL
        row["difficulty_source"] = source
    return result


def probe_audio_metadata(
    rows: Sequence[Mapping[str, Any]], dataset_dir: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Attach decoded audio duration and reject unreadable audio groups."""

    root = Path(dataset_dir)
    by_audio: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_audio[str(row["audio_fingerprint"])].append(dict(row))
    valid = []
    rejected = []
    for fingerprint, candidates in by_audio.items():
        representative = candidates[0]
        try:
            with zipfile.ZipFile(root / str(representative["archive"])) as archive:
                with archive.open(str(representative["audio_member"])) as compressed:
                    with sf.SoundFile(compressed) as audio_file:
                        duration = len(audio_file) / audio_file.samplerate
        except (OSError, RuntimeError, sf.LibsndfileError) as error:
            rejected.append(
                {
                    "audio_fingerprint": fingerprint,
                    "archive": str(representative["archive"]),
                    "audio_member": str(representative["audio_member"]),
                    "error": str(error),
                    "charts": str(len(candidates)),
                }
            )
            continue
        for row in candidates:
            row["audio_duration_seconds"] = duration
            valid.append(row)
    return valid, rejected


def select_training_charts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep every playable chart with one unambiguous full-track audio."""

    ready = [
        dict(row)
        for row in rows
        if row.get("audio_status") == "ready"
        and "jump_effect" not in row.get("warnings", [])
        and row.get("audio_fingerprint")
    ]
    return sorted(
        ready, key=lambda row: (str(row["archive"]), str(row["chart_member"]))
    )


def _split_value(archive: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{archive}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_by_archive(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split rows while keeping every copy of the same audio in one split."""

    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")

    result: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row_value in rows:
        row = dict(row_value)
        group = row.get("audio_fingerprint") or row["archive"]
        value = _split_value(str(group), seed)
        if value < test_fraction:
            split = "test"
        elif value < test_fraction + validation_fraction:
            split = "validation"
        else:
            split = "train"
        result[split].append(row)
    return result


def prepare_splits(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    dataset_dir: str | Path | None = "malody-map",
) -> dict[str, Any]:
    """Select charts, write grouped split manifests, and return a summary."""

    input_rows = read_manifest(manifest_path)
    selected = select_training_charts(input_rows)
    rejected_audio: list[dict[str, str]] = []
    if dataset_dir is not None:
        selected, rejected_audio = probe_audio_metadata(selected, dataset_dir)
    selected = annotate_difficulties(selected)
    splits = split_by_archive(
        selected,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    output = Path(output_dir)
    for name, rows in splits.items():
        write_manifest_rows(rows, output / f"{name}.jsonl")
    summary = {
        "source_charts": len(input_rows),
        "selected_charts": len(selected),
        "seed": seed,
        "rejected_audio": rejected_audio,
        "difficulty": {
            "parsed": sum(row["difficulty_source"] == "version" for row in selected),
            "estimated": sum(
                row["difficulty_source"] == "density_estimate" for row in selected
            ),
            "normalization_level": MAX_DIFFICULTY_LEVEL,
        },
        "splits": {
            name: {
                "charts": len(rows),
                "archives": len({str(row["archive"]) for row in rows}),
                "audio_tracks": len(
                    {
                        str(row.get("audio_fingerprint") or row["archive"])
                        for row in rows
                    }
                ),
                "notes": sum(int(row.get("note_count") or 0) for row in rows),
            }
            for name, rows in splits.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


@dataclass(frozen=True)
class WindowReference:
    row_index: int
    start_seconds: float


class MalodyWindowDataset(Dataset[dict[str, Any]]):
    """Read fixed audio windows and beat-aligned 4K targets from MCZ."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_dir: str | Path,
        *,
        window_seconds: float = 5.0,
        hop_seconds: float = 5.0,
        tail_seconds: float = 0.5,
        sample_rate: int = MERT_SAMPLE_RATE,
        chart_cache_size: int = 16,
        audio_cache_dir: str | Path | None = None,
        audio_cache_size: int = 64,
        mirror_probability: float = 0.0,
    ) -> None:
        if window_seconds <= 0 or hop_seconds <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        if sample_rate != MERT_SAMPLE_RATE:
            raise ValueError(f"MERT-v1-330M requires {MERT_SAMPLE_RATE} Hz audio")
        if not 0 <= mirror_probability <= 1:
            raise ValueError("mirror_probability must be in [0, 1]")
        self.dataset_dir = Path(dataset_dir)
        self.rows = annotate_difficulties(
            select_training_charts(read_manifest(manifest_path))
        )
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.tail_seconds = tail_seconds
        self.sample_rate = sample_rate
        self.num_samples = round(window_seconds * sample_rate)
        self.num_frames = mert_frame_count(self.num_samples)
        self.mirror_probability = mirror_probability
        self.chart_cache_size = chart_cache_size
        self.audio_cache = (
            AudioCache(
                audio_cache_dir,
                sample_rate=sample_rate,
                max_open_files=audio_cache_size,
            )
            if audio_cache_dir is not None
            else None
        )
        self._chart_cache: dict[tuple[str, str], LoadedChart] = {}
        self._chart_cache_order: list[tuple[str, str]] = []

        self.windows: list[WindowReference] = []
        for row_index, row in enumerate(self.rows):
            last_note = max(0.0, float(row.get("last_note_seconds") or 0.0))
            covered_duration = last_note + tail_seconds
            if row.get("audio_duration_seconds") is not None:
                # Keep intros, breaks, and outros so the song planner learns when
                # an empty segment is musically intentional.
                covered_duration = float(row["audio_duration_seconds"])
            covered_duration = max(window_seconds, covered_duration)
            count = max(1, math.ceil((covered_duration - window_seconds) / hop_seconds) + 1)
            for index in range(count):
                self.windows.append(WindowReference(row_index, index * hop_seconds))

    def __len__(self) -> int:
        return len(self.windows)

    def _loaded_chart(self, row: Mapping[str, Any]) -> LoadedChart:
        key = (str(row["archive"]), str(row["chart_member"]))
        if key in self._chart_cache:
            return self._chart_cache[key]
        chart = load_chart(self.dataset_dir / key[0], key[1])
        if self.chart_cache_size > 0:
            self._chart_cache[key] = chart
            self._chart_cache_order.append(key)
            while len(self._chart_cache_order) > self.chart_cache_size:
                oldest = self._chart_cache_order.pop(0)
                self._chart_cache.pop(oldest, None)
        return chart

    def _read_audio_window(
        self, row: Mapping[str, Any], start_seconds: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.audio_cache is not None:
            waveform, valid_samples = self.audio_cache.read_window(
                str(row["audio_fingerprint"]),
                start_sample=round(start_seconds * self.sample_rate),
                num_samples=self.num_samples,
            )
            if waveform.numel() < self.num_samples:
                waveform = torch_functional.pad(
                    waveform, (0, self.num_samples - waveform.numel())
                )
            attention_mask = torch.zeros(self.num_samples, dtype=torch.long)
            attention_mask[:valid_samples] = 1
            return waveform.to(torch.float32), attention_mask

        archive_path = self.dataset_dir / str(row["archive"])
        member = str(row["audio_member"])
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member) as compressed:
                with sf.SoundFile(compressed) as audio_file:
                    source_rate = audio_file.samplerate
                    source_start = max(0, round(start_seconds * source_rate))
                    source_frames = math.ceil(self.window_seconds * source_rate) + 1
                    audio_file.seek(min(source_start, len(audio_file)))
                    array = audio_file.read(
                        source_frames, dtype="float32", always_2d=True
                    )

        waveform = torch.from_numpy(np.asarray(array)).mean(dim=1)
        if source_rate != self.sample_rate and waveform.numel():
            waveform = resample(waveform, source_rate, self.sample_rate)
        valid_samples = min(self.num_samples, waveform.numel())
        waveform = waveform[: self.num_samples]
        if waveform.numel() < self.num_samples:
            waveform = torch_functional.pad(
                waveform, (0, self.num_samples - waveform.numel())
            )
        attention_mask = torch.zeros(self.num_samples, dtype=torch.long)
        attention_mask[:valid_samples] = 1
        return waveform.to(torch.float32), attention_mask

    def _load_reference(
        self, reference: WindowReference, *, mirror: bool | None = None
    ) -> dict[str, Any]:
        row = self.rows[reference.row_index]
        waveform, attention_mask = self._read_audio_window(row, reference.start_seconds)
        loaded_chart = self._loaded_chart(row)
        if mirror is None:
            mirror = bool(
                self.mirror_probability
                and torch.rand(()) < self.mirror_probability
            )
        valid_frames = mert_frame_count(int(attention_mask.sum()))
        beat_grid = encode_beat_grid_window(
            loaded_chart,
            window_start_seconds=reference.start_seconds,
            window_seconds=int(attention_mask.sum()) / self.sample_rate,
            num_frames=max(1, valid_frames),
            sample_rate=self.sample_rate,
        )
        pattern_targets = beat_grid.pattern_targets
        beat_hold_targets = beat_grid.hold_targets
        beat_hold_duration_ticks = beat_grid.hold_duration_ticks
        if mirror:
            mirrored_masks = torch.zeros_like(pattern_targets)
            positive = pattern_targets >= 0
            masks = pattern_targets[positive] + 1
            mirrored = torch.zeros_like(masks)
            for lane in range(4):
                mirrored |= ((masks >> lane) & 1) << (3 - lane)
            mirrored_masks[positive] = mirrored - 1
            mirrored_masks[~positive] = -1
            pattern_targets = mirrored_masks
            beat_hold_targets = beat_hold_targets.flip(dims=(1,))
            beat_hold_duration_ticks = beat_hold_duration_ticks.flip(dims=(1,))
        result = {
            "input_values": waveform,
            "attention_mask": attention_mask,
            "difficulty": torch.tensor(float(row["difficulty"]), dtype=torch.float32),
            "beat_frame_indices": beat_grid.frame_indices,
            "beat_ticks": beat_grid.ticks,
            "beat_pattern_targets": pattern_targets,
            "beat_hold_targets": beat_hold_targets,
            "beat_hold_duration_ticks": beat_hold_duration_ticks,
        }
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._load_reference(self.windows[index])


class MalodySongDataset(MalodyWindowDataset):
    """Return every fixed window of one chart as an ordered song sequence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        windows_by_row: list[list[WindowReference]] = [[] for _ in self.rows]
        for reference in self.windows:
            windows_by_row[reference.row_index].append(reference)
        self.song_windows = [tuple(values) for values in windows_by_row]
        self.window_counts = [len(values) for values in self.song_windows]

    def __len__(self) -> int:
        return len(self.song_windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        mirror = bool(
            self.mirror_probability and torch.rand(()) < self.mirror_probability
        )
        items = [
            self._load_reference(reference, mirror=mirror)
            for reference in self.song_windows[index]
        ]
        tensor_names = (
            "input_values",
            "attention_mask",
            "difficulty",
        )
        result: dict[str, Any] = {
            name: torch.stack([item[name] for item in items]) for name in tensor_names
        }
        beat_tensor_names = (
            "beat_frame_indices",
            "beat_ticks",
            "beat_pattern_targets",
            "beat_hold_targets",
            "beat_hold_duration_ticks",
        ) + (("beat_support_targets",) if "beat_support_targets" in items[0] else ())
        for name in beat_tensor_names:
            result[name] = torch.cat([item[name] for item in items], dim=0)
        result["beat_window_indices"] = torch.cat(
            [
                torch.full(
                    (int(item["beat_ticks"].numel()),),
                    window_index,
                    dtype=torch.long,
                )
                for window_index, item in enumerate(items)
            ]
        )
        return result


def collate_song_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten variable-length song sequences while preserving song boundaries."""

    if not items:
        raise ValueError("cannot collate an empty song batch")
    tensor_names = (
        "input_values",
        "attention_mask",
        "difficulty",
    )
    result: dict[str, Any] = {
        name: torch.cat([item[name] for item in items], dim=0) for name in tensor_names
    }
    beat_tensor_names = (
        "beat_frame_indices",
        "beat_ticks",
        "beat_pattern_targets",
        "beat_hold_targets",
        "beat_hold_duration_ticks",
    ) + (("beat_support_targets",) if "beat_support_targets" in items[0] else ())
    for name in beat_tensor_names:
        result[name] = torch.cat([item[name] for item in items], dim=0)
    window_offsets = []
    offset = 0
    for item in items:
        window_offsets.append(item["beat_window_indices"] + offset)
        offset += int(item["input_values"].shape[0])
    result["beat_window_indices"] = torch.cat(window_offsets)
    result["beat_song_lengths"] = torch.tensor(
        [int(item["beat_ticks"].numel()) for item in items], dtype=torch.long
    )
    return result


class SongWindowBatchSampler:
    """Pack complete songs without exceeding a target number of audio windows."""

    def __init__(
        self,
        window_counts: Sequence[int],
        *,
        max_windows: int,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive")
        if any(count <= 0 for count in window_counts):
            raise ValueError("every song must contain at least one window")
        self.shuffle = shuffle
        self.generator = generator
        self.batches: list[list[int]] = []
        batch_windows: list[int] = []
        for index in sorted(
            range(len(window_counts)), key=lambda value: window_counts[value], reverse=True
        ):
            count = int(window_counts[index])
            destination = next(
                (
                    batch_index
                    for batch_index, used in enumerate(batch_windows)
                    if used + count <= max_windows
                ),
                None,
            )
            if destination is None:
                self.batches.append([index])
                batch_windows.append(count)
            else:
                self.batches[destination].append(index)
                batch_windows[destination] += count

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            order = torch.randperm(len(self.batches), generator=self.generator).tolist()
        else:
            order = range(len(self.batches))
        for batch_index in order:
            yield self.batches[batch_index]
