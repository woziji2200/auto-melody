"""Predecoded 24 kHz mono audio cache for stable training throughput."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample
from tqdm import tqdm

AUDIO_CACHE_VERSION = 2


@dataclass(frozen=True)
class CacheTask:
    fingerprint: str
    archive_name: str
    audio_member: str
    audio_output_name: str
    sample_rate: int
    overwrite: bool
    chart_count: int


def _audio_filename(fingerprint: str) -> str:
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    return f"audio/{digest}.npy"


def _read_manifest_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                rows.append(value)
    return rows


def _save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _build_one_audio(task: CacheTask) -> tuple[str, dict[str, Any]]:
    archive_path = Path(task.archive_name)
    output_path = Path(task.audio_output_name)
    if output_path.is_file() and not task.overwrite:
        cached = np.load(output_path, mmap_mode="r")
        if cached.ndim != 1 or cached.dtype != np.float32:
            raise ValueError(f"invalid cached waveform: {output_path}")
        samples = int(cached.shape[0])
    else:
        torch.set_num_threads(1)
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(task.audio_member) as compressed:
                with sf.SoundFile(compressed) as audio_file:
                    source_rate = audio_file.samplerate
                    array = audio_file.read(dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.asarray(array)).mean(dim=1)
        if source_rate != task.sample_rate and waveform.numel():
            waveform = resample(waveform, source_rate, task.sample_rate)
        values = waveform.to(dtype=torch.float32).contiguous().numpy()
        _save_array(output_path, values)
        samples = int(values.shape[0])

    return task.fingerprint, {
        "file": str(output_path),
        "samples": samples,
        "chart_count": task.chart_count,
        "archive": archive_path.name,
        "audio_member": task.audio_member,
    }


def build_audio_cache(
    manifest_paths: Sequence[str | Path],
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    sample_rate: int = 24_000,
    workers: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Decode every unique manifest audio once and write memory-mapped arrays."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if workers < 0:
        raise ValueError("workers cannot be negative")
    root = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest_rows(manifest_paths)
    unique: dict[str, Mapping[str, Any]] = {}
    chart_counts: dict[str, int] = {}
    for row in rows:
        fingerprint = row.get("audio_fingerprint")
        if (
            row.get("audio_status") == "ready"
            and isinstance(fingerprint, str)
            and fingerprint
        ):
            unique.setdefault(fingerprint, row)
            chart_counts[fingerprint] = chart_counts.get(fingerprint, 0) + 1

    tasks = []
    relative_files = {}
    for fingerprint, row in sorted(unique.items()):
        relative = _audio_filename(fingerprint)
        relative_files[fingerprint] = relative
        tasks.append(
            CacheTask(
                fingerprint=fingerprint,
                archive_name=str(root / str(row["archive"])),
                audio_member=str(row["audio_member"]),
                audio_output_name=str(output / relative),
                sample_rate=sample_rate,
                overwrite=overwrite,
                chart_count=chart_counts[fingerprint],
            )
        )

    entries: dict[str, dict[str, Any]] = {}
    if workers == 0:
        iterator = (_build_one_audio(task) for task in tasks)
        for fingerprint, entry in tqdm(iterator, total=len(tasks), unit="audio"):
            entry["file"] = relative_files[fingerprint]
            entries[fingerprint] = entry
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_build_one_audio, task) for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures), unit="audio"
            ):
                fingerprint, entry = future.result()
                entry["file"] = relative_files[fingerprint]
                entries[fingerprint] = entry

    index = {
        "version": AUDIO_CACHE_VERSION,
        "sample_rate": sample_rate,
        "dtype": "float32",
        "manifest_paths": [str(Path(path)) for path in manifest_paths],
        "audio_count": len(entries),
        "total_samples": sum(int(entry["samples"]) for entry in entries.values()),
        "entries": dict(sorted(entries.items())),
    }
    index_path = output / "index.json"
    temporary_index = output / "index.json.tmp"
    temporary_index.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_index, index_path)
    return {
        "output": str(output),
        "audio_count": len(entries),
        "total_samples": index["total_samples"],
        "audio_size_bytes": sum(
            (output / str(entry["file"])).stat().st_size for entry in entries.values()
        ),
        "sample_rate": sample_rate,
    }


class AudioCache:
    """Small per-worker LRU of memory-mapped cached waveforms."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int,
        max_open_files: int = 64,
    ) -> None:
        self.root = Path(path)
        index_path = self.root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"audio cache index not found: {index_path}; run prepare.py cache-audio first"
            )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("version") != AUDIO_CACHE_VERSION:
            raise ValueError(f"unsupported audio cache version in {index_path}")
        if index.get("sample_rate") != sample_rate or index.get("dtype") != "float32":
            raise ValueError(
                f"audio cache must contain float32 audio at {sample_rate} Hz"
            )
        entries = index.get("entries")
        if not isinstance(entries, dict):
            raise ValueError(f"invalid audio cache entries in {index_path}")
        self.entries = entries
        self.max_open_files = max_open_files
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()

    def _array(self, fingerprint: str) -> np.ndarray:
        existing = self._arrays.pop(fingerprint, None)
        if existing is not None:
            self._arrays[fingerprint] = existing
            return existing
        entry = self.entries.get(fingerprint)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("file"), str):
            raise KeyError(f"audio fingerprint is missing from cache: {fingerprint}")
        array = np.load(self.root / entry["file"], mmap_mode="r")
        if array.ndim != 1 or array.dtype != np.float32:
            raise ValueError(f"invalid cached waveform for {fingerprint}")
        self._arrays[fingerprint] = array
        while len(self._arrays) > self.max_open_files:
            self._arrays.popitem(last=False)
        return array

    def read_window(
        self, fingerprint: str, *, start_sample: int, num_samples: int
    ) -> tuple[torch.Tensor, int]:
        array = self._array(fingerprint)
        start = min(max(0, start_sample), len(array))
        end = min(len(array), start + num_samples)
        waveform = torch.from_numpy(np.array(array[start:end], copy=True))
        return waveform, end - start
