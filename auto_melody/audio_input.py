"""Wrap an audio file with an explicitly supplied timing grid as a temporary MCZ."""

from __future__ import annotations

import math
from pathlib import Path

from .malody import build_key_chart_from_seconds, pack_mcz


def write_temporary_source_mcz(
    audio: str | Path,
    output: str | Path,
    *,
    bpm: float,
    beat_zero_ms: float,
) -> None:
    """Create the minimal timing-only MCZ consumed by the mainline generator."""

    audio = Path(audio)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("--bpm must be a positive finite number")
    if not math.isfinite(beat_zero_ms):
        raise ValueError("--beat-zero-ms must be finite")
    chart = build_key_chart_from_seconds(
        audio_filename=audio.name,
        notes=(),
        title=audio.stem,
        creator="auto-melody audio input",
        version="4K Audio Source",
        bpm=bpm,
    )
    chart["note"][-1]["offset"] = int(round(-beat_zero_ms))
    pack_mcz(output, chart=chart, audio_path=audio, chart_name="input.mc")
