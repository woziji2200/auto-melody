#!/usr/bin/env python3
"""Generate a Malody 4K chart directly from an OGG file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from auto_melody.audio_input import write_temporary_source_mcz
from auto_melody.chart_generation import format_generation_report, read_chart_audio
from auto_melody.malody import load_chart
from auto_melody.phrase_generation import generate_phrase_charts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="input OGG audio file")
    parser.add_argument("--levels", type=float, nargs="+", required=True)
    parser.add_argument("--bpm", type=float, required=True, help="manual constant BPM")
    parser.add_argument(
        "--beat-zero-ms",
        type=float,
        required=True,
        help="manual audio timestamp of Malody beat zero in milliseconds",
    )
    parser.add_argument("--mert-path", type=Path, default=Path("data/MERT"))
    parser.add_argument(
        "--salience-checkpoint",
        type=Path,
        default=Path("data/consensus-salience-v1.pt"),
    )
    parser.add_argument(
        "--rhythm-checkpoint",
        type=Path,
        default=Path("data/consensus-rhythm-planner-v1.pt"),
    )
    parser.add_argument(
        "--lane-checkpoint",
        type=Path,
        default=Path("data/lane-arranger-v2.pt"),
    )
    parser.add_argument(
        "--train-index", type=Path, default=Path("data/rhythm-train.pt")
    )
    parser.add_argument(
        "--feature-store", type=Path, default=Path("data/phrase-features.pt")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--author", default="Auto-Melody", help="chart author")
    parser.add_argument("--cover", type=Path, help="optional JPG or PNG cover image")
    parser.add_argument("--rhythm-temperature", type=float, default=0.7)
    parser.add_argument("--lane-temperature", type=float, default=0.7)
    parser.add_argument("--lane-top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.audio.suffix.lower() != ".ogg":
        raise ValueError("the public audio input must be an .ogg file")
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    levels = sorted(set(args.levels))
    if not levels or levels[0] <= 0:
        raise ValueError("--levels must contain positive values")
    level_names = "-".join(f"{level:g}".replace(".", "p") for level in levels)
    output = args.output or Path(
        f"generated/{args.audio.stem}-levels-{level_names}.mcz"
    )
    print(
        f"audio grid | bpm={args.bpm:.3f} | beat-zero={args.beat_zero_ms:.1f}ms",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="auto-melody-audio-") as directory:
        source_archive = Path(directory) / "source.mcz"
        write_temporary_source_mcz(
            args.audio,
            source_archive,
            bpm=args.bpm,
            beat_zero_ms=args.beat_zero_ms,
        )
        source = load_chart(source_archive, "0/input.mc")
        report = generate_phrase_charts(
            source,
            read_chart_audio(source),
            levels,
            output=output,
            mert_path=args.mert_path,
            salience_checkpoint=args.salience_checkpoint,
            rhythm_checkpoint=args.rhythm_checkpoint,
            lane_checkpoint=args.lane_checkpoint,
            train_index=args.train_index,
            feature_store=args.feature_store,
            rhythm_temperature=args.rhythm_temperature,
            lane_temperature=args.lane_temperature,
            lane_top_p=args.lane_top_p,
            seed=args.seed,
            author=args.author,
            cover=args.cover,
        )
        report["source"] = {
            "audio": str(args.audio),
            "bpm": args.bpm,
            "beat_zero_ms": args.beat_zero_ms,
        }
        report_path = output.with_suffix(".json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            format_generation_report(
                report,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
