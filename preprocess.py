#!/usr/bin/env python3
"""Build all local training artifacts from a directory of user-owned MCZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_melody.audio_cache import build_audio_cache
from auto_melody.consensus_index import build_sequence_index
from auto_melody.dataset import prepare_splits
from auto_melody.malody import write_manifest
from auto_melody.phrase_features import build_feature_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="directory containing user MCZ files")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--mert-path", type=Path, default=Path("data/MERT"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device used to extract MERT phrase features",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            parser.error("--device cuda was requested, but CUDA is unavailable")
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = args.output_dir / "manifests" / "all-4k.jsonl"
    splits = args.output_dir / "splits-difficulty"
    cache = args.output_dir / "audio-cache-24k"
    features = args.output_dir / "phrase-features.pt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    counts = write_manifest(args.dataset, manifest)
    summary = prepare_splits(
        manifest,
        splits,
        seed=args.seed,
        dataset_dir=args.dataset,
    )
    split_paths = [splits / f"{name}.jsonl" for name in ("train", "validation", "test")]
    build_audio_cache(
        split_paths,
        dataset_dir=args.dataset,
        output_dir=cache,
        workers=args.workers,
    )
    for name in ("train", "validation"):
        build_sequence_index(
            splits / f"{name}.jsonl",
            args.dataset,
            args.output_dir / f"rhythm-{name}.pt",
        )
    phrase_summary = build_feature_store(
        [splits / "train.jsonl"],
        dataset_dir=args.dataset,
        audio_cache_dir=cache,
        mert_path=args.mert_path,
        output_path=features,
        device=device,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "scan": dict(counts),
                "splits": summary["splits"],
                "phrase_features": phrase_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
