"""Build and query measure-aligned MERT features for human phrases."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from auto_melody.audio_cache import AudioCache
from auto_melody.dataset import (
    MERT_FIRST_FRAME_CENTER,
    MERT_HOP_SAMPLES,
    MERT_SAMPLE_RATE,
    read_manifest,
)
from auto_melody.malody import beat_fraction, fraction_to_beat, load_chart


PHRASE_FEATURE_SCHEMA = "human_phrase_mert_measure_features_v2"
WINDOW_SECONDS = 5.0


def _measure_ranges(chart, duration: float) -> list[tuple[float, float]]:
    timeline = chart.timeline
    origin = chart.audio_start_chart_seconds
    beat = 0
    ranges: list[tuple[float, float]] = []
    while True:
        start = (
            timeline.beat_to_seconds(
                fraction_to_beat(beat_fraction((beat, 0, 1)))
            )
            - origin
        )
        end = (
            timeline.beat_to_seconds(
                fraction_to_beat(beat_fraction((beat + 4, 0, 1)))
            )
            - origin
        )
        if start >= duration and ranges:
            break
        if end > 0:
            ranges.append((max(0.0, start), min(duration, end)))
        beat += 4
        if beat > 20_000:
            raise RuntimeError("chart timeline produced too many measures")
    return ranges or [(0.0, max(duration, 0.01))]


def encode_measure_features(
    waveform: torch.Tensor,
    chart,
    encoder: nn.Module,
    device: torch.device,
    *,
    window_seconds: float = WINDOW_SECONDS,
) -> torch.Tensor:
    """Pool frozen MERT audio and structural context by four-beat measure."""

    if waveform.numel() == 0:
        raise ValueError("cannot encode an empty waveform")
    duration = waveform.numel() / MERT_SAMPLE_RATE
    measure_ranges = _measure_ranges(chart, duration)
    pooled_feature_sums = torch.zeros(
        (len(measure_ranges), int(encoder.config.hidden_size)), dtype=torch.float32
    )
    pooled_frame_counts = torch.zeros((len(measure_ranges), 1), dtype=torch.float32)
    window_samples = round(window_seconds * MERT_SAMPLE_RATE)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    encoder.eval()
    with torch.inference_mode():
        for window_start_sample in range(0, waveform.numel(), window_samples):
            audio_window = waveform[
                window_start_sample : window_start_sample + window_samples
            ]
            valid_samples = int(audio_window.numel())
            if not valid_samples:
                continue
            if valid_samples < window_samples:
                audio_window = torch.nn.functional.pad(
                    audio_window, (0, window_samples - valid_samples)
                )
            audio_window = audio_window.unsqueeze(0).to(device)
            attention_mask = torch.zeros(
                (1, window_samples), dtype=torch.long, device=device
            )
            attention_mask[:, :valid_samples] = 1
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                window_hidden = encoder(
                    input_values=audio_window,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).last_hidden_state[0]
            window_hidden = window_hidden.float().cpu()
            frame_times = (
                window_start_sample
                + MERT_FIRST_FRAME_CENTER
                + torch.arange(window_hidden.shape[0]) * MERT_HOP_SAMPLES
            ) / MERT_SAMPLE_RATE
            for index, (measure_start, measure_end) in enumerate(measure_ranges):
                selected = (frame_times >= measure_start) & (
                    frame_times < measure_end
                )
                if selected.any():
                    pooled_feature_sums[index] += window_hidden[selected].sum(dim=0)
                    pooled_frame_counts[index] += float(selected.sum())

    pooled_mert_features = pooled_feature_sums / pooled_frame_counts.clamp_min(1.0)
    measure_energy = []
    for measure_start, measure_end in measure_ranges:
        start_sample = min(
            waveform.numel() - 1,
            max(0, round(measure_start * MERT_SAMPLE_RATE)),
        )
        end_sample = min(
            waveform.numel(),
            max(start_sample + 1, round(measure_end * MERT_SAMPLE_RATE)),
        )
        measure_energy.append(
            float(waveform[start_sample:end_sample].float().square().mean().sqrt())
        )
    normalized_energy = torch.tensor(measure_energy, dtype=torch.float32).unsqueeze(-1)
    normalized_energy = (normalized_energy - normalized_energy.mean()) / normalized_energy.std().clamp_min(
        1e-5
    )
    measure_features = torch.cat((pooled_mert_features, normalized_energy), dim=-1)
    normalized_mert_features = torch.nn.functional.normalize(
        pooled_mert_features, dim=-1
    )
    prior_measure_similarity = torch.zeros(
        (measure_features.shape[0], 1), dtype=torch.float32
    )
    repeat_distance = torch.zeros_like(prior_measure_similarity)
    for index in range(4, measure_features.shape[0]):
        prior_similarities = (
            normalized_mert_features[: index - 3] @ normalized_mert_features[index]
        )
        closest_prior_index = int(prior_similarities.argmax())
        prior_measure_similarity[index, 0] = prior_similarities[closest_prior_index]
        repeat_distance[index, 0] = math.log1p(index - closest_prior_index)
    return torch.cat(
        (measure_features, prior_measure_similarity, repeat_distance), dim=-1
    ).half()


def build_feature_store(
    manifests: list[str | Path],
    *,
    dataset_dir: str | Path,
    audio_cache_dir: str | Path,
    mert_path: str | Path,
    output_path: str | Path,
    max_audios: int = 0,
    device: str = "cuda",
) -> dict[str, int]:
    rows = []
    for manifest in manifests:
        rows.extend(read_manifest(manifest))
    unique: OrderedDict[str, dict] = OrderedDict()
    for row in rows:
        fingerprint = str(row.get("audio_fingerprint") or "")
        if fingerprint:
            unique.setdefault(fingerprint, row)
    if max_audios > 0:
        unique = OrderedDict(list(unique.items())[:max_audios])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = AudioCache(
        audio_cache_dir, sample_rate=MERT_SAMPLE_RATE, max_open_files=2
    )
    target_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    encoder = AutoModel.from_pretrained(
        str(mert_path),
        trust_remote_code=True,
        local_files_only=True,
    ).to(target_device)
    encoder.requires_grad_(False)
    entries: dict[str, torch.Tensor] = {}
    for number, (fingerprint, row) in enumerate(unique.items(), start=1):
        entry = cache.entries[fingerprint]
        waveform, _ = cache.read_window(
            fingerprint,
            start_sample=0,
            num_samples=int(entry["samples"]),
        )
        chart = load_chart(
            Path(dataset_dir) / str(row["archive"]),
            str(row["chart_member"]),
        )
        entries[fingerprint] = encode_measure_features(
            waveform, chart, encoder, target_device
        ).cpu()
        print(
            f"phrase features {number:4d}/{len(unique):4d} | {fingerprint[:12]}",
            flush=True,
        )
    store = {
        "schema": PHRASE_FEATURE_SCHEMA,
        "feature_dim": int(encoder.config.hidden_size) + 3,
        "entries": entries,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(store, temporary)
    temporary.replace(output)
    return {
        "audios": len(entries),
        "measures": sum(features.shape[0] for features in entries.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one measure-aligned MERT feature store"
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=Path("malody-map"))
    parser.add_argument(
        "--audio-cache-dir", type=Path, default=Path("data/audio-cache-24k")
    )
    parser.add_argument("--mert-path", type=Path, default=Path("data/MERT"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/phrase-features.pt")
    )
    parser.add_argument("--max-audios", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    build_feature_store(
        args.manifests,
        dataset_dir=args.dataset_dir,
        audio_cache_dir=args.audio_cache_dir,
        mert_path=args.mert_path,
        output_path=args.output,
        max_audios=args.max_audios,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
