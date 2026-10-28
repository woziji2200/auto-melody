"""Generate a consensus rhythm chart with retrieved human lane phrases."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath
import sys
import tempfile

import torch

from auto_melody.actions import action_classes_to_lane_states
from auto_melody.chart_generation import (
    PredictedNote,
    build_generated_chart,
    chart_quality_summary,
)
from auto_melody.consensus_global_generation import build_inference_batch, decode_rhythm_plan
from auto_melody.consensus_rhythm import (
    GUIDED_RHYTHM_SCHEMA,
    SalienceGuidedRhythmPlanner,
)
from auto_melody.consensus_index import extract_chart_events
from auto_melody.lane_arranger import LANE_ARRANGER_SCHEMA, LaneArranger
from auto_melody.malody import (
    IMAGE_SUFFIXES,
    beat_fraction,
    load_chart,
    pack_mcz_charts,
)
from auto_melody.phrase_features import encode_measure_features
from auto_melody.phrase_retrieval import (
    PhraseBank,
    retrieve_phrase_actions,
    measure_consistency_report,
)


PHRASE_CHART_SCHEMA = "consensus_rhythm_phrase_chart_v1"


def _difficulty_name(level: float) -> str:
    return f"{level:g}".replace(".", "p")


def _generated_chart_name(level: float) -> str:
    return f"phrase-retrieval-lv{_difficulty_name(level)}.mc"


def _load_rhythm_model(
    path: Path,
    device: torch.device,
    mert_path: Path,
    salience_checkpoint: Path | None = None,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != GUIDED_RHYTHM_SCHEMA:
        raise ValueError(f"{path} is not a consensus rhythm planner checkpoint")
    config = checkpoint["config"]
    model = SalienceGuidedRhythmPlanner(
        salience_checkpoint or Path(config["salience_checkpoint"]),
        mert_path=mert_path,
        decoder_dim=int(config["decoder_dim"]),
        refinement_layers=int(config["refinement_layers"]),
        measure_layers=int(config["measure_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_decoder_state_dict(checkpoint["decoder"])
    return model.eval(), checkpoint


def _load_lane_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != LANE_ARRANGER_SCHEMA:
        raise ValueError(f"{path} is not a lane arranger checkpoint")
    config = checkpoint["config"]
    model = LaneArranger(
        decoder_dim=int(config["decoder_dim"]),
        decoder_layers=int(config["decoder_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def _chart_event_tensors(chart):
    events = extract_chart_events(chart)
    ticks = torch.tensor([event[0] for event in events], dtype=torch.long)
    actions = torch.tensor([event[1] for event in events], dtype=torch.long)
    return events, ticks, action_classes_to_lane_states(actions)


def _extract_event_groups(source):
    events, ticks, lane_states = _chart_event_tensors(source)
    event_groups_by_beat = defaultdict(list)
    for note in source.notes:
        event_groups_by_beat[beat_fraction(note.beat)].append(note)
    event_groups = [
        event_groups_by_beat[position] for position in sorted(event_groups_by_beat)
    ]
    if len(events) != len(event_groups):
        raise RuntimeError("chart event groups do not match source notes")
    hold_duration_ticks = torch.zeros((len(events), 4), dtype=torch.long)
    for index, _event_group in enumerate(event_groups):
        event_hold_durations = sorted(
            int(value) for value in events[index][2] if value > 0
        )
        hold_duration_ticks[index, : len(event_hold_durations)] = torch.tensor(
            event_hold_durations, dtype=torch.long
        )
    return event_groups, ticks, lane_states, hold_duration_ticks


def _apply_lane_actions(source, ordered_groups, actions: torch.Tensor):
    states = action_classes_to_lane_states(actions)
    rewritten: list[PredictedNote] = []
    origin = source.audio_start_chart_seconds
    for group, lane_states in zip(ordered_groups, states.tolist()):
        start_chart_seconds = source.timeline.beat_to_seconds(group[0].beat)
        start = start_chart_seconds - origin
        hold_durations = sorted(
            source.timeline.beat_to_seconds(note.end_beat) - start_chart_seconds
            for note in group
            if note.end_beat is not None
        )
        hold_index = 0
        for lane, state in enumerate(lane_states):
            if state == 0:
                continue
            end = None
            if state == 2:
                end = start + hold_durations[hold_index]
                hold_index += 1
            rewritten.append(PredictedNote(start, lane, end))
        if hold_index != len(hold_durations):
            raise RuntimeError("retrieved action changed Hold cardinality")
    return rewritten


def _write_mcz_archive(
    output: Path,
    source,
    charts: dict[str, dict],
    *,
    cover: Path | None,
) -> None:
    audio_name = PurePosixPath(
        source.primary_audio_event.member or "audio.ogg"
    ).name
    with tempfile.TemporaryDirectory(prefix="auto-melody-phrase-") as directory:
        audio_path = Path(directory) / audio_name
        audio_path.write_bytes(source.read_audio_bytes())
        pack_mcz_charts(
            output,
            charts=charts,
            audio_path=audio_path,
            cover_path=cover,
        )


def _count_chart_events(chart):
    _, ticks, states = _chart_event_tensors(chart)
    return ticks, (states > 0).sum(dim=-1), (states == 2).sum(dim=-1)


def generate_phrase_charts(
    source,
    waveform: torch.Tensor,
    levels: list[float],
    *,
    output: Path,
    mert_path: Path,
    salience_checkpoint: Path | None = None,
    rhythm_checkpoint: Path,
    lane_checkpoint: Path,
    train_index: Path,
    feature_store: Path,
    rhythm_temperature: float = 0.7,
    lane_temperature: float = 0.7,
    lane_top_p: float = 0.9,
    seed: int = 42,
    author: str = "Auto-Melody",
    cover: Path | None = None,
) -> dict:
    levels = sorted(set(levels))
    if not levels or levels[0] <= 0:
        raise ValueError("levels must contain positive values")
    if rhythm_temperature <= 0 or lane_temperature <= 0:
        raise ValueError("temperatures must be positive")
    if not 0 < lane_top_p <= 1:
        raise ValueError("lane_top_p must be in (0, 1]")
    author = author.strip()
    if not author:
        raise ValueError("author must not be empty")
    if cover is not None:
        cover = Path(cover)
        if not cover.is_file():
            raise FileNotFoundError(cover)
        if cover.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("cover must be a JPG or PNG image")
    for path in (
        salience_checkpoint,
        rhythm_checkpoint,
        lane_checkpoint,
        train_index,
        feature_store,
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    # Load every immutable dependency once before generating all requested levels.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rhythm_model, rhythm_checkpoint_state = _load_rhythm_model(
        rhythm_checkpoint, device, mert_path, salience_checkpoint
    )
    lane_model, lane_checkpoint_state = _load_lane_model(lane_checkpoint, device)
    phrase_bank = PhraseBank(train_index, feature_store)
    audio_measure_features = encode_measure_features(
        waveform, source, rhythm_model.salience.mert, device
    ).float()
    base_inference_batch = build_inference_batch(source, waveform, levels[0])

    # First produce lane-invariant rhythm skeletons for every difficulty.
    skeleton_chart_data = {}
    for index, level in enumerate(levels):
        print(
            f"generating level {level:g} | {index + 1}/{len(levels)}",
            file=sys.stderr,
            flush=True,
        )
        inference_batch = {
            **base_inference_batch,
            "difficulty": torch.full_like(
                base_inference_batch["difficulty"],
                max(0.0, min(1.0, level / 40.0)),
            ),
        }
        skeleton_notes = decode_rhythm_plan(
            rhythm_model,
            inference_batch,
            source,
            device,
            rhythm_temperature,
            seed=seed + index,
        )
        skeleton_chart_data[f"skeleton-lv{_difficulty_name(level)}.mc"] = (
            build_generated_chart(
                source,
                skeleton_notes,
                difficulty_level=level,
                creator="auto-melody consensus rhythm skeleton",
            )
        )

    # Reload serialized skeletons so lane arrangement uses the final quantized notes.
    with tempfile.TemporaryDirectory(
        prefix="auto-melody-consensus-skeleton-"
    ) as directory:
        skeleton_archive = Path(directory) / "skeleton.mcz"
        audio_name = PurePosixPath(
            source.primary_audio_event.member or "audio.ogg"
        ).name
        audio_path = Path(directory) / audio_name
        audio_path.write_bytes(source.read_audio_bytes())
        pack_mcz_charts(
            skeleton_archive,
            charts=skeleton_chart_data,
            audio_path=audio_path,
        )
        final_chart_data = {}
        verification_records = []
        for index, level in enumerate(levels):
            skeleton_name = f"skeleton-lv{_difficulty_name(level)}.mc"
            skeleton = load_chart(skeleton_archive, f"0/{skeleton_name}")
            event_groups, ticks, lane_states, skeleton_hold_durations = _extract_event_groups(skeleton)
            candidate_measure_features = audio_measure_features[
                (ticks // 96).clamp_max(audio_measure_features.shape[0] - 1)
            ]
            press_counts = (lane_states > 0).sum(dim=-1).to(device)
            hold_counts = (lane_states == 2).sum(dim=-1).to(device)
            fallback_actions = lane_model.sample_actions(
                ticks.to(device),
                press_counts,
                hold_counts,
                difficulty=max(0.0, min(1.0, level / 40.0)),
                hold_duration_ticks=skeleton_hold_durations.to(device),
                temperature=lane_temperature,
                top_p=lane_top_p,
                seed=seed + index,
            ).cpu()
            lane_actions, retrieved_measures = retrieve_phrase_actions(
                phrase_bank, ticks, fallback_actions, candidate_measure_features
            )
            consistency_report = {
                "lane_fallback": measure_consistency_report(
                    ticks, fallback_actions, candidate_measure_features
                ),
                "phrase_retrieval": measure_consistency_report(
                    ticks, lane_actions, candidate_measure_features
                ),
            }
            chart_name = _generated_chart_name(level)
            final_chart_data[chart_name] = build_generated_chart(
                source,
                _apply_lane_actions(skeleton, event_groups, lane_actions),
                difficulty_level=level,
                creator=author,
                background=cover.name if cover is not None else "",
            )
            verification_records.append(
                {
                    "level": level,
                    "chart_name": chart_name,
                    "ticks": ticks,
                    "press_counts": press_counts.cpu(),
                    "hold_counts": hold_counts.cpu(),
                    "retrieved_measures": retrieved_measures,
                    "total_measures": int(audio_measure_features.shape[0]),
                    "consistency": consistency_report,
                }
            )

    _write_mcz_archive(output, source, final_chart_data, cover=cover)

    # Verify serialization preserved the skeleton before reporting the result.
    generated_chart_reports = []
    for item in verification_records:
        member = f"0/{item['chart_name']}"
        generated = load_chart(output, member)
        actual_ticks, actual_press, actual_hold = _count_chart_events(generated)
        checks = {
            "ticks": torch.equal(actual_ticks, item["ticks"]),
            "press_counts": torch.equal(
                actual_press, item["press_counts"]
            ),
            "hold_counts": torch.equal(actual_hold, item["hold_counts"]),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"level {item['level']:g} changed the rhythm skeleton: {checks}"
            )
        generated_chart_reports.append(
            {
                "level": item["level"],
                "member": member,
                "retrieved_measures": item["retrieved_measures"],
                "total_measures": item["total_measures"],
                "similar_measure_consistency": item["consistency"],
                "skeleton": {"events": int(item["ticks"].numel()), **checks},
                "quality": chart_quality_summary(
                    source,
                    generated.timed_notes(),
                    duration_seconds=waveform.numel() / 24_000,
                ),
            }
        )

    report = {
        "schema": PHRASE_CHART_SCHEMA,
        "levels": levels,
        "metadata": {
            "author": author,
            "cover": str(cover) if cover is not None else None,
        },
        "rhythm": {
            "checkpoint": str(rhythm_checkpoint),
            "step": int(rhythm_checkpoint_state["step"]),
        },
        "lane_fallback": {
            "checkpoint": str(lane_checkpoint),
            "step": int(lane_checkpoint_state["step"]),
        },
        "retrieval": {
            "phrase_bank": len(phrase_bank.phrases),
            "feature_store": str(feature_store),
        },
        "charts": generated_chart_reports,
        "output": str(output),
    }
    return report
