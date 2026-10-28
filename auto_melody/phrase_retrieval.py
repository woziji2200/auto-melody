"""Music-conditioned retrieval of human lane phrases."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from auto_melody.actions import ACTION_VOCAB_SIZE, action_classes_to_lane_states
from auto_melody.dataset import BEAT_GRID_TICKS_PER_MEASURE
from auto_melody.phrase_features import PHRASE_FEATURE_SCHEMA


@dataclass(frozen=True)
class Phrase:
    audio: torch.Tensor
    signature: tuple[tuple[int, int], ...]
    actions: torch.Tensor
    fingerprint: str
    measure: int


_ACTION_SIGNATURES = tuple(
    (int((state > 0).sum()), int((state == 2).sum()))
    for state in action_classes_to_lane_states(torch.arange(ACTION_VOCAB_SIZE))
)


def _signature(
    actions: torch.Tensor | list[int],
) -> tuple[tuple[int, int], ...]:
    values = actions.tolist() if isinstance(actions, torch.Tensor) else actions
    return tuple(
        _ACTION_SIGNATURES[int(action)] for action in values
    )


class PhraseBank:
    """One-measure human actions indexed by skeleton and MERT audio."""

    def __init__(self, rhythm_index: str | Path, feature_store: str | Path):
        raw = torch.load(rhythm_index, map_location="cpu", weights_only=False)
        feature_data = torch.load(
            feature_store, map_location="cpu", weights_only=False
        )
        if feature_data.get("schema") != PHRASE_FEATURE_SCHEMA:
            raise ValueError(f"{feature_store} is not a phrase feature store")
        feature_entries = {
            fingerprint: torch.nn.functional.normalize(
                features.float(), dim=-1
            )
            for fingerprint, features in feature_data["entries"].items()
        }
        self.phrases: list[Phrase] = []
        self.by_signature: dict[tuple[tuple[int, int], ...], list[int]] = (
            defaultdict(list)
        )
        for record in raw["records"]:
            row = record["row"]
            fingerprint = str(row.get("audio_fingerprint") or "")
            features = feature_entries.get(fingerprint)
            if features is None:
                continue
            grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for event in record["events"].long().tolist():
                grouped[event[0] // BEAT_GRID_TICKS_PER_MEASURE].append(
                    (event[0], event[1])
                )
            for measure, events in grouped.items():
                if measure >= features.shape[0] or not events:
                    continue
                events.sort()
                action_values = [action for _, action in events]
                actions = torch.tensor(action_values, dtype=torch.long)
                signature = _signature(action_values)
                phrase = Phrase(
                    features[measure],
                    signature,
                    actions,
                    fingerprint,
                    measure,
                )
                self.by_signature[signature].append(len(self.phrases))
                self.phrases.append(phrase)
        if not self.phrases:
            raise ValueError("phrase bank is empty")

    def retrieve(
        self, audio: torch.Tensor, signature: tuple[tuple[int, int], ...]
    ) -> int | None:
        candidates = self.by_signature.get(signature)
        if not candidates:
            return None
        query = torch.nn.functional.normalize(audio.float(), dim=0)
        similarities = torch.stack(
            [self.phrases[index].audio for index in candidates]
        ) @ query
        return candidates[int(similarities.argmax())]


def retrieve_phrase_actions(
    bank: PhraseBank,
    ticks: torch.Tensor,
    fallback_actions: torch.Tensor,
    audio_features: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Replace compatible measures without changing skeleton cardinality."""

    if (
        ticks.ndim != 1
        or fallback_actions.shape != ticks.shape
        or audio_features.shape[0] != ticks.numel()
    ):
        raise ValueError("ticks, fallback actions, and features must align")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, tick in enumerate(ticks.tolist()):
        grouped[tick // BEAT_GRID_TICKS_PER_MEASURE].append(index)
    arranged_actions = fallback_actions.clone()
    retrieved = 0
    for positions in grouped.values():
        signature = _signature(fallback_actions[positions])
        candidate = bank.retrieve(audio_features[positions[0]], signature)
        if candidate is None:
            continue
        pattern = bank.phrases[candidate].actions
        if pattern.shape != (len(positions),):
            raise RuntimeError("retrieved phrase does not match its signature")
        arranged_actions[positions] = pattern
        retrieved += 1
    return arranged_actions, retrieved


def measure_consistency_report(
    ticks: torch.Tensor,
    actions: torch.Tensor,
    audio_features: torch.Tensor,
    *,
    similarity_threshold: float = 0.9,
) -> dict[str, float | int]:
    """Measure lane consistency between musically similar measures."""

    if (
        ticks.ndim != 1
        or actions.shape != ticks.shape
        or audio_features.shape[0] != ticks.numel()
    ):
        raise ValueError("ticks, actions, and features must align")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, tick in enumerate(ticks.tolist()):
        grouped[tick // BEAT_GRID_TICKS_PER_MEASURE].append(index)
    measures = [
        (
            positions,
            torch.nn.functional.normalize(
                audio_features[positions[0]].float(), dim=0
            ),
        )
        for positions in grouped.values()
    ]
    pair_count = exact_count = matching_states = state_count = 0
    for left in range(len(measures)):
        left_positions, left_audio = measures[left]
        for right in range(left + 1, len(measures)):
            right_positions, right_audio = measures[right]
            if len(left_positions) != len(right_positions):
                continue
            if float(left_audio @ right_audio) < similarity_threshold:
                continue
            left_actions = actions[left_positions]
            right_actions = actions[right_positions]
            left_states = action_classes_to_lane_states(left_actions)
            right_states = action_classes_to_lane_states(right_actions)
            pair_count += 1
            exact_count += int(torch.equal(left_actions, right_actions))
            matching_states += int((left_states == right_states).sum())
            state_count += int(left_states.numel())
    return {
        "similar_measure_pairs": pair_count,
        "exact_action_consistency": exact_count / max(1, pair_count),
        "lane_state_consistency": matching_states / max(1, state_count),
    }
