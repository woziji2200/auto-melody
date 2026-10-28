"""A global chart planner guided by a frozen consensus-onset encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from auto_melody.actions import RHYTHM_TYPE_COUNT
from auto_melody.consensus_salience import OnsetSalienceEncoder
from auto_melody.global_rhythm import (
    RhythmPlanOutput,
    TemporalBlock,
    _load_state_dict_without,
    _measure_context,
    _sequence_mask,
    _state_dict_without,
)


GUIDED_RHYTHM_SCHEMA = "malody_4k_consensus_guided_rhythm_v1"


class SalienceGuidedRhythmPlanner(nn.Module):
    """Convert independently trained musical salience into a chart skeleton."""

    def __init__(
        self,
        salience_checkpoint: str | Path,
        *,
        mert_path: str | Path = "data/MERT",
        decoder_dim: int = 512,
        refinement_layers: int = 2,
        measure_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        checkpoint = torch.load(salience_checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        self.salience = OnsetSalienceEncoder(
            mert_path,
            decoder_dim=int(config["decoder_dim"]),
            local_layers=int(config["local_layers"]),
            dropout=float(config["dropout"]),
        )
        _load_state_dict_without(
            self.salience, checkpoint["decoder"], prefixes=("mert.",), label="salience"
        )
        self.salience.requires_grad_(False)
        self.salience.eval()
        self.salience_dim = int(config["decoder_dim"])
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.salience_dim), nn.Linear(self.salience_dim, decoder_dim), nn.GELU()
        )
        self.salience_projection = nn.Linear(1, decoder_dim, bias=False)
        self.refinement = nn.ModuleList(
            TemporalBlock(decoder_dim, 2**index, dropout) for index in range(refinement_layers)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=8,
            dim_feedforward=decoder_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.measure_encoder = nn.TransformerEncoder(
            layer,
            num_layers=measure_layers,
            norm=nn.LayerNorm(decoder_dim),
            enable_nested_tensor=False,
        )
        self.placement_head = nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, 1))
        self.count_head = nn.Sequential(
            nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim // 2), nn.GELU(), nn.Linear(decoder_dim // 2, 1)
        )
        self.type_head = nn.Linear(decoder_dim, RHYTHM_TYPE_COUNT - 1)
        self.duration_head = nn.Linear(decoder_dim, 4)

    def train(self, mode: bool = True) -> "SalienceGuidedRhythmPlanner":
        super().train(mode)
        self.salience.eval()
        return self

    def decoder_state_dict(self) -> dict[str, torch.Tensor]:
        return _state_dict_without(self, "salience.")

    def load_decoder_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        _load_state_dict_without(
            self, state, prefixes=("salience.",), label="guided rhythm"
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        difficulty: torch.Tensor,
        beat_window_indices: torch.Tensor,
        beat_frame_indices: torch.Tensor,
        beat_ticks: torch.Tensor,
        beat_song_lengths: torch.Tensor,
    ) -> RhythmPlanOutput:
        with torch.no_grad():
            encoded_candidates = self.salience.encode_beat_candidates(
                input_values,
                attention_mask,
                difficulty,
                beat_window_indices,
                beat_frame_indices,
                beat_ticks,
                beat_song_lengths,
            )
            salience_features, ticks, candidate_mask = encoded_candidates
            salience_scores = self.salience.placement_head(salience_features).squeeze(-1)
        candidate_features = self.adapter(salience_features) + self.salience_projection(torch.sigmoid(salience_scores).unsqueeze(-1))
        candidate_features = candidate_features * candidate_mask.unsqueeze(-1)
        for block in self.refinement:
            candidate_features = block(candidate_features, candidate_mask)
        broadcast, summaries, measure_indices, measure_mask = _measure_context(
            candidate_features, ticks, candidate_mask, measure_encoder=self.measure_encoder
        )
        context_features = (candidate_features + broadcast) * candidate_mask.unsqueeze(-1)
        placement_scores = self.placement_head(context_features).squeeze(-1)
        return RhythmPlanOutput(
            placement_scores=placement_scores,
            measure_log_counts=torch.nn.functional.softplus(self.count_head(summaries).squeeze(-1)),
            type_logits=self.type_head(context_features),
            duration_logs=torch.nn.functional.softplus(self.duration_head(context_features)),
            ticks=ticks,
            candidate_mask=candidate_mask,
            measure_indices=measure_indices,
            measure_mask=measure_mask,
        )
