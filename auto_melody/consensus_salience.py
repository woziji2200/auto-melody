"""Difficulty-aware, same-song consensus onset encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import melscale_fbanks
from transformers import AutoModel

from auto_melody.dataset import BEAT_GRID_TICKS_PER_MEASURE
from auto_melody.global_rhythm import TemporalBlock, _sequence_mask, _state_dict_without


CONSENSUS_SALIENCE_SCHEMA = "malody_4k_consensus_salience_v1"
MERT_WINDOW_CHUNK = 16


class OnsetSalienceEncoder(nn.Module):
    """Score audible note opportunities without predicting chart actions."""

    def __init__(
        self,
        mert_path: str | Path = "data/MERT",
        *,
        decoder_dim: int = 512,
        local_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if decoder_dim < 128 or decoder_dim % 8 or local_layers <= 0:
            raise ValueError("decoder dimensions and layer count must be valid")
        self.mert = AutoModel.from_pretrained(
            str(mert_path), trust_remote_code=True, local_files_only=True
        )
        self.mert.requires_grad_(False)
        self.mert.eval()
        hidden_size = int(self.mert.config.hidden_size)
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, decoder_dim), nn.GELU()
        )
        self.register_buffer("stft_window", torch.hann_window(1024), persistent=False)
        self.register_buffer(
            "mel_filter",
            melscale_fbanks(
                1024 // 2 + 1,
                f_min=30.0,
                f_max=12_000.0,
                n_mels=80,
                sample_rate=24_000,
                norm="slaney",
                mel_scale="slaney",
            ),
            persistent=False,
        )
        rhythm_dim = max(64, decoder_dim // 4)
        self.rhythm_encoder = nn.Sequential(
            nn.Conv1d(160, rhythm_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(rhythm_dim, rhythm_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.rhythm_projection = nn.Linear(rhythm_dim, decoder_dim, bias=False)
        self.phase_embedding = nn.Embedding(BEAT_GRID_TICKS_PER_MEASURE, decoder_dim)
        self.difficulty_projection = nn.Sequential(
            nn.Linear(1, decoder_dim // 4),
            nn.SiLU(),
            nn.Linear(decoder_dim // 4, decoder_dim),
        )
        self.local_blocks = nn.ModuleList(
            TemporalBlock(decoder_dim, 2**index, dropout)
            for index in range(local_layers)
        )
        self.placement_head = nn.Sequential(
            nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, 1)
        )

    def train(self, mode: bool = True) -> "OnsetSalienceEncoder":
        super().train(mode)
        self.mert.eval()
        return self

    def decoder_state_dict(self) -> dict[str, torch.Tensor]:
        return _state_dict_without(self, "mert.")

    def _audio_feature_mask(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self.mert, "_get_feature_vector_attention_mask"):
            return self.mert._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            ).bool()
        lengths = attention_mask.sum(dim=1).float()
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        return positions < lengths.unsqueeze(1) * hidden.shape[1] / attention_mask.shape[1]

    def _transient_features(self, audio_windows: torch.Tensor, output_frames: int) -> torch.Tensor:
        waveform = audio_windows.float()
        if waveform.shape[1] < 1024:
            waveform = torch.nn.functional.pad(waveform, (0, 1024 - waveform.shape[1]))
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.stft(
                waveform,
                n_fft=1024,
                hop_length=320,
                win_length=1024,
                window=self.stft_window.float(),
                center=False,
                return_complex=True,
            ).abs().square()
            mel = torch.einsum("bft,fm->bmt", spectrum, self.mel_filter.float())
            log_mel = torch.log1p(mel)
            flux = torch.nn.functional.pad(
                torch.relu(log_mel[:, :, 1:] - log_mel[:, :, :-1]), (1, 0)
            )
            features = torch.cat((log_mel, flux), dim=1)
        encoded = self.rhythm_encoder(features)
        if encoded.shape[-1] != output_frames:
            encoded = torch.nn.functional.interpolate(
                encoded, size=output_frames, mode="linear", align_corners=False
            )
        return self.rhythm_projection(encoded.transpose(1, 2))

    def _encode_audio_windows(
        self, audio_windows: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memories = []
        masks = []
        for start in range(0, audio_windows.shape[0], MERT_WINDOW_CHUNK):
            end = min(audio_windows.shape[0], start + MERT_WINDOW_CHUNK)
            values = audio_windows[start:end]
            mask = attention_mask[start:end]
            with torch.no_grad():
                hidden = self.mert(
                    input_values=values, attention_mask=mask, return_dict=True
                ).last_hidden_state
            memories.append(
                self.audio_projection(hidden) + self._transient_features(values, hidden.shape[1])
            )
            masks.append(self._audio_feature_mask(hidden, mask))
        return torch.cat(memories), torch.cat(masks)

    def encode_beat_candidates(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        difficulty: torch.Tensor,
        beat_window_indices: torch.Tensor,
        beat_frame_indices: torch.Tensor,
        beat_ticks: torch.Tensor,
        beat_song_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return frozen tick features that support the salience score."""
        window_hidden, feature_mask = self._encode_audio_windows(input_values, attention_mask)
        if beat_ticks.numel() == 0 or int(beat_song_lengths.sum()) != int(beat_ticks.numel()):
            raise ValueError("candidate lengths must exactly describe beat ticks")
        if not bool(feature_mask[beat_window_indices, beat_frame_indices].all()):
            raise ValueError("candidate refers to a padded audio feature")
        candidate_features = window_hidden[beat_window_indices, beat_frame_indices]
        candidate_features = candidate_features + self.phase_embedding(beat_ticks.remainder(BEAT_GRID_TICKS_PER_MEASURE))
        candidate_features = candidate_features + self.difficulty_projection(
            difficulty[beat_window_indices].unsqueeze(-1).to(candidate_features.dtype)
        )
        lengths = [int(value) for value in beat_song_lengths.detach().cpu().tolist()]
        candidate_features = pad_sequence(list(candidate_features.split(lengths)), batch_first=True)
        ticks = pad_sequence(list(beat_ticks.split(lengths)), batch_first=True)
        candidate_mask = _sequence_mask(beat_song_lengths, candidate_features.shape[1])
        for block in self.local_blocks:
            candidate_features = block(candidate_features, candidate_mask)
        return candidate_features, ticks, candidate_mask

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        difficulty: torch.Tensor,
        beat_window_indices: torch.Tensor,
        beat_frame_indices: torch.Tensor,
        beat_ticks: torch.Tensor,
        beat_song_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_features, ticks, candidate_mask = self.encode_beat_candidates(
            input_values,
            attention_mask,
            difficulty,
            beat_window_indices,
            beat_frame_indices,
            beat_ticks,
            beat_song_lengths,
        )
        return self.placement_head(candidate_features).squeeze(-1), ticks, candidate_mask
