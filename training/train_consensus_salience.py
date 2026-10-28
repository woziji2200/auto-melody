"""Train the consensus salience stage from random initialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional

from auto_melody.actions import action_classes_from_pattern_and_holds, action_to_rhythm_type
from auto_melody.dataset import collate_song_batch
from auto_melody.consensus_salience import CONSENSUS_SALIENCE_SCHEMA, OnsetSalienceEncoder
from auto_melody.consensus_dataset import OnsetSupportSongDataset
from auto_melody.consensus_training import (
    append_jsonl,
    cycle_data_batches,
    build_data_loader,
    build_song_batch_sampler,
    move_batch_tensors,
    pad_flattened_song_values,
    save_checkpoint,
    seed_everything,
    select_training_device,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train consensus-salience-v1")
    parser.add_argument("--train-manifest", type=Path, default=Path("data/splits-difficulty/train.jsonl"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/splits-difficulty/validation.jsonl"))
    parser.add_argument("--train-index", type=Path, default=Path("data/rhythm-train.pt"))
    parser.add_argument("--validation-index", type=Path, default=Path("data/rhythm-validation.pt"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("malody-map"))
    parser.add_argument("--audio-cache-dir", type=Path, default=Path("data/audio-cache-24k"))
    parser.add_argument("--mert-path", default="data/MERT")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/consensus-salience-v1"))
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--local-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--support-scale-levels", type=float, default=4.0)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=3407)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def build_chart_onset_mask(batch: dict[str, torch.Tensor], width: int) -> torch.Tensor:
    actions = action_classes_from_pattern_and_holds(batch["beat_pattern_targets"], batch["beat_hold_targets"])
    return pad_flattened_song_values((action_to_rhythm_type(actions) > 0).to(torch.float32), batch["beat_song_lengths"], width)


def select_by_chart_onset_counts(scores, ticks, mask, chart_onset_mask) -> torch.Tensor:
    prediction = torch.zeros_like(mask)
    for song in range(mask.shape[0]):
        measures = torch.unique(torch.div(ticks[song, mask[song]], 96, rounding_mode="floor"))
        for measure in measures.tolist():
            local = mask[song] & (torch.div(ticks[song], 96, rounding_mode="floor") == measure)
            count = int(chart_onset_mask[song, local].sum())
            if count:
                positions = torch.nonzero(local).flatten()
                prediction[song, positions[scores[song, positions].topk(count).indices]] = True
    return prediction


def salience_loss_metrics(scores, ticks, mask, batch):
    onset_support = pad_flattened_song_values(batch["beat_support_targets"], batch["beat_song_lengths"], scores.shape[1])
    chart_onset_mask = build_chart_onset_mask(batch, scores.shape[1])
    positive = onset_support[mask].sum().clamp_min(1)
    negative = (1 - onset_support[mask]).sum().clamp_min(1)
    pos_weight = (negative / positive).sqrt().clamp(1, 8)
    loss = functional.binary_cross_entropy_with_logits(scores[mask], onset_support[mask], pos_weight=pos_weight)
    selected = select_by_chart_onset_counts(scores.detach(), ticks, mask, chart_onset_mask)
    tp = (selected & (chart_onset_mask > 0) & mask).sum().float()
    fp = (selected & ~(chart_onset_mask > 0) & mask).sum().float()
    fn = (~selected & (chart_onset_mask > 0) & mask).sum().float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    return loss, {"loss": float(loss.detach()), "hard_count_selection_f1": float(f1), "positive_weight": float(pos_weight.detach())}


@torch.inference_mode()
def validate(model, loader, device, amp_dtype):
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    for original in loader:
        batch = move_batch_tensors(original, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            scores, ticks, mask = model(
                batch["input_values"],
                batch["attention_mask"],
                batch["difficulty"],
                batch["beat_window_indices"],
                batch["beat_frame_indices"],
                batch["beat_ticks"],
                batch["beat_song_lengths"],
            )
            _, metrics = salience_loss_metrics(scores, ticks, mask, batch)
        weight = int(batch["beat_song_lengths"].numel())
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * weight
        seen += weight
    model.train()
    return {f"validation_{name}": value / max(seen, 1) for name, value in totals.items()}


def main() -> int:
    args = build_parser().parse_args()
    if (
        min(args.max_windows, args.max_steps, args.eval_steps, args.local_layers)
        <= 0
        or args.support_scale_levels <= 0
    ):
        raise ValueError("invalid training limits")
    seed_everything(args.seed)
    device, amp_dtype = select_training_device()
    train_data = OnsetSupportSongDataset(
        args.train_manifest,
        args.dataset_dir,
        audio_cache_dir=args.audio_cache_dir,
        mirror_probability=0.5,
        rhythm_index=args.train_index,
        support_scale_levels=args.support_scale_levels,
    )
    validation_data = OnsetSupportSongDataset(
        args.validation_manifest,
        args.dataset_dir,
        audio_cache_dir=args.audio_cache_dir,
        mirror_probability=0.0,
        rhythm_index=args.validation_index,
        support_scale_levels=args.support_scale_levels,
    )
    train_loader = build_data_loader(
        train_data,
        batch_sampler=build_song_batch_sampler(
            train_data.window_counts,
            max_windows=args.max_windows,
            shuffle=True,
            seed=args.seed,
        ),
        workers=args.num_workers,
        collate_fn=collate_song_batch,
    )
    validation_loader = build_data_loader(
        validation_data,
        batch_sampler=build_song_batch_sampler(
            validation_data.window_counts,
            max_windows=args.max_windows,
            shuffle=False,
            seed=args.seed,
        ),
        workers=args.num_workers,
        collate_fn=collate_song_batch,
    )
    model = OnsetSalienceEncoder(
        args.mert_path,
        decoder_dim=args.decoder_dim,
        local_layers=args.local_layers,
        dropout=args.dropout,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.min_learning_rate)
    write_json(args.output_dir / "config.json", vars(args))
    print(
        "consensus-salience training ready\n"
        f"  songs: {len(train_data):,} train / {len(validation_data):,} validation\n"
        f"  trainable parameters: {sum(parameter.numel() for parameter in parameters):,}",
        flush=True,
    )
    batches = cycle_data_batches(train_loader)
    best = -1.0
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        batch = move_batch_tensors(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            scores, ticks, mask = model(
                batch["input_values"],
                batch["attention_mask"],
                batch["difficulty"],
                batch["beat_window_indices"],
                batch["beat_frame_indices"],
                batch["beat_ticks"],
                batch["beat_song_lengths"],
            )
            loss, metrics = salience_loss_metrics(scores, ticks, mask, batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_steps == 0 or step == 1:
            print(
                f"step {step:>4}/{args.max_steps} | loss {metrics['loss']:.4f} "
                f"| hard-f1 {metrics['hard_count_selection_f1']:.3f} "
                f"| positive-weight {metrics['positive_weight']:.2f} "
                f"| grad {float(gradient_norm):.2f} "
                f"| {(time.perf_counter() - started) / step:.2f}s/step",
                flush=True,
            )
        if step % args.eval_steps == 0 or step == args.max_steps:
            metrics = validate(model, validation_loader, device, amp_dtype)
            metrics.update(step=step, learning_rate=optimizer.param_groups[0]["lr"])
            append_jsonl(args.output_dir / "metrics.jsonl", metrics)
            state = {
                "schema": CONSENSUS_SALIENCE_SCHEMA,
                "model_id": "consensus-salience-v1",
                "initialization": "random",
                "parent_checkpoint": None,
                "step": step,
                "config": {
                    "mert_path": str(args.mert_path),
                    "decoder_dim": args.decoder_dim,
                    "local_layers": args.local_layers,
                    "dropout": args.dropout,
                    "support_scale_levels": args.support_scale_levels,
                },
                "decoder": model.decoder_state_dict(),
                "metrics": metrics,
            }
            save_checkpoint(args.output_dir / "latest.pt", state)
            current = float(metrics["validation_hard_count_selection_f1"])
            is_best = current >= best
            if is_best:
                best = current
                save_checkpoint(args.output_dir / "best.pt", state)
            print(
                f"validation step {step}: loss {metrics['validation_loss']:.4f} "
                f"| hard-f1 {current:.3f}" + (" | best" if is_best else ""),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
