"""Train the consensus rhythm stage on a frozen salience encoder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch

from auto_melody.dataset import MalodySongDataset, collate_song_batch
from auto_melody.consensus_training import (
    append_jsonl,
    cycle_data_batches,
    rhythm_loss_metrics,
    build_data_loader,
    build_song_batch_sampler,
    move_batch_tensors,
    save_checkpoint,
    seed_everything,
    select_training_device,
    write_json,
)
from auto_melody.consensus_rhythm import GUIDED_RHYTHM_SCHEMA, SalienceGuidedRhythmPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train consensus-rhythm-planner-v1")
    parser.add_argument("--salience-checkpoint", type=Path, required=True)
    parser.add_argument("--mert-path", type=Path, default=Path("data/MERT"))
    parser.add_argument("--train-manifest", type=Path, default=Path("data/splits-difficulty/train.jsonl"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/splits-difficulty/validation.jsonl"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("malody-map"))
    parser.add_argument("--audio-cache-dir", type=Path, default=Path("data/audio-cache-24k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/consensus-rhythm-v1"))
    parser.add_argument("--decoder-dim", type=int, default=512)
    parser.add_argument("--refinement-layers", type=int, default=2)
    parser.add_argument("--measure-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--count-weight", type=float, default=1.0)
    parser.add_argument("--contour-weight", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=3407)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


@torch.inference_mode()
def validate(model, loader, device, amp_dtype, count_weight, contour_weight):
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    for original in loader:
        batch = move_batch_tensors(original, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            rhythm_output = model(
                batch["input_values"],
                batch["attention_mask"],
                batch["difficulty"],
                batch["beat_window_indices"],
                batch["beat_frame_indices"],
                batch["beat_ticks"],
                batch["beat_song_lengths"],
            )
            _, metrics = rhythm_loss_metrics(
                rhythm_output,
                batch,
                count_weight=count_weight,
                contour_weight=contour_weight,
            )
        weight = int(batch["beat_song_lengths"].numel())
        for name, metric in metrics.items():
            totals[name] = totals.get(name, 0.0) + metric * weight
        seen += weight
    model.train()
    return {f"validation_{name}": total / max(seen, 1) for name, total in totals.items()}


def save_training_checkpoint(path, model, args, step, metrics):
    state = {
        "schema": GUIDED_RHYTHM_SCHEMA,
        "model_id": "consensus-rhythm-planner-v1",
        "stage": "consensus-rhythm",
        "initialization": "random-decoder-guided-by-frozen-consensus-salience",
        "parent_checkpoint": str(args.salience_checkpoint),
        "step": step,
        "config": {
            "salience_checkpoint": str(args.salience_checkpoint),
            "decoder_dim": args.decoder_dim,
            "refinement_layers": args.refinement_layers,
            "measure_layers": args.measure_layers,
            "dropout": args.dropout,
            "window_seconds": 5.0,
            "representation": "lane_invariant_type",
            "count_weight": args.count_weight,
            "contour_weight": args.contour_weight,
        },
        "decoder": model.decoder_state_dict(),
        "metrics": metrics,
    }
    save_checkpoint(path, state)


def main() -> int:
    args = build_parser().parse_args()
    if (
        min(
            args.max_windows,
            args.max_steps,
            args.eval_steps,
            args.refinement_layers,
            args.measure_layers,
        )
        <= 0
    ):
        raise ValueError("invalid training limits")
    seed_everything(args.seed)
    device, amp_dtype = select_training_device()
    train_data = MalodySongDataset(
        args.train_manifest,
        args.dataset_dir,
        audio_cache_dir=args.audio_cache_dir,
        mirror_probability=0.5,
    )
    validation_data = MalodySongDataset(
        args.validation_manifest,
        args.dataset_dir,
        audio_cache_dir=args.audio_cache_dir,
        mirror_probability=0.0,
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
    model = SalienceGuidedRhythmPlanner(
        args.salience_checkpoint,
        mert_path=args.mert_path,
        decoder_dim=args.decoder_dim,
        refinement_layers=args.refinement_layers,
        measure_layers=args.measure_layers,
        dropout=args.dropout,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.min_learning_rate
    )
    write_json(args.output_dir / "config.json", vars(args))
    print(
        "consensus-guided rhythm training ready\n"
        f"  songs: {len(train_data):,} train / {len(validation_data):,} validation\n"
        f"  trainable parameters: {sum(parameter.numel() for parameter in parameters):,}",
        flush=True,
    )
    batches = cycle_data_batches(train_loader)
    best = math.inf
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        batch = move_batch_tensors(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            rhythm_output = model(
                batch["input_values"],
                batch["attention_mask"],
                batch["difficulty"],
                batch["beat_window_indices"],
                batch["beat_frame_indices"],
                batch["beat_ticks"],
                batch["beat_song_lengths"],
            )
            loss, metrics = rhythm_loss_metrics(
                rhythm_output,
                batch,
                count_weight=args.count_weight,
                contour_weight=args.contour_weight,
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_steps == 0 or step == 1:
            print(
                f"step {step:>4}/{args.max_steps} | loss {metrics['loss']:.4f} "
                f"| select-f1 {metrics['selection_f1']:.3f} "
                f"| type {metrics['type_accuracy']:.3f} "
                f"| grad {float(gradient_norm):.2f} "
                f"| {(time.perf_counter() - started) / step:.2f}s/step",
                flush=True,
            )
        if step % args.eval_steps == 0 or step == args.max_steps:
            metrics = validate(model, validation_loader, device, amp_dtype, args.count_weight, args.contour_weight)
            metrics.update(step=step, learning_rate=optimizer.param_groups[0]["lr"])
            append_jsonl(args.output_dir / "metrics.jsonl", metrics)
            current = float(metrics["validation_loss"])
            save_training_checkpoint(
                args.output_dir / "latest.pt", model, args, step, metrics
            )
            is_best = current < best
            if is_best:
                best = current
                save_training_checkpoint(args.output_dir / "best.pt", model, args, step, metrics)
            print(
                f"validation step {step}: loss {current:.4f} "
                f"| select-f1 {metrics['validation_selection_f1']:.3f} "
                f"| type {metrics['validation_type_accuracy']:.3f}"
                + (" | best" if is_best else ""),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
