#!/usr/bin/env python3
"""Train the lane arranger from a fixed rhythm skeleton."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from auto_melody.lane_arranger import (
    LANE_ARRANGER_SCHEMA,
    EventCountBatchSampler,
    LaneArranger,
    LaneActionDataset,
    collate_lane_actions,
)
from auto_melody.consensus_training import (
    append_jsonl,
    cycle_data_batches,
    build_data_loader,
    save_checkpoint,
    seed_everything,
    select_training_device,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train lane-arranger-v2")
    parser.add_argument("--train-manifest", default="data/splits-difficulty/train.jsonl")
    parser.add_argument("--validation-manifest", default="data/splits-difficulty/validation.jsonl")
    parser.add_argument("--dataset-dir", default="malody-map")
    parser.add_argument("--output-dir", default="outputs/lane-arranger-v2")
    parser.add_argument("--decoder-dim", type=int, default=384)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def loss_and_accuracy(model: LaneArranger, batch: dict[str, torch.Tensor], device: torch.device):
    lane_output = model.teacher_forced(
        batch["ticks"].to(device),
        batch["actions"].to(device),
        batch["lengths"].to(device),
        batch["difficulty"].to(device),
    )
    loss = functional.cross_entropy(
        lane_output.logits[lane_output.mask], lane_output.targets[lane_output.mask]
    )
    accuracy = float(
        (
            lane_output.logits[lane_output.mask].argmax(dim=-1)
            == lane_output.targets[lane_output.mask]
        )
        .float().mean()
    )
    return loss, accuracy


@torch.inference_mode()
def validate(model, loader, device, amp_dtype):
    model.eval()
    losses, accuracies = [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, accuracy = loss_and_accuracy(model, batch, device)
        losses.append(float(loss))
        accuracies.append(accuracy)
    model.train()
    return {
        "validation_loss": float(np.mean(losses)) if losses else float("inf"),
        "validation_action_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "validation_batches": len(losses),
    }


def save_training_checkpoint(path: Path, model, config, step: int, metrics: dict):
    checkpoint = {
        "schema": LANE_ARRANGER_SCHEMA,
        "model_id": "lane-arranger-v2",
        "stage": "lane-arranger",
        "initialization": "random",
        "parent_checkpoint": None,
        "model": model.state_dict(),
        "step": step,
        "config": config,
        "metrics": metrics,
    }
    save_checkpoint(path, checkpoint)


def main() -> int:
    args = build_parser().parse_args()
    if min(args.max_events, args.max_steps, args.eval_steps) <= 0:
        raise ValueError("event and step counts must be positive")
    seed_everything(args.seed)
    device, amp_dtype = select_training_device()
    output_dir = Path(args.output_dir)
    if any(
        (output_dir / name).exists()
        for name in ("best.pt", "latest.pt", "metrics.jsonl")
    ):
        raise FileExistsError(f"refusing to overwrite existing run in {output_dir}")
    train = LaneActionDataset(args.train_manifest, args.dataset_dir)
    validation = LaneActionDataset(args.validation_manifest, args.dataset_dir, mirror_probability=0.0)
    train_sampler = EventCountBatchSampler(
        train.lengths, max_events=args.max_events, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    validation_sampler = EventCountBatchSampler(
        validation.lengths, max_events=args.max_events, shuffle=False
    )
    train_loader = build_data_loader(
        train,
        batch_sampler=train_sampler,
        workers=0,
        collate_fn=collate_lane_actions,
    )
    validation_loader = build_data_loader(
        validation,
        batch_sampler=validation_sampler,
        workers=0,
        collate_fn=collate_lane_actions,
    )
    model = LaneArranger(
        decoder_dim=args.decoder_dim, decoder_layers=args.decoder_layers, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    config = {
        **vars(args),
        "schema": LANE_ARRANGER_SCHEMA,
        "model_id": "lane-arranger-v2",
        "stage": "lane-arranger",
        "initialization": "random",
        "parent_checkpoint": None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(output_dir / "config.json", config)
    print(
        f"lane-arranger-v2 | device={device.type} charts={len(train):,} "
        f"trainable={config['parameters'] / 1e6:.2f}M",
        flush=True,
    )
    batches = cycle_data_batches(train_loader)
    best = float("inf")
    started = time.monotonic()
    for step in range(1, args.max_steps + 1):
        batch = next(batches)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, accuracy = loss_and_accuracy(model, batch, device)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_steps == 0:
            print(
                f"step {step:4d}/{args.max_steps} | loss {float(loss):.4f} "
                f"acc {accuracy:.3f} grad {float(gradient):.3f}", flush=True
            )
        if step % args.eval_steps == 0 or step == args.max_steps:
            metrics = validate(model, validation_loader, device, amp_dtype)
            metrics["step"] = step
            append_jsonl(output_dir / "metrics.jsonl", metrics)
            print(
                f"validation {step:4d} | loss {metrics['validation_loss']:.4f} "
                f"acc {metrics['validation_action_accuracy']:.3f}", flush=True
            )
            save_training_checkpoint(
                output_dir / "latest.pt", model, config, step, metrics
            )
            if metrics["validation_loss"] < best:
                best = metrics["validation_loss"]
                save_training_checkpoint(
                    output_dir / "best.pt", model, config, step, metrics
                )
    print(f"finished in {time.monotonic() - started:.1f}s best={best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
