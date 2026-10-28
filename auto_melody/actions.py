"""Canonical 4K action encoding shared by the released model stages."""

from __future__ import annotations

import torch


ACTION_VOCAB_SIZE = 80
ACTION_BOS_TOKEN = ACTION_VOCAB_SIZE

# Lane-invariant event types used by the rhythm-focused planner.  The type is
# (press_count, hold_count); lane assignment belongs exclusively to the second
# stage.  There are 1 + 2 + 3 + 4 + 5 = 15 legal non-mask values.
RHYTHM_TYPE_COUNT = 15
RHYTHM_TYPE_MASK_TOKEN = RHYTHM_TYPE_COUNT
RHYTHM_TYPE_VOCAB_SIZE = RHYTHM_TYPE_COUNT + 1


def action_to_rhythm_type(actions: torch.Tensor) -> torch.Tensor:
    """Convert 80 lane actions to a lane-invariant press/Hold type."""

    states = action_classes_to_lane_states(actions.clamp_min(0))
    press_count = (states > 0).sum(dim=-1)
    hold_count = (states == 2).sum(dim=-1)
    # Triangular packing: p=0 -> 0, p=1 -> 1..2, p=2 -> 3..5, ...
    type_ids = press_count * (press_count + 1) // 2 + hold_count
    return torch.where(
        actions >= 0,
        type_ids.to(dtype=torch.long),
        type_ids.new_full((), -1),
    )


def rhythm_type_counts(types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode lane-invariant types into press and Hold counts."""

    safe = types.clamp_min(0).to(dtype=torch.long)
    press_count = torch.zeros_like(safe)
    hold_count = torch.zeros_like(safe)
    for press in range(1, 5):
        start = press * (press + 1) // 2
        end = start + press + 1
        selected = (safe >= start) & (safe < end)
        press_count = torch.where(selected, press, press_count)
        hold_count = torch.where(selected, safe - start, hold_count)
    return press_count, hold_count


def action_classes_from_pattern_and_holds(
    pattern_targets: torch.Tensor, hold_targets: torch.Tensor
) -> torch.Tensor:
    """Encode each non-empty 4K event as one of 80 tap/Hold actions."""

    lanes = torch.arange(4, device=pattern_targets.device)
    masks = pattern_targets.clamp_min(0).unsqueeze(-1) + 1
    active = ((masks >> lanes) & 1).to(dtype=torch.long)
    states = active * (1 + (hold_targets.to(dtype=torch.long) & active))
    powers = states.new_tensor((1, 3, 9, 27))
    classes = (states * powers).sum(dim=-1) - 1
    return torch.where(pattern_targets >= 0, classes, classes.new_full((), -1))


def action_classes_to_lane_states(classes: torch.Tensor) -> torch.Tensor:
    """Decode action classes into four lane states: empty, tap, or Hold start."""

    powers = classes.new_tensor((1, 3, 9, 27))
    return ((classes.to(dtype=torch.long).unsqueeze(-1) + 1) // powers) % 3
