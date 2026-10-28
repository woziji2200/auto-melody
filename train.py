#!/usr/bin/env python3
"""Train one of the three consensus-rhythm-phrase-chart-v1 stages."""

from __future__ import annotations

import subprocess
import sys


STAGES = {
    "salience": "training.train_consensus_salience",
    "rhythm": "training.train_consensus_rhythm",
    "lane": "training.train_lane_arranger",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        names = " | ".join(STAGES)
        print(f"usage: python train.py <{names}> [stage options]", file=sys.stderr)
        return 2
    return subprocess.run(
        [sys.executable, "-m", STAGES[sys.argv[1]], *sys.argv[2:]],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
