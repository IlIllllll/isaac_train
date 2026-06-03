#!/usr/bin/env python3
"""Export selected TensorBoard scalar tags from RSL-RL run directories."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_PATTERNS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_body_pos",
    "error_body_rot",
    "error_joint_pos",
    "error_joint_vel",
    "termination",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", action="append", default=None, help="Substring to match; can be passed multiple times.")
    args = parser.parse_args()

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        raise RuntimeError("Install tensorboard to export scalar event files.") from exc

    patterns = tuple(args.tag or DEFAULT_PATTERNS)
    event_files = sorted(args.run_dir.rglob("events.out.tfevents.*"))
    rows = []
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if not any(pattern in tag for pattern in patterns):
                continue
            for item in accumulator.Scalars(tag):
                rows.append(
                    {
                        "event_file": str(event_file),
                        "tag": tag,
                        "step": item.step,
                        "wall_time": item.wall_time,
                        "value": item.value,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event_file", "tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] wrote {len(rows)} scalar rows to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
