#!/usr/bin/env python3
"""Run or print the six single-motion optimization experiments."""

from __future__ import annotations

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from single_motion_experiment_matrix import EXPERIMENT_NAME, Experiment, experiments


REPO_ROOT = Path(__file__).resolve().parents[1]


def _selected(names: str | None) -> tuple[Experiment, ...]:
    all_experiments = experiments()
    if not names:
        return all_experiments
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    unknown = wanted - {exp.name for exp in all_experiments}
    if unknown:
        raise ValueError(f"Unknown experiment(s): {', '.join(sorted(unknown))}")
    return tuple(exp for exp in all_experiments if exp.name in wanted)


def _motion_frames(exp: Experiment) -> int:
    frames = []
    for path in exp.motion_files:
        data = np.load(REPO_ROOT / path, allow_pickle=False)
        frames.append(int(data["joint_pos"].shape[0]))
    return max(frames)


def _command_prefix(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("--python command cannot be empty")
    if parts[0].startswith("~"):
        parts[0] = str(Path(parts[0]).expanduser())
    return parts


def _train_cmd(exp: Experiment, iterations: int, num_envs: int, run_name: str, python_prefix: list[str]) -> list[str]:
    return [
        *python_prefix,
        "scripts/rsl_rl/train.py",
        "--task",
        exp.task,
        "--motion_file",
        exp.motion_arg(),
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(iterations),
        "--seed",
        str(exp.seed),
        "--experiment_name",
        EXPERIMENT_NAME,
        "--run_name",
        run_name,
        "--headless",
    ]


def _play_cmd(exp: Experiment, run_dir: Path, no_terminations: bool, python_prefix: list[str]) -> list[str]:
    checkpoint = _latest_checkpoint(run_dir)
    cmd = [
        *python_prefix,
        "scripts/rsl_rl/play.py",
        "--task",
        exp.task,
        "--motion_file",
        exp.motion_arg(),
        "--num_envs",
        "1",
        "--experiment_name",
        EXPERIMENT_NAME,
        "--load_run",
        run_dir.name,
        "--checkpoint",
        checkpoint.name,
        "--headless",
        "--video",
        "--video_length",
        str(_motion_frames(exp)),
    ]
    if no_terminations:
        cmd.extend(["--no_terminations", "true"])
    return cmd


def _latest_run(exp: Experiment) -> Path:
    log_root = REPO_ROOT / "logs" / "rsl_rl" / EXPERIMENT_NAME
    candidates = sorted(log_root.glob(f"*_{exp.name}"))
    if not candidates:
        raise FileNotFoundError(f"No full run directory found for {exp.name} under {log_root}")
    return candidates[-1]


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dir}")
    return checkpoints[-1]


def _append_result(archive_dir: Path, row: dict[str, object]) -> None:
    path = archive_dir / "results.csv"
    fieldnames = [
        "name",
        "stage",
        "seed",
        "task",
        "motion_files",
        "run_dir",
        "checkpoint",
        "play_video_no_terminations",
        "play_video_normal",
        "error_anchor_pos",
        "error_anchor_rot",
        "error_body_pos",
        "error_body_rot",
        "error_joint_pos",
        "error_joint_vel",
        "termination_frequency",
        "manual_completeness",
        "manual_jitter",
        "manual_fall",
        "manual_edge_jump",
        "manual_early_done",
        "notes",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _run(cmd: list[str], log_file: Path, dry_run: bool) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def _copy_params(run_dir: Path, archive_dir: Path, exp_name: str) -> None:
    params_dir = run_dir / "params"
    if not params_dir.exists():
        return
    target = archive_dir / "configs" / "generated" / exp_name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(params_dir, target)


def run_stage(
    stage: str,
    archive_dir: Path,
    selected: tuple[Experiment, ...],
    num_envs: int,
    smoke_iterations: int,
    iterations: int,
    python_prefix: list[str],
    dry_run: bool,
) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for exp in selected:
        if stage in {"smoke", "all"}:
            run_name = f"{exp.name}_smoke{smoke_iterations}"
            cmd = _train_cmd(exp, smoke_iterations, num_envs, run_name, python_prefix)
            rc = _run(cmd, archive_dir / "logs" / f"{run_name}.log", dry_run)
            _append_result(archive_dir, _base_row(exp, "smoke", return_code=rc))
            if rc != 0:
                raise RuntimeError(f"Smoke run failed for {exp.name}; see {archive_dir / 'logs' / (run_name + '.log')}")

        if stage in {"full", "all"}:
            cmd = _train_cmd(exp, iterations, num_envs, exp.name, python_prefix)
            rc = _run(cmd, archive_dir / "logs" / f"{exp.name}.log", dry_run)
            row = _base_row(exp, "full", return_code=rc)
            if rc == 0 and not dry_run:
                run_dir = _latest_run(exp)
                row["run_dir"] = run_dir
                row["checkpoint"] = _latest_checkpoint(run_dir)
                _copy_params(run_dir, archive_dir, exp.name)
            _append_result(archive_dir, row)
            if rc != 0:
                raise RuntimeError(f"Full run failed for {exp.name}; see {archive_dir / 'logs' / (exp.name + '.log')}")

        if stage in {"play", "all"}:
            if dry_run:
                print(f"# play requires an existing run for {exp.name}")
                continue
            run_dir = _latest_run(exp)
            checkpoint = _latest_checkpoint(run_dir)
            no_term_cmd = _play_cmd(exp, run_dir, True, python_prefix)
            normal_cmd = _play_cmd(exp, run_dir, False, python_prefix)
            rc_no_term = _run(no_term_cmd, archive_dir / "logs" / f"{exp.name}_play_no_terminations.log", dry_run)
            rc_normal = _run(normal_cmd, archive_dir / "logs" / f"{exp.name}_play_normal.log", dry_run)
            video_dir = run_dir / "videos" / "play"
            _append_result(
                archive_dir,
                _base_row(
                    exp,
                    "play",
                    run_dir=run_dir,
                    checkpoint=checkpoint,
                    play_video_no_terminations=video_dir,
                    play_video_normal=video_dir,
                    notes=f"return_codes no_terminations={rc_no_term} normal={rc_normal}",
                ),
            )
            if rc_no_term != 0 or rc_normal != 0:
                raise RuntimeError(f"Play failed for {exp.name}; see logs under {archive_dir / 'logs'}")


def _base_row(exp: Experiment, stage: str, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": exp.name,
        "stage": stage,
        "seed": exp.seed,
        "task": exp.task,
        "motion_files": exp.motion_arg(),
        "notes": "",
    }
    if "return_code" in extra:
        row["notes"] = f"return_code={extra.pop('return_code')}"
    row.update(extra)
    return row


def write_command_preview(archive_dir: Path, selected: tuple[Experiment, ...], python_prefix: list[str], num_envs: int) -> None:
    lines = [
        f"# Command preview generated {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for exp in selected:
        lines.append(f"## {exp.name}")
        lines.append("```bash")
        lines.append(" ".join(_train_cmd(exp, 200, num_envs, f"{exp.name}_smoke200", python_prefix)))
        lines.append(" ".join(_train_cmd(exp, 5000, num_envs, exp.name, python_prefix)))
        lines.append("```")
        lines.append("")
    (archive_dir / "commands.preview.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("commands", "smoke", "full", "play", "all"), default="commands")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated experiment names.")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--smoke_iterations", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument(
        "--python",
        dest="python_bin",
        default="python",
        help='Python command prefix, for example "python" or "~/Desktop/IsaacLab/isaaclab.sh -p".',
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = _selected(args.only)
    python_prefix = _command_prefix(args.python_bin)
    archive_dir = args.archive_dir if args.archive_dir.is_absolute() else REPO_ROOT / args.archive_dir
    if args.stage == "commands":
        archive_dir.mkdir(parents=True, exist_ok=True)
        write_command_preview(archive_dir, selected, python_prefix, args.num_envs)
        print(f"[commands] wrote {archive_dir / 'commands.preview.md'}")
        return

    run_stage(
        args.stage,
        archive_dir,
        selected,
        args.num_envs,
        args.smoke_iterations,
        args.iterations,
        python_prefix,
        args.dry_run,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
