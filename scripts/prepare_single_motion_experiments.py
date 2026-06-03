#!/usr/bin/env python3
"""Prepare data files and archive scaffolding for single-motion experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from augment_motion_npz import add_edge_hold, load_motion, motion_fps, save_motion, speed_resample
from single_motion_experiment_matrix import EXPERIMENT_NAME, TARGET_DURATION_S, experiments, required_motion_files


REPO_ROOT = Path(__file__).resolve().parents[1]


def _duration(path: Path) -> tuple[int, float, float]:
    data = load_motion(path)
    frames = int(data["joint_pos"].shape[0])
    fps = motion_fps(data)
    duration = (frames - 1) / fps if frames else 0.0
    return frames, fps, duration


def _run_conversion(input_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "npy_to_npz.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--fps",
        "50",
        "--input_fps",
        "30",
        "--use_dfs",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def prepare_motion_data(force: bool = False) -> list[Path]:
    generated: list[Path] = []
    npz_dir = REPO_ROOT / "data" / "npz"
    augmented_dir = REPO_ROOT / "data" / "augmented_npz"
    speed_dir = REPO_ROOT / "data" / "speed_augmented_npz"
    augmented_dir.mkdir(parents=True, exist_ok=True)
    speed_dir.mkdir(parents=True, exist_ok=True)

    _run_conversion(REPO_ROOT / "data" / "npy" / "Punch_Swing_L.npy", npz_dir / "Punch_Swing_L_50hz.npz")
    _run_conversion(REPO_ROOT / "data" / "npy" / "riot_combo.npy", npz_dir / "riot_combo_50hz.npz")

    edge_sources = [
        npz_dir / "kick_Turn_50hz.npz",
        npz_dir / "victory_50hz.npz",
        npz_dir / "Punch_Swing_L_50hz.npz",
        npz_dir / "riot_combo_50hz.npz",
    ]
    for source in edge_sources:
        output = augmented_dir / f"{source.stem}_edge_hold_3p2s.npz"
        if force or not output.exists():
            save_motion(output, add_edge_hold(load_motion(source), TARGET_DURATION_S))
        generated.append(output)

    kick_source = npz_dir / "kick_Turn_50hz.npz"
    for factor in (0.75, 1.0, 1.25):
        tag = f"{factor:.2f}".replace(".", "p")
        output = speed_dir / f"{kick_source.stem}_edge_hold_3p2s_speed_{tag}.npz"
        if force or not output.exists():
            save_motion(output, add_edge_hold(speed_resample(load_motion(kick_source), factor), TARGET_DURATION_S))
        generated.append(output)

    missing = [path for path in required_motion_files() if not (REPO_ROOT / path).exists()]
    if missing:
        raise FileNotFoundError("Missing required motion files after preparation: " + ", ".join(map(str, missing)))
    return generated


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_archive(archive_dir: Path | None = None) -> Path:
    if archive_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        archive_dir = REPO_ROOT / "experiments" / f"{EXPERIMENT_NAME}_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for child in ("videos", "configs", "configs/generated", "logs"):
        (archive_dir / child).mkdir(parents=True, exist_ok=True)

    matrix_rows = [
        {
            "name": exp.name,
            "task": exp.task,
            "motion_files": exp.motion_arg(),
            "seed": exp.seed,
            "hypothesis": exp.hypothesis,
        }
        for exp in experiments()
    ]
    _write_csv(
        archive_dir / "configs" / "experiment_matrix.csv",
        matrix_rows,
        ["name", "task", "motion_files", "seed", "hypothesis"],
    )

    manifest_rows = []
    for path in sorted({path for exp in experiments() for path in exp.motion_files}):
        frames, fps, duration = _duration(REPO_ROOT / path)
        manifest_rows.append(
            {
                "motion_file": str(path),
                "frames": frames,
                "fps": f"{fps:g}",
                "duration_s": f"{duration:.3f}",
            }
        )
    _write_csv(archive_dir / "configs" / "motion_manifest.csv", manifest_rows, ["motion_file", "frames", "fps", "duration_s"])

    commands_md = archive_dir / "commands.md"
    commands_md.write_text(_commands_markdown(), encoding="utf-8")

    readme = archive_dir / "README.md"
    readme.write_text(_readme_markdown(archive_dir), encoding="utf-8")

    notes = archive_dir / "notes.md"
    if not notes.exists():
        notes.write_text(_notes_markdown(), encoding="utf-8")

    results = archive_dir / "results.csv"
    if not results.exists():
        results.write_text(
            "name,stage,seed,task,motion_files,run_dir,checkpoint,play_video_no_terminations,"
            "play_video_normal,error_anchor_pos,error_anchor_rot,error_body_pos,error_body_rot,error_joint_pos,"
            "error_joint_vel,termination_frequency,manual_completeness,manual_jitter,manual_fall,"
            "manual_edge_jump,manual_early_done,notes\n",
            encoding="utf-8",
        )
    return archive_dir


def _commands_markdown() -> str:
    lines = [
        "# Single Motion Optimization Commands",
        "",
        "Run from the repository root.",
        "",
        'If the remote IsaacLab install needs the launcher wrapper, pass `--python "~/Desktop/IsaacLab/isaaclab.sh -p"` to `scripts/run_single_motion_experiments.py`.',
        "",
        "```bash",
        "python scripts/prepare_single_motion_experiments.py --archive-dir experiments/<archive-name>",
        "```",
        "",
    ]
    for exp in experiments():
        motion_arg = exp.motion_arg()
        lines.extend(
            [
                f"## {exp.name}",
                "",
                "Smoke test:",
                "",
                "```bash",
                "python scripts/rsl_rl/train.py "
                f"--task {exp.task} --motion_file {motion_arg} --num_envs 128 --max_iterations 200 "
                f"--seed {exp.seed} --experiment_name single_motion_optimization --run_name {exp.name}_smoke200 --headless",
                "```",
                "",
                "Full run:",
                "",
                "```bash",
                "python scripts/rsl_rl/train.py "
                f"--task {exp.task} --motion_file {motion_arg} --num_envs 128 --max_iterations 5000 "
                f"--seed {exp.seed} --experiment_name single_motion_optimization --run_name {exp.name} --headless",
                "```",
                "",
                "Play with and without terminations after the checkpoint exists:",
                "",
                "```bash",
                "python scripts/run_single_motion_experiments.py "
                f"--archive-dir experiments/<archive-name> --stage play --only {exp.name}",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _readme_markdown(archive_dir: Path) -> str:
    rel = archive_dir.relative_to(REPO_ROOT) if archive_dir.is_relative_to(REPO_ROOT) else archive_dir
    rows = [
        "| Experiment | Task | Seed | Motion files |",
        "| --- | --- | ---: | --- |",
    ]
    for exp in experiments():
        rows.append(f"| {exp.name} | `{exp.task}` | {exp.seed} | `{exp.motion_arg()}` |")
    return "\n".join(
        [
            "# Single Motion Optimization",
            "",
            f"Archive directory: `{rel}`",
            "",
            "This archive is the control point for the six 5000-iteration comparisons. Generated motion npz files and videos are intentionally ignored by Git; matrix and result documents stay in this directory.",
            "",
            "## Matrix",
            "",
            *rows,
            "",
            "## Workflow",
            "",
            "1. Run `python scripts/prepare_single_motion_experiments.py --archive-dir "
            f"{rel}` to regenerate data and manifests.",
            "2. Run `python scripts/run_single_motion_experiments.py --archive-dir "
            f"{rel} --stage smoke --python \"~/Desktop/IsaacLab/isaaclab.sh -p\"` and check reward values/checkpoint creation.",
            "3. Run `python scripts/run_single_motion_experiments.py --archive-dir "
            f"{rel} --stage full --python \"~/Desktop/IsaacLab/isaaclab.sh -p\"` for 5000 iterations per experiment.",
            "4. Run `python scripts/run_single_motion_experiments.py --archive-dir "
            f"{rel} --stage play --python \"~/Desktop/IsaacLab/isaaclab.sh -p\"` to record both no-termination and normal-termination videos.",
            "5. Fill `results.csv` and `notes.md` with manual video ratings and scalar summaries.",
            "",
        ]
    )


def _notes_markdown() -> str:
    return "\n".join(
        [
            "# Notes",
            "",
            "Manual video rating scale: 1 poor, 3 acceptable, 5 good.",
            "",
            "| Experiment | Completeness | Jitter | Fall | Edge jump | Early done | Notes |",
            "| --- | ---: | ---: | --- | --- | --- | --- |",
            *[f"| {exp.name} |  |  |  |  |  |  |" for exp in experiments()],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate augmented motion files even if they exist.")
    args = parser.parse_args()

    prepare_motion_data(force=args.force)
    archive_dir = create_archive(args.archive_dir)
    print(f"[prepare] archive: {archive_dir}")


if __name__ == "__main__":
    main()
