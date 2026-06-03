#!/usr/bin/env python3
"""Run the high-dynamic speed-lead training pipeline on GPU 7."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from augment_motion_npz import add_edge_hold, load_motion, motion_summary, save_motion, speed_resample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = Path("/workspace/isaaclab/isaaclab.sh")
TASK = "Tracking-Flat-T800-HighDyn-SpeedLead3p2s-v0"
EXPERIMENT = "highdyn_single"
LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / EXPERIMENT
GPU_INDEX = int(os.environ.get("HIGHDYN_GPU_INDEX", "7"))
MIN_FREE_GPU_MB = int(os.environ.get("HIGHDYN_MIN_FREE_GPU_MB", "30000"))
SPEED_FACTORS = (0.85, 1.00, 1.15)
VIDEO_LENGTH = 180

MOTIONS = {
    "kick": PROJECT_ROOT / "data" / "augmented_npz" / "kick_Turn_50hz_edge_hold_3p2s.npz",
    "riot": PROJECT_ROOT / "data" / "augmented_npz" / "riot_combo_50hz_edge_hold_3p2s.npz",
}

SEEDS = {"kick": 2201, "riot": 2301}
BASE_RUNS = {
    "kick": "2026-05-20_06-42-59_kick_edgehold_p3_final_seed2201",
    "riot": "2026-05-20_06-42-59_riot_edgehold_p3_final_seed2301",
}
BASE_CHECKPOINT = "model_57997.pt"

SCALAR_TAGS = (
    "Train/mean_episode_length",
    "Train/mean_reward",
    "Episode_Termination/time_out",
    "Episode_Termination/anchor_pos",
    "Episode_Termination/anchor_ori",
    "Episode_Termination/ee_body_pos",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_body_lin_vel",
    "Metrics/motion/error_body_ang_vel",
)


def speed_tag(speed: float) -> str:
    return f"{speed:.2f}".replace(".", "p")


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[COMMAND] " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def gpu_free_mb(index: int) -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    used, total = [int(part.strip()) for part in output.splitlines()[0].split(",")]
    return total - used


def wait_for_gpu(index: int, min_free_mb: int, status_path: Path, poll_s: int = 300) -> None:
    while True:
        free_mb = gpu_free_mb(index)
        timestamp = datetime.now().isoformat(timespec="seconds")
        if free_mb is None:
            status_path.write_text(f"{timestamp} gpu_status=unknown, continuing\n", encoding="utf-8")
            return
        status_path.write_text(
            f"{timestamp} gpu={index} free_mb={free_mb} required_mb={min_free_mb}\n", encoding="utf-8"
        )
        if free_mb >= min_free_mb:
            return
        time.sleep(poll_s)


def ensure_speed_files() -> dict[str, list[Path]]:
    output_dir = PROJECT_ROOT / "data" / "highdyn_speed_npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}
    for name, source_path in MOTIONS.items():
        source = load_motion(source_path)
        result[name] = []
        for factor in SPEED_FACTORS:
            output_path = output_dir / f"{source_path.stem}_speed_{speed_tag(factor)}.npz"
            if not output_path.exists():
                data = add_edge_hold(speed_resample(source, factor), 3.2)
                save_motion(output_path, data)
            print(motion_summary(output_path), flush=True)
            result[name].append(output_path)
    return result


def latest_run_dir(run_name: str) -> Path:
    candidates = sorted(LOG_ROOT.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run directory found for run_name={run_name}")
    return candidates[-1]


def latest_checkpoint(run_dir: Path) -> str:
    pattern = re.compile(r"model_(\d+)\.pt$")
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        match = pattern.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path.name))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dir}")
    return max(checkpoints)[1]


def read_last_scalars(run_dir: Path) -> dict[str, float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values: dict[str, float] = {}
    for event_file in sorted(run_dir.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = set(accumulator.Tags().get("scalars", []))
        for tag in SCALAR_TAGS:
            if tag in tags:
                scalars = accumulator.Scalars(tag)
                if scalars:
                    values[tag] = float(scalars[-1].value)
    return values


def selection_score(metrics: dict[str, float]) -> float:
    lin = metrics.get("Metrics/motion/error_body_lin_vel", 1.0e6)
    ang = metrics.get("Metrics/motion/error_body_ang_vel", 1.0e6)
    body_pos = metrics.get("Metrics/motion/error_body_pos", 1.0e6)
    early = (
        metrics.get("Episode_Termination/anchor_pos", 0.0)
        + metrics.get("Episode_Termination/anchor_ori", 0.0)
        + metrics.get("Episode_Termination/ee_body_pos", 0.0)
    )
    timeout = metrics.get("Episode_Termination/time_out", 0.0)
    return lin + 0.25 * ang + 0.5 * body_pos + 10.0 * early + max(0.0, 0.5 - timeout) * 10.0


def train_stage(
    *,
    action: str,
    stage: str,
    run_name: str,
    motion_file: str,
    iterations: int,
    resume_run: str,
    checkpoint: str,
    lead: int,
    pipeline_dir: Path,
    extra_overrides: list[str] | None = None,
) -> tuple[Path, str, dict[str, float]]:
    wait_for_gpu(GPU_INDEX, MIN_FREE_GPU_MB, pipeline_dir / "gpu_wait_status.txt")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    env["PYTHONUNBUFFERED"] = "1"
    source_path = str(PROJECT_ROOT / "source" / "whole_body_tracking")
    env["PYTHONPATH"] = source_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        str(ISAACLAB),
        "-p",
        "scripts/rsl_rl/train.py",
        "--task",
        TASK,
        "--motion_file",
        motion_file,
        "--num_envs",
        "4096",
        "--max_iterations",
        str(iterations),
        "--seed",
        str(SEEDS[action]),
        "--headless",
        "--logger",
        "tensorboard",
        "--experiment_name",
        EXPERIMENT,
        "--run_name",
        run_name,
        "--resume",
        "True",
        "--load_run",
        resume_run,
        "--checkpoint",
        checkpoint,
        f"env.actions.joint_pos.ref_advance_steps={lead}",
        "env.scene.contact_forces.debug_vis=false",
        "env.commands.motion.debug_vis=false",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)
    run_cmd(cmd, pipeline_dir / f"{stage}_{run_name}.log", env=env)
    run_dir = latest_run_dir(run_name)
    last_ckpt = latest_checkpoint(run_dir)
    metrics = read_last_scalars(run_dir)
    return run_dir, last_ckpt, metrics


def record_video(
    *,
    stage: str,
    run_dir: Path,
    checkpoint: str,
    motion_file: str,
    lead: int,
    no_terminations: bool,
    pipeline_dir: Path,
) -> None:
    wait_for_gpu(GPU_INDEX, MIN_FREE_GPU_MB, pipeline_dir / "gpu_wait_status.txt")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    env["PYTHONUNBUFFERED"] = "1"
    source_path = str(PROJECT_ROOT / "source" / "whole_body_tracking")
    env["PYTHONPATH"] = source_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    label = "no_terminations" if no_terminations else "normal"
    cmd = [
        str(ISAACLAB),
        "-p",
        "scripts/rsl_rl/play.py",
        "--task",
        TASK,
        "--motion_file",
        motion_file,
        "--num_envs",
        "1",
        "--load_run",
        run_dir.name,
        "--checkpoint",
        checkpoint,
        "--experiment_name",
        EXPERIMENT,
        "--video",
        "--video_length",
        str(VIDEO_LENGTH),
        "--headless",
        "--skip_mnn",
        f"env.actions.joint_pos.ref_advance_steps={lead}",
        "env.scene.contact_forces.debug_vis=false",
        "env.commands.motion.debug_vis=false",
    ]
    if no_terminations:
        cmd.extend(["--no_terminations", "true"])
    run_cmd(cmd, pipeline_dir / f"video_{stage}_{run_dir.name}_{label}.log", env=env)

    play_dir = run_dir / "videos" / "play"
    videos = sorted(play_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if videos:
        target = run_dir / "videos" / f"speedlead_{stage}_{label}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(videos[-1]), target)


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def run_action(action: str, speed_files: list[Path], pipeline_dir: Path, state: dict[str, Any]) -> None:
    source_motion = str(MOTIONS[action])
    stage_a: dict[int, dict[str, Any]] = {}
    for lead in (1, 2):
        run_name = f"{action}_speedlead_l{lead}_sweep_seed{SEEDS[action]}"
        run_dir, checkpoint, metrics = train_stage(
            action=action,
            stage=f"{action}_stage_a_l{lead}",
            run_name=run_name,
            motion_file=source_motion,
            iterations=6000,
            resume_run=BASE_RUNS[action],
            checkpoint=BASE_CHECKPOINT,
            lead=lead,
            pipeline_dir=pipeline_dir,
        )
        stage_a[lead] = {
            "run_dir": run_dir.name,
            "checkpoint": checkpoint,
            "metrics": metrics,
            "score": selection_score(metrics),
        }
        state[action] = {"stage_a": stage_a}
        write_state(pipeline_dir / "state.json", state)

    lead1_score = stage_a[1]["score"]
    lead2_score = stage_a[2]["score"]
    selected_lead = 1 if lead1_score <= lead2_score * 1.03 else 2
    selected = stage_a[selected_lead]
    selected_run = LOG_ROOT / selected["run_dir"]
    selected_checkpoint = selected["checkpoint"]
    state[action]["selected_lead"] = selected_lead
    write_state(pipeline_dir / "state.json", state)

    for no_terminations in (False, True):
        record_video(
            stage="stage_a",
            run_dir=selected_run,
            checkpoint=selected_checkpoint,
            motion_file=source_motion,
            lead=selected_lead,
            no_terminations=no_terminations,
            pipeline_dir=pipeline_dir,
        )

    multispeed_motion = ",".join(str(path) for path in speed_files)
    run_dir_b, checkpoint_b, metrics_b = train_stage(
        action=action,
        stage=f"{action}_stage_b",
        run_name=f"{action}_speedlead_multispeed_seed{SEEDS[action]}",
        motion_file=multispeed_motion,
        iterations=12000,
        resume_run=selected_run.name,
        checkpoint=selected_checkpoint,
        lead=selected_lead,
        pipeline_dir=pipeline_dir,
    )
    state[action]["stage_b"] = {"run_dir": run_dir_b.name, "checkpoint": checkpoint_b, "metrics": metrics_b}
    write_state(pipeline_dir / "state.json", state)
    for no_terminations in (False, True):
        record_video(
            stage="stage_b",
            run_dir=run_dir_b,
            checkpoint=checkpoint_b,
            motion_file=multispeed_motion,
            lead=selected_lead,
            no_terminations=no_terminations,
            pipeline_dir=pipeline_dir,
        )

    final_overrides = [
        "env.terminations.anchor_pos.params.threshold=0.55",
        "env.terminations.anchor_ori.params.threshold=1.25",
        "env.terminations.ee_body_pos.params.threshold=0.65",
        "env.rewards.action_rate_l2.weight=-0.06",
    ]
    run_dir_c, checkpoint_c, metrics_c = train_stage(
        action=action,
        stage=f"{action}_stage_c",
        run_name=f"{action}_speedlead_final_seed{SEEDS[action]}",
        motion_file=source_motion,
        iterations=16000,
        resume_run=run_dir_b.name,
        checkpoint=checkpoint_b,
        lead=selected_lead,
        pipeline_dir=pipeline_dir,
        extra_overrides=final_overrides,
    )
    state[action]["stage_c"] = {"run_dir": run_dir_c.name, "checkpoint": checkpoint_c, "metrics": metrics_c}
    write_state(pipeline_dir / "state.json", state)
    for no_terminations in (False, True):
        record_video(
            stage="stage_c",
            run_dir=run_dir_c,
            checkpoint=checkpoint_c,
            motion_file=source_motion,
            lead=selected_lead,
            no_terminations=no_terminations,
            pipeline_dir=pipeline_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", nargs="+", choices=("kick", "riot"), default=("kick", "riot"))
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pipeline_dir = LOG_ROOT / f"speedlead_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"pipeline_dir": str(pipeline_dir), "gpu": GPU_INDEX}
    write_state(pipeline_dir / "state.json", state)

    speed_files = ensure_speed_files()
    state["speed_files"] = {name: [str(path) for path in paths] for name, paths in speed_files.items()}
    write_state(pipeline_dir / "state.json", state)

    for action in args.actions:
        run_action(action, speed_files[action], pipeline_dir, state)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
