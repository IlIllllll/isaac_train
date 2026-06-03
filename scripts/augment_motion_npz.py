#!/usr/bin/env python3
"""Create edge-hold and speed-augmented T800 motion npz files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from npy_to_npz import compute_angular_velocity, compute_linear_velocity, quat_slerp, save_npz_non_zip64


POSE_KEYS = ("joint_pos", "body_pos_w", "body_quat_w")


def load_motion(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = ("joint_pos", "body_pos_w", "body_quat_w", "fps")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
    return {key: np.asarray(data[key]) for key in data.files}


def motion_fps(data: dict[str, np.ndarray]) -> float:
    return float(np.asarray(data["fps"]).reshape(-1)[0])


def _linear_resample(array: np.ndarray, sample_positions: np.ndarray) -> np.ndarray:
    idx0 = np.floor(sample_positions).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, array.shape[0] - 1)
    alpha = (sample_positions - idx0).astype(np.float32)
    view_shape = (alpha.shape[0],) + (1,) * (array.ndim - 1)
    return ((1.0 - alpha).reshape(view_shape) * array[idx0] + alpha.reshape(view_shape) * array[idx1]).astype(
        np.float32
    )


def _quat_resample(array: np.ndarray, sample_positions: np.ndarray) -> np.ndarray:
    idx0 = np.floor(sample_positions).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, array.shape[0] - 1)
    alpha = sample_positions - idx0
    output = np.empty((sample_positions.shape[0],) + array.shape[1:], dtype=np.float32)
    for frame_idx, (a, b, blend) in enumerate(zip(idx0, idx1, alpha, strict=False)):
        for body_idx in range(array.shape[1]):
            output[frame_idx, body_idx] = quat_slerp(array[a, body_idx], array[b, body_idx], float(blend)).astype(
                np.float32
            )
    return output


def _recompute_velocities(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fps = motion_fps(data)
    dt = 1.0 / fps
    data["joint_vel"] = compute_linear_velocity(data["joint_pos"], dt).astype(np.float32)
    data["body_lin_vel_w"] = compute_linear_velocity(data["body_pos_w"], dt).astype(np.float32)
    data["body_ang_vel_w"] = compute_angular_velocity(data["body_quat_w"], dt).astype(np.float32)
    data["fps"] = np.asarray([fps], dtype=np.float32)
    return data


def add_edge_hold(data: dict[str, np.ndarray], target_duration: float) -> dict[str, np.ndarray]:
    fps = motion_fps(data)
    frames = int(data["joint_pos"].shape[0])
    target_frames = max(frames, int(round(target_duration * fps)) + 1)
    extra_frames = target_frames - frames
    start_frames = extra_frames // 2
    end_frames = extra_frames - start_frames

    output = {}
    for key in POSE_KEYS:
        array = np.asarray(data[key], dtype=np.float32)
        pieces = []
        if start_frames:
            pieces.append(np.repeat(array[0:1], start_frames, axis=0))
        pieces.append(array)
        if end_frames:
            pieces.append(np.repeat(array[-1:], end_frames, axis=0))
        output[key] = np.concatenate(pieces, axis=0).astype(np.float32)

    output["fps"] = np.asarray([fps], dtype=np.float32)
    return _recompute_velocities(output)


def speed_resample(data: dict[str, np.ndarray], speed: float) -> dict[str, np.ndarray]:
    if speed <= 0.0:
        raise ValueError(f"Speed factor must be positive, got {speed}")
    frames = int(data["joint_pos"].shape[0])
    if frames <= 1:
        return _recompute_velocities({key: np.asarray(data[key]).copy() for key in POSE_KEYS} | {"fps": data["fps"]})

    new_frames = max(2, int(round((frames - 1) / speed)) + 1)
    sample_positions = np.linspace(0.0, frames - 1, new_frames, dtype=np.float64)
    output = {
        "joint_pos": _linear_resample(np.asarray(data["joint_pos"], dtype=np.float32), sample_positions),
        "body_pos_w": _linear_resample(np.asarray(data["body_pos_w"], dtype=np.float32), sample_positions),
        "body_quat_w": _quat_resample(np.asarray(data["body_quat_w"], dtype=np.float32), sample_positions),
        "fps": np.asarray([motion_fps(data)], dtype=np.float32),
    }
    return _recompute_velocities(output)


def save_motion(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_non_zip64(
        path,
        joint_pos=np.asarray(data["joint_pos"], dtype=np.float32),
        joint_vel=np.asarray(data["joint_vel"], dtype=np.float32),
        body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float32),
        body_quat_w=np.asarray(data["body_quat_w"], dtype=np.float32),
        body_lin_vel_w=np.asarray(data["body_lin_vel_w"], dtype=np.float32),
        body_ang_vel_w=np.asarray(data["body_ang_vel_w"], dtype=np.float32),
        fps=np.asarray(data["fps"], dtype=np.float32),
    )


def motion_summary(path: Path) -> str:
    data = load_motion(path)
    fps = motion_fps(data)
    frames = int(data["joint_pos"].shape[0])
    duration = (frames - 1) / fps if frames else 0.0
    return f"{path}: frames={frames}, fps={fps:g}, duration={duration:.2f}s"


def _speed_tag(speed: float) -> str:
    return f"{speed:.2f}".replace(".", "p")


def _speed_output_name(input_stem: str, target_duration: float, speed: float) -> str:
    duration_tag = f"{target_duration:g}".replace(".", "p")
    edge_suffix = f"_edge_hold_{duration_tag}s"
    if input_stem.endswith(edge_suffix):
        return f"{input_stem}_speed_{_speed_tag(speed)}.npz"
    return f"{input_stem}{edge_suffix}_speed_{_speed_tag(speed)}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    edge = subparsers.add_parser("edge-hold", help="Pad first and last frames until target duration.")
    edge.add_argument("--input", type=Path, required=True)
    edge.add_argument("--output", type=Path, required=True)
    edge.add_argument("--target-duration", type=float, default=3.2)

    speed = subparsers.add_parser("speed", help="Create speed variants, optionally edge-held to target duration.")
    speed.add_argument("--input", type=Path, required=True)
    speed.add_argument("--output-dir", type=Path, required=True)
    speed.add_argument("--speed", type=float, action="append", required=True)
    speed.add_argument("--target-duration", type=float, default=3.2)

    summary = subparsers.add_parser("summary", help="Print motion frame counts.")
    summary.add_argument("files", type=Path, nargs="+")

    args = parser.parse_args()

    if args.command == "edge-hold":
        data = add_edge_hold(load_motion(args.input), args.target_duration)
        save_motion(args.output, data)
        print(motion_summary(args.output))
    elif args.command == "speed":
        source = load_motion(args.input)
        for factor in args.speed:
            data = add_edge_hold(speed_resample(source, factor), args.target_duration)
            output = args.output_dir / _speed_output_name(args.input.stem, args.target_duration, factor)
            save_motion(output, data)
            print(motion_summary(output))
    elif args.command == "summary":
        for path in args.files:
            print(motion_summary(path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
