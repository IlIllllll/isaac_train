#!/usr/bin/env python3
"""Extend a T800 motion npz by holding the first and last frames.

The script writes a new npz file and refuses to overwrite the input file.
It reconstructs body states from root pose and joint positions so velocities
remain consistent after the added transition frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from npy_to_npz import (  # noqa: E402
    T800_BODY_ORDER_BFS,
    T800_BODY_ORDER_DFS,
    T800_DFS_JOINT_NAMES,
    build_body_arrays,
    compute_linear_velocity,
    forward_kinematics,
    parse_urdf_joints,
    quat_normalize,
    resolve_urdf,
    save_npz_non_zip64,
)


def seconds_to_frames(seconds: float, fps: float) -> int:
    return max(0, int(round(seconds * fps)))


def normalize_quat_sequence(quats: np.ndarray) -> np.ndarray:
    result = quat_normalize(quats.astype(np.float64))
    for i in range(1, result.shape[0]):
        if np.dot(result[i - 1], result[i]) < 0.0:
            result[i] *= -1.0
    return result


def repeat_frame(base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray, frames: int):
    if frames <= 0:
        return None
    return (
        np.repeat(base_pos[None, :], frames, axis=0),
        np.repeat(base_quat[None, :], frames, axis=0),
        np.repeat(joint_pos[None, :], frames, axis=0),
    )


def append_part(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], part) -> None:
    if part is not None:
        parts.append(part)


def load_motion(input_path: Path):
    with np.load(input_path, allow_pickle=False) as data:
        required = ["joint_pos", "body_pos_w", "body_quat_w", "fps"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{input_path} is missing required arrays: {', '.join(missing)}")
        joint_pos = data["joint_pos"].astype(np.float64)
        base_pos = data["body_pos_w"][:, 0, :].astype(np.float64)
        base_quat = normalize_quat_sequence(data["body_quat_w"][:, 0, :])
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if joint_pos.shape[0] != base_pos.shape[0] or joint_pos.shape[1] != len(T800_DFS_JOINT_NAMES):
        raise ValueError(f"Unexpected motion shapes: joint_pos={joint_pos.shape}, base_pos={base_pos.shape}")
    return base_pos, base_quat, joint_pos, fps


def extend_motion(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Refusing to overwrite the input file. Choose a different --output path.")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to replace it.")

    base_pos, base_quat, joint_pos, fps = load_motion(input_path)
    pre_hold_frames = seconds_to_frames(args.pre_hold, fps)
    post_hold_frames = seconds_to_frames(args.post_hold, fps)

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    append_part(parts, repeat_frame(base_pos[0], base_quat[0], joint_pos[0], pre_hold_frames))
    append_part(parts, (base_pos, base_quat, joint_pos))
    append_part(parts, repeat_frame(base_pos[-1], base_quat[-1], joint_pos[-1], post_hold_frames))

    out_base_pos = np.concatenate([part[0] for part in parts], axis=0)
    out_base_quat = normalize_quat_sequence(np.concatenate([part[1] for part in parts], axis=0))
    out_joint_pos = np.concatenate([part[2] for part in parts], axis=0)

    body_order = T800_BODY_ORDER_DFS if args.body_order == "dfs" else T800_BODY_ORDER_BFS
    dt = 1.0 / fps
    incoming_joints = parse_urdf_joints(resolve_urdf(args.urdf))
    link_positions, link_quaternions = forward_kinematics(out_base_pos, out_base_quat, out_joint_pos, incoming_joints)
    body_pos, body_quat, body_lin_vel, body_ang_vel = build_body_arrays(body_order, link_positions, link_quaternions, dt)
    joint_vel = compute_linear_velocity(out_joint_pos, dt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_non_zip64(
        output_path,
        joint_pos=out_joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        fps=np.asarray([fps], dtype=np.float32),
    )

    print(f"[extend_motion_npz] Input: {input_path}")
    print(f"[extend_motion_npz] Output: {output_path}")
    print(
        "[extend_motion_npz] Frames: "
        f"{joint_pos.shape[0]} -> {out_joint_pos.shape[0]} "
        f"(pre_hold={pre_hold_frames}, post_hold={post_hold_frames})"
    )
    print(f"[extend_motion_npz] FPS: {fps:g}, body_order={args.body_order}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold the first and last frames of a T800 motion npz.")
    parser.add_argument("--input", "-i", required=True, help="Input .npz motion file")
    parser.add_argument("--output", "-o", required=True, help="Output .npz motion file")
    parser.add_argument("--pre_hold", type=float, default=0.8, help="Seconds to repeat the first motion frame")
    parser.add_argument("--post_hold", type=float, default=1.0, help="Seconds to repeat the last motion frame")
    parser.add_argument("--body_order", choices=["dfs", "bfs"], default="dfs", help="Body order for output arrays")
    parser.add_argument("--urdf", type=str, default=None, help="Path to T800 URDF; auto-detected by default")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args()
    extend_motion(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
