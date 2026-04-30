#!/usr/bin/env python3
"""Extend a T800 motion npz with stand, blend-in, blend-out, and hold phases.

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
    quat_slerp,
    resolve_urdf,
    save_npz_non_zip64,
)


DEFAULT_JOINT_POS = {
    "J00_HIP_PITCH_L": -0.06,
    "J01_HIP_ROLL_L": 0.0,
    "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": 0.12,
    "J04_ANKLE_PITCH_L": -0.06,
    "J05_ANKLE_ROLL_L": 0.0,
    "J06_HIP_PITCH_R": -0.06,
    "J07_HIP_ROLL_R": 0.0,
    "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": 0.12,
    "J10_ANKLE_PITCH_R": -0.06,
    "J11_ANKLE_ROLL_R": 0.0,
    "J12_TORSO_YAW": 0.0,
    "J13_SHOULDER_PITCH_L": 0.0,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": -0.25,
    "J17_ELBOW_YAW_L": 0.0,
    "J20_SHOULDER_PITCH_R": 0.0,
    "J21_SHOULDER_ROLL_R": -0.15,
    "J22_SHOULDER_YAW_R": 0.0,
    "J23_ELBOW_PITCH_R": -0.25,
    "J24_ELBOW_YAW_R": 0.0,
    "J27_HEAD_PITCH": 0.0,
    "J28_HEAD_YAW": 0.0,
}


def seconds_to_frames(seconds: float, fps: float) -> int:
    return max(0, int(round(seconds * fps)))


def smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def default_joint_pose() -> np.ndarray:
    return np.asarray([DEFAULT_JOINT_POS[name] for name in T800_DFS_JOINT_NAMES], dtype=np.float64)


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = quat_normalize(quat.astype(np.float64))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_from_yaw_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def normalize_quat_sequence(quats: np.ndarray) -> np.ndarray:
    result = quat_normalize(quats.astype(np.float64))
    for i in range(1, result.shape[0]):
        if np.dot(result[i - 1], result[i]) < 0.0:
            result[i] *= -1.0
    return result


def make_stand_frame(reference_pos: np.ndarray, reference_quat: np.ndarray, stand_height: float):
    base_pos = np.asarray([reference_pos[0], reference_pos[1], stand_height], dtype=np.float64)
    base_quat = quat_from_yaw_wxyz(yaw_from_quat_wxyz(reference_quat))
    return base_pos, base_quat, default_joint_pose()


def repeat_frame(base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray, frames: int):
    if frames <= 0:
        return None
    return (
        np.repeat(base_pos[None, :], frames, axis=0),
        np.repeat(base_quat[None, :], frames, axis=0),
        np.repeat(joint_pos[None, :], frames, axis=0),
    )


def blend_frames(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    start_joint: np.ndarray,
    end_pos: np.ndarray,
    end_quat: np.ndarray,
    end_joint: np.ndarray,
    frames: int,
):
    if frames <= 0:
        return None
    t = smoothstep(np.linspace(0.0, 1.0, frames + 2, dtype=np.float64)[1:-1])
    base_pos = (1.0 - t[:, None]) * start_pos + t[:, None] * end_pos
    joint_pos = (1.0 - t[:, None]) * start_joint + t[:, None] * end_joint
    base_quat = np.stack([quat_slerp(start_quat, end_quat, float(alpha)) for alpha in t], axis=0)
    return base_pos, base_quat, joint_pos


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
    stand_start = make_stand_frame(base_pos[0], base_quat[0], args.stand_height)
    stand_end = make_stand_frame(base_pos[-1], base_quat[-1], args.stand_height)

    pre_hold_frames = seconds_to_frames(args.pre_hold, fps)
    blend_in_frames = seconds_to_frames(args.blend_in, fps)
    blend_out_frames = seconds_to_frames(args.blend_out, fps)
    post_hold_frames = seconds_to_frames(args.post_hold, fps)

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    append_part(parts, repeat_frame(*stand_start, pre_hold_frames))
    append_part(parts, blend_frames(*stand_start, base_pos[0], base_quat[0], joint_pos[0], blend_in_frames))
    append_part(parts, (base_pos, base_quat, joint_pos))
    append_part(parts, blend_frames(base_pos[-1], base_quat[-1], joint_pos[-1], *stand_end, blend_out_frames))
    append_part(parts, repeat_frame(*stand_end, post_hold_frames))

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
        f"(pre_hold={pre_hold_frames}, blend_in={blend_in_frames}, "
        f"blend_out={blend_out_frames}, post_hold={post_hold_frames})"
    )
    print(f"[extend_motion_npz] FPS: {fps:g}, body_order={args.body_order}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add preparation and recovery phases to a T800 motion npz.")
    parser.add_argument("--input", "-i", required=True, help="Input .npz motion file")
    parser.add_argument("--output", "-o", required=True, help="Output .npz motion file")
    parser.add_argument("--pre_hold", type=float, default=0.8, help="Seconds to hold default stand before blending in")
    parser.add_argument("--blend_in", type=float, default=0.5, help="Seconds to blend from stand to first motion frame")
    parser.add_argument("--blend_out", type=float, default=0.8, help="Seconds to blend from last motion frame to stand")
    parser.add_argument("--post_hold", type=float, default=1.0, help="Seconds to hold default stand after blending out")
    parser.add_argument("--stand_height", type=float, default=0.8, help="Base height used for the inserted stand pose")
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
