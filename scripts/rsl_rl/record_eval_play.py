"""Record one policy rollout and print motion-tracking metrics as JSON."""

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _disable_robot_terminations(env_cfg):
    if not hasattr(env_cfg, "terminations") or env_cfg.terminations is None:
        return
    for name in list(vars(env_cfg.terminations).keys()):
        if name != "time_out":
            setattr(env_cfg.terminations, name, None)


def _abspath_motion_files(motion_file: str) -> str:
    return ",".join(os.path.abspath(part.strip()) for part in motion_file.split(",") if part.strip())


parser = argparse.ArgumentParser(description="Record and score an RSL-RL policy rollout.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=160)
parser.add_argument("--video_dir_name", type=str, default="play")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--force_start_zero", type=_str2bool, default=True)
parser.add_argument("--no_terminations", type=_str2bool, default=True)
parser.add_argument("--skip_mnn", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402,F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx  # noqa: E402


def _force_motion_zero(raw_env):
    command = raw_env.command_manager.get_term("motion")
    env_ids = torch.arange(raw_env.num_envs, device=raw_env.device)
    command.motion_ids[env_ids] = 0
    command.time_steps[env_ids] = 0
    command._reset_envs_from_motion(env_ids)
    raw_env.scene.update(raw_env.physics_dt)
    command._refresh_relative_motion_state(env_ids)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.seed is not None:
        random.seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = _abspath_motion_files(args_cli.motion_file)
    if args_cli.no_terminations:
        _disable_robot_terminations(env_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    raw_env = env.unwrapped

    if args_cli.force_start_zero:
        _force_motion_zero(raw_env)

    log_dir = os.path.dirname(resume_path)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", args_cli.video_dir_name),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    onnx_file = os.path.join(export_model_dir, "policy.onnx")
    mnn_file = os.path.join(export_model_dir, "policy.mnn")
    if os.path.exists(onnx_file) and not args_cli.skip_mnn:
        subprocess.run(
            [
                "python",
                "-m",
                "MNN.tools.mnnconvert",
                "-f",
                "ONNX",
                "--modelFile",
                onnx_file,
                "--MNNModel",
                mnn_file,
                "--bizCode",
                "MNN",
            ],
            check=True,
        )
    elif os.path.exists(onnx_file):
        print(f"[INFO] Skipping MNN conversion, ONNX exported at: {onnx_file}")

    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    obs, _ = env.get_observations()
    command = raw_env.command_manager.get_term("motion")
    metric_names = [
        "error_anchor_pos",
        "error_anchor_rot",
        "error_body_pos",
        "error_body_rot",
        "error_joint_pos",
        "error_body_lin_vel",
        "error_body_ang_vel",
    ]
    sums = {name: 0.0 for name in metric_names}
    maxes = {name: 0.0 for name in metric_names}
    early_done_steps = []
    steps = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        for name in metric_names:
            value = float(command.metrics[name].mean().item())
            sums[name] += value
            maxes[name] = max(maxes[name], value)

        if torch.as_tensor(dones).any().item() and steps < args_cli.video_length - 1:
            early_done_steps.append(steps)

        steps += 1
        if args_cli.video and steps >= args_cli.video_length:
            break

    score = {
        "seed": args_cli.seed,
        "steps": steps,
        "video_dir_name": args_cli.video_dir_name,
        "force_start_zero": bool(args_cli.force_start_zero),
        "no_terminations": bool(args_cli.no_terminations),
        "early_done_count": len(early_done_steps),
        "early_done_steps": early_done_steps[:20],
        "mean": {name: sums[name] / max(steps, 1) for name in metric_names},
        "max": maxes,
    }
    print("SCORE_JSON " + json.dumps(score, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
