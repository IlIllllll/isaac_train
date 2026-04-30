"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python scripts/replay_npz.py --input_file /path/to/motion.npz --robot t800
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--input_file", type=str, default=None, help="Path to a local .npz motion file.")
parser.add_argument("--robot", type=str, default="t800", choices=["pm01", "t800"], help="Robot type to use.")
parser.add_argument("--video_file", type=str, default=None, help="Path to save an mp4 recording of the replay.")
parser.add_argument("--video_length", type=int, default=None, help="Number of rendered frames to record.")
parser.add_argument("--video_fps", type=int, default=50, help="Output video fps.")
parser.add_argument("--video_width", type=int, default=1280, help="Output video width.")
parser.add_argument("--video_height", type=int, default=720, help="Output video height.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video_file is not None:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.pm01 import PM01_CYLINDER_CFG
from whole_body_tracking.robots.t800 import T800_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader

ROBOT_CFGS = {
    "pm01": PM01_CYLINDER_CFG,
    "t800": T800_CFG,
}

T800_MOTION_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
    "J27_HEAD_PITCH",
    "J28_HEAD_YAW",
]

ROBOT_MOTION_JOINT_NAMES = {
    "t800": T800_MOTION_JOINT_NAMES,
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation (will be overridden in main based on --robot)
    robot: ArticulationCfg = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _create_recording_camera() -> Camera:
    camera_cfg = CameraCfg(
        prim_path="/World/ReplayCamera",
        update_period=0,
        height=args_cli.video_height,
        width=args_cli.video_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
    )
    return Camera(camera_cfg)


def _open_video_writer(video_file: str, fps: int):
    os.makedirs(os.path.dirname(os.path.abspath(video_file)), exist_ok=True)
    import imageio.v2 as imageio

    return imageio.get_writer(video_file, fps=fps, codec="libx264", quality=8, macro_block_size=16)


def _camera_rgb_frame(camera: Camera) -> np.ndarray:
    frame = camera.data.output["rgb"][0].detach().cpu().numpy()
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, camera: Camera | None = None):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if args_cli.input_file is not None:
        motion_file = args_cli.input_file
    elif args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        raise ValueError("Either --input_file or --registry_name must be provided.")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    motion_joint_names = ROBOT_MOTION_JOINT_NAMES.get(args_cli.robot)
    robot_joint_indexes = None
    if motion_joint_names is not None:
        robot_joint_indexes = robot.find_joints(motion_joint_names, preserve_order=True)[0]
        if len(robot_joint_indexes) != motion.joint_pos.shape[1]:
            raise ValueError(
                f"Motion joint count ({motion.joint_pos.shape[1]}) does not match mapped robot joints "
                f"({len(robot_joint_indexes)}) for robot '{args_cli.robot}'."
        )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    video_writer = None
    max_recorded_frames = args_cli.video_length
    camera_eye_offset = torch.tensor([2.4, 2.4, 1.2], dtype=torch.float32, device=sim.device)
    camera_target_offset = torch.tensor([0.0, 0.0, 0.45], dtype=torch.float32, device=sim.device)

    if args_cli.video_file is not None:
        if camera is None:
            raise RuntimeError("A recording camera is required when --video_file is set.")
        max_recorded_frames = max_recorded_frames or motion.time_step_total
        video_writer = _open_video_writer(args_cli.video_file, args_cli.video_fps)
        print(
            f"[INFO] Recording {max_recorded_frames} frames to {args_cli.video_file} "
            f"at {args_cli.video_fps} fps ({args_cli.video_width}x{args_cli.video_height})."
        )

    # Simulation loop
    recorded_frames = 0
    try:
        while simulation_app.is_running():
            time_steps += 1
            reset_ids = time_steps >= motion.time_step_total
            time_steps[reset_ids] = 0

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
            root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

            robot.write_root_state_to_sim(root_states)
            if robot_joint_indexes is None:
                robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
            else:
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                joint_pos[:, robot_joint_indexes] = motion.joint_pos[time_steps]
                joint_vel[:, robot_joint_indexes] = motion.joint_vel[time_steps]
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.write_data_to_sim()

            lookat = root_states[0, :3] + camera_target_offset
            eye = lookat + camera_eye_offset
            if camera is not None:
                camera.set_world_poses_from_view(eye.unsqueeze(0), lookat.unsqueeze(0))
            sim.set_camera_view(eye.cpu().numpy(), lookat.cpu().numpy())
            sim.render()  # We don't want physics (sim.step()).
            scene.update(sim_dt)

            if video_writer is not None and camera is not None:
                camera.update(sim_dt, force_recompute=True)
                video_writer.append_data(_camera_rgb_frame(camera))
                recorded_frames += 1

                if recorded_frames >= max_recorded_frames:
                    break
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"[INFO] Wrote video: {args_cli.video_file}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    camera = _create_recording_camera() if args_cli.video_file is not None else None
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene, camera)


if __name__ == "__main__":
    # run the main function
    main()
    # Isaac Sim/Replicator may hang during headless camera shutdown after the mp4 is already flushed.
    if args_cli.video_file is not None:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    # close sim app
    simulation_app.close()
