from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_joint_indexes(command: MotionCommand, joint_names: list[str] | None) -> list[int]:
    if joint_names is None:
        return list(range(command.joint_pos.shape[1]))
    motion_joint_names = command.cfg.motion_joint_names
    if motion_joint_names is None:
        return list(range(command.joint_pos.shape[1]))
    return [i for i, name in enumerate(motion_joint_names) if name in joint_names]


def _phase_window_weight(
    command: MotionCommand,
    phase_start: float,
    phase_end: float,
    ramp_phase: float,
) -> torch.Tensor:
    motion_lengths = command.motion.lengths_for(command.motion_ids).float()
    phase = command.time_steps.float() / torch.clamp(motion_lengths - 1.0, min=1.0)
    if ramp_phase <= 0.0:
        return ((phase >= phase_start) & (phase <= phase_end)).float()
    ramp_up = torch.clamp((phase - phase_start) / ramp_phase, min=0.0, max=1.0)
    ramp_down = torch.clamp((phase_end - phase) / ramp_phase, min=0.0, max=1.0)
    return torch.minimum(ramp_up, ramp_down)


def _projection_error(
    reference: torch.Tensor,
    actual: torch.Tensor,
    min_ref_speed: float,
    lateral_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_speed = torch.norm(reference, dim=-1)
    active = ref_speed >= min_ref_speed
    direction = reference / torch.clamp(ref_speed.unsqueeze(-1), min=1e-6)
    delta = actual - reference
    parallel_error = torch.sum(delta * direction, dim=-1)
    lateral_error = torch.sum(torch.square(delta - parallel_error.unsqueeze(-1) * direction), dim=-1)
    error = torch.square(parallel_error) + lateral_scale * lateral_error
    active_f = active.float()
    active_count = active_f.sum(dim=-1)
    mean_error = (error * active_f).sum(dim=-1) / torch.clamp(active_count, min=1.0)
    return mean_error, active_count > 0.0


def _phase_body_position_error(command: MotionCommand, body_indexes: list[int]) -> torch.Tensor:
    if not body_indexes:
        return torch.zeros(command.num_envs, device=command.device)
    error = torch.norm(
        command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes],
        dim=-1,
    )
    return error.mean(dim=-1)


def _set_metric(command: MotionCommand, name: str | None, value: torch.Tensor) -> None:
    if name:
        command.metrics[name] = value.detach()


def motion_phase_body_lin_vel_projection_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    body_names: list[str],
    ramp_phase: float = 0.04,
    lateral_scale: float = 0.2,
    min_ref_speed: float = 0.8,
    metric_name: str | None = None,
    body_pos_metric_name: str | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    if not body_indexes:
        return torch.zeros(command.num_envs, device=command.device)

    window = _phase_window_weight(command, phase_start, phase_end, ramp_phase)
    error, has_active = _projection_error(
        command.body_lin_vel_w[:, body_indexes],
        command.robot_body_lin_vel_w[:, body_indexes],
        min_ref_speed=min_ref_speed,
        lateral_scale=lateral_scale,
    )
    active_window = window * has_active.float()
    _set_metric(command, metric_name, active_window * torch.sqrt(error))
    _set_metric(command, body_pos_metric_name, active_window * _phase_body_position_error(command, body_indexes))
    return active_window * torch.exp(-error / std**2)


def motion_phase_body_ang_vel_projection_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    body_names: list[str],
    ramp_phase: float = 0.04,
    lateral_scale: float = 0.2,
    min_ref_speed: float = 0.8,
    metric_name: str | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    if not body_indexes:
        return torch.zeros(command.num_envs, device=command.device)

    window = _phase_window_weight(command, phase_start, phase_end, ramp_phase)
    error, has_active = _projection_error(
        command.body_ang_vel_w[:, body_indexes],
        command.robot_body_ang_vel_w[:, body_indexes],
        min_ref_speed=min_ref_speed,
        lateral_scale=lateral_scale,
    )
    active_window = window * has_active.float()
    _set_metric(command, metric_name, active_window * torch.sqrt(error))
    return active_window * torch.exp(-error / std**2)


def motion_phase_joint_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    phase_start: float,
    phase_end: float,
    ramp_phase: float = 0.04,
    joint_names: list[str] | None = None,
    min_ref_speed: float = 0.5,
    metric_name: str | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    if not joint_indexes:
        return torch.zeros(command.num_envs, device=command.device)

    window = _phase_window_weight(command, phase_start, phase_end, ramp_phase)
    reference = command.joint_vel[:, joint_indexes]
    actual = command.robot_joint_vel[:, joint_indexes]
    active = torch.abs(reference) >= min_ref_speed
    active_f = active.float()
    active_count = active_f.sum(dim=-1)
    error = torch.square(actual - reference)
    mean_error = (error * active_f).sum(dim=-1) / torch.clamp(active_count, min=1.0)
    active_window = window * (active_count > 0.0).float()
    _set_metric(command, metric_name, active_window * torch.sqrt(mean_error))
    return active_window * torch.exp(-mean_error / std**2)


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
