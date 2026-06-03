from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import t800_mdp
from whole_body_tracking.robots.t800 import T800_ACTION_SCALE, T800_CFG
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.t800.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class T800FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos = t800_mdp.ResidualRefJointPositionActionCfg(
            asset_name="robot",
            joint_names=t800_mdp.T800_DFS_JOINT_NAMES,
            command_name="motion",
            preserve_order=True,
        )
        self.actions.joint_pos.scale = T800_ACTION_SCALE
        dfs_joint_asset_cfg = SceneEntityCfg("robot", joint_names=t800_mdp.T800_DFS_JOINT_NAMES, preserve_order=True)
        self.observations.policy.joint_pos.params = {"asset_cfg": dfs_joint_asset_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": dfs_joint_asset_cfg}
        self.observations.critic.joint_pos.params = {"asset_cfg": dfs_joint_asset_cfg}
        self.observations.critic.joint_vel.params = {"asset_cfg": dfs_joint_asset_cfg}
        self.commands.motion.anchor_body_name = "LINK_BASE"
        self.commands.motion.motion_joint_names = t800_mdp.T800_DFS_JOINT_NAMES
        self.commands.motion.motion_body_names = t800_mdp.T800_MOTION_BODY_NAMES
        self.commands.motion.min_traj_duration = self.episode_length_s
        self.commands.motion.bridge_frames = 20
        self.commands.motion.pd_stand_reset_ratio = 0.2
        self.commands.motion.body_names = [
            "LINK_BASE",
            "LINK_HIP_ROLL_L",
            "LINK_KNEE_PITCH_L",
            "LINK_ANKLE_ROLL_L",
            "LINK_HIP_ROLL_R",
            "LINK_KNEE_PITCH_R",
            "LINK_ANKLE_ROLL_R",
            "LINK_TORSO_YAW",
            "LINK_SHOULDER_PITCH_L",
            "LINK_ELBOW_PITCH_L",
            "LINK_ELBOW_YAW_L",
            "LINK_SHOULDER_PITCH_R",
            "LINK_ELBOW_PITCH_R",
            "LINK_ELBOW_YAW_R",
            "LINK_HEAD_PITCH",
            "LINK_HEAD_YAW",
        ]
        self.events.base_com.params["asset_cfg"].body_names = "LINK_BASE"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$)(?!LINK_ELBOW_YAW_L$)(?!LINK_ELBOW_YAW_R$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_ELBOW_YAW_L",
            "LINK_ELBOW_YAW_R",
        ]


@configclass
class T800FlatSingleMotionCurrent10sEnvCfg(T800FlatEnvCfg):
    """Experiment copy of the current T800 flat task."""

    pass


@configclass
class T800FlatSingleMotionCurrent3p2sEnvCfg(T800FlatEnvCfg):
    """Current rewards and terminations with a 3.2 s episode for short single motions."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 3.2
        self.commands.motion.min_traj_duration = self.episode_length_s


@configclass
class T800FlatSingleMotionRelaxedDone3p2sEnvCfg(T800FlatSingleMotionCurrent3p2sEnvCfg):
    """Short-motion task with looser fall/out-of-track termination thresholds."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.anchor_pos.params["threshold"] = 0.50
        self.terminations.anchor_ori.params["threshold"] = 1.20
        self.terminations.ee_body_pos.params["threshold"] = 0.50


@configclass
class T800FlatSingleMotionRelaxedDoneLowVel3p2sEnvCfg(T800FlatSingleMotionRelaxedDone3p2sEnvCfg):
    """Relaxed short-motion task with reduced body velocity reward pressure."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_body_lin_vel.weight = 0.25
        self.rewards.motion_body_ang_vel.weight = 0.25


@configclass
class T800FlatHighDynBase3p2sEnvCfg(T800FlatEnvCfg):
    """Base task for high-dynamic 3.2s single-motion tracking."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 3.2
        self.commands.motion.min_traj_duration = self.episode_length_s
        self.commands.motion.bridge_frames = 20
        self.commands.motion.adaptive_kernel_size = 5
        self.commands.motion.adaptive_uniform_ratio = 0.30
        self.commands.motion.adaptive_alpha = 0.001
        self.commands.motion.pd_stand_reset_ratio = 0.0
        self.events.push_robot = None
        self.rewards.motion_body_lin_vel.weight = 0.10
        self.rewards.motion_body_ang_vel.weight = 0.10
        self.rewards.action_rate_l2.weight = -0.03


@configclass
class T800FlatHighDynWarmup3p2sEnvCfg(T800FlatHighDynBase3p2sEnvCfg):
    """Warmup task: follow the full high-dynamic motion without early tracking terminations."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.pd_stand_reset_ratio = 0.0
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None


@configclass
class T800FlatHighDynStable3p2sEnvCfg(T800FlatHighDynBase3p2sEnvCfg):
    """Stable task: restore loose tracking terminations and moderate velocity rewards."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.pd_stand_reset_ratio = 0.05
        self.terminations.anchor_pos.params["threshold"] = 0.70
        self.terminations.anchor_ori.params["threshold"] = 1.40
        self.terminations.ee_body_pos.params["threshold"] = 0.80
        self.rewards.motion_body_lin_vel.weight = 0.25
        self.rewards.motion_body_ang_vel.weight = 0.25
        self.rewards.action_rate_l2.weight = -0.05
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.4)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.1)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.005, 0.005)
        self.events.base_com.params["com_range"] = {
            "x": (-0.010, 0.010),
            "y": (-0.020, 0.020),
            "z": (-0.020, 0.020),
        }


@configclass
class T800FlatHighDynFinal3p2sEnvCfg(T800FlatHighDynStable3p2sEnvCfg):
    """Final task: tighter tracking terminations and stronger velocity tracking."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.anchor_pos.params["threshold"] = 0.50
        self.terminations.anchor_ori.params["threshold"] = 1.20
        self.terminations.ee_body_pos.params["threshold"] = 0.60
        self.rewards.motion_body_lin_vel.weight = 0.50
        self.rewards.motion_body_ang_vel.weight = 0.50
        self.rewards.action_rate_l2.weight = -0.08


@configclass
class T800FlatHighDynSpeedLead3p2sEnvCfg(T800FlatHighDynFinal3p2sEnvCfg):
    """High-dynamic task tuned for speed catch-up with lead reference targets."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.command_lookahead_steps = (0,)
        self.actions.joint_pos.ref_advance_steps = 1
        self.commands.motion.pd_stand_reset_ratio = 0.05
        self.terminations.anchor_pos.params["threshold"] = 0.70
        self.terminations.anchor_ori.params["threshold"] = 1.40
        self.terminations.ee_body_pos.params["threshold"] = 0.80
        self.rewards.motion_body_lin_vel.weight = 0.80
        self.rewards.motion_body_lin_vel.params["std"] = 1.50
        self.rewards.motion_body_ang_vel.weight = 0.80
        self.rewards.motion_body_ang_vel.params["std"] = 4.00
        self.rewards.action_rate_l2.weight = -0.04

        ee_body_names = [
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_ELBOW_YAW_L",
            "LINK_ELBOW_YAW_R",
        ]
        self.rewards.motion_ee_lin_vel = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=0.60,
            params={"command_name": "motion", "std": 1.00, "body_names": ee_body_names},
        )
        self.rewards.motion_ee_ang_vel = RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=0.40,
            params={"command_name": "motion", "std": 3.14, "body_names": ee_body_names},
        )


KICK_PHASE_RANGE = (0.29, 0.64)
RIOT_PHASE_RANGE = (0.21, 0.58)
PHASE_RAMP = 0.04

KICK_PHASE_LEG_BODIES = ["LINK_ANKLE_ROLL_R", "LINK_KNEE_PITCH_R"]
KICK_PHASE_ARM_BODIES = ["LINK_ELBOW_YAW_R", "LINK_ELBOW_YAW_L"]
RIOT_PHASE_ARM_BODIES = ["LINK_ELBOW_YAW_L", "LINK_ELBOW_YAW_R"]
RIOT_PHASE_LEG_BODIES = ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]
RIOT_PHASE_TORSO_BODIES = ["LINK_TORSO_YAW"]

KICK_PHASE_JOINTS = [
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
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
]

RIOT_PHASE_JOINTS = [
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
]


@configclass
class T800FlatHighDynPhaseLocalBase3p2sEnvCfg(T800FlatHighDynSpeedLead3p2sEnvCfg):
    """High-dynamic speed-lead task with phase-gated local velocity rewards."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_body_lin_vel.weight = 0.45
        self.rewards.motion_body_lin_vel.params["std"] = 1.50
        self.rewards.motion_body_ang_vel.weight = 0.45
        self.rewards.motion_body_ang_vel.params["std"] = 4.00
        self.rewards.motion_ee_lin_vel.weight = 0.30
        self.rewards.motion_ee_ang_vel.weight = 0.20
        self.rewards.action_rate_l2.weight = -0.03


@configclass
class T800FlatHighDynPhaseLocalKick3p2sEnvCfg(T800FlatHighDynPhaseLocalBase3p2sEnvCfg):
    """Phase-local reward task for kick_Turn."""

    def __post_init__(self):
        super().__post_init__()
        phase_start, phase_end = KICK_PHASE_RANGE
        self.actions.joint_pos.ref_advance_steps = 2

        self.rewards.phase_leg_lin_vel_projection = RewTerm(
            func=mdp.motion_phase_body_lin_vel_projection_exp,
            weight=1.20,
            params={
                "command_name": "motion",
                "std": 0.80,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "body_names": KICK_PHASE_LEG_BODIES,
                "lateral_scale": 0.2,
                "min_ref_speed": 0.8,
                "metric_name": "phase_fast/error_local_lin_vel",
                "body_pos_metric_name": "phase_fast/error_local_body_pos",
            },
        )
        self.rewards.phase_leg_ang_vel_projection = RewTerm(
            func=mdp.motion_phase_body_ang_vel_projection_exp,
            weight=0.60,
            params={
                "command_name": "motion",
                "std": 3.00,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "body_names": KICK_PHASE_LEG_BODIES,
                "lateral_scale": 0.2,
                "min_ref_speed": 0.8,
                "metric_name": "phase_fast/error_local_ang_vel",
            },
        )
        self.rewards.phase_arm_lin_vel_projection = RewTerm(
            func=mdp.motion_phase_body_lin_vel_projection_exp,
            weight=0.45,
            params={
                "command_name": "motion",
                "std": 1.00,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "body_names": KICK_PHASE_ARM_BODIES,
                "lateral_scale": 0.2,
                "min_ref_speed": 0.8,
                "metric_name": "phase_fast/error_aux_lin_vel",
            },
        )
        self.rewards.phase_joint_vel = RewTerm(
            func=mdp.motion_phase_joint_velocity_error_exp,
            weight=0.35,
            params={
                "command_name": "motion",
                "std": 6.00,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "joint_names": KICK_PHASE_JOINTS,
                "min_ref_speed": 0.5,
                "metric_name": "phase_fast/error_joint_vel",
            },
        )


@configclass
class T800FlatHighDynPhaseLocalRiot3p2sEnvCfg(T800FlatHighDynPhaseLocalBase3p2sEnvCfg):
    """Phase-local reward task for riot_combo."""

    def __post_init__(self):
        super().__post_init__()
        phase_start, phase_end = RIOT_PHASE_RANGE
        self.actions.joint_pos.ref_advance_steps = 1

        self.rewards.phase_arm_lin_vel_projection = RewTerm(
            func=mdp.motion_phase_body_lin_vel_projection_exp,
            weight=1.10,
            params={
                "command_name": "motion",
                "std": 0.85,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "body_names": RIOT_PHASE_ARM_BODIES,
                "lateral_scale": 0.2,
                "min_ref_speed": 0.8,
                "metric_name": "phase_fast/error_local_lin_vel",
                "body_pos_metric_name": "phase_fast/error_local_body_pos",
            },
        )
        self.rewards.phase_arm_ang_vel_projection = RewTerm(
            func=mdp.motion_phase_body_ang_vel_projection_exp,
            weight=0.70,
            params={
                "command_name": "motion",
                "std": 3.50,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "body_names": RIOT_PHASE_ARM_BODIES,
                "lateral_scale": 0.2,
                "min_ref_speed": 0.8,
                "metric_name": "phase_fast/error_local_ang_vel",
            },
        )
        self.rewards.phase_leg_lin_vel_projection = RewTerm(
            func=mdp.motion_phase_body_lin_vel_projection_exp,
            weight=0.55,
            params={
                "command_name": "motion",
                "std": 1.00,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "body_names": RIOT_PHASE_LEG_BODIES,
                "lateral_scale": 0.2,
                "min_ref_speed": 0.8,
                "metric_name": "phase_fast/error_aux_lin_vel",
            },
        )
        self.rewards.phase_torso_ang_vel_projection = RewTerm(
            func=mdp.motion_phase_body_ang_vel_projection_exp,
            weight=0.35,
            params={
                "command_name": "motion",
                "std": 2.50,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "body_names": RIOT_PHASE_TORSO_BODIES,
                "lateral_scale": 0.2,
                "min_ref_speed": 0.8,
                "metric_name": "phase_fast/error_aux_ang_vel",
            },
        )
        self.rewards.phase_joint_vel = RewTerm(
            func=mdp.motion_phase_joint_velocity_error_exp,
            weight=0.35,
            params={
                "command_name": "motion",
                "std": 7.00,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "ramp_phase": PHASE_RAMP,
                "joint_names": RIOT_PHASE_JOINTS,
                "min_ref_speed": 0.5,
                "metric_name": "phase_fast/error_joint_vel",
            },
        )


@configclass
class T800FlatWoStateEstimationEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class T800FlatLowFreqEnvCfg(T800FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
