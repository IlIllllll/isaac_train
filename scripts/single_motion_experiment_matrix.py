#!/usr/bin/env python3
"""Shared matrix for the single-motion optimization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_NAME = "single_motion_optimization"
TARGET_DURATION_S = 3.2


@dataclass(frozen=True)
class Experiment:
    name: str
    task: str
    motion_files: tuple[Path, ...]
    seed: int
    hypothesis: str

    def motion_arg(self) -> str:
        return ",".join(str(path) for path in self.motion_files)


def experiments() -> tuple[Experiment, ...]:
    kick_raw = Path("data/npz/kick_Turn_50hz.npz")
    kick_edge = Path("data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz")
    kick_speed = (
        Path("data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_0p75.npz"),
        Path("data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_1p00.npz"),
        Path("data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_1p25.npz"),
    )
    multi_motion = (
        kick_edge,
        Path("data/augmented_npz/victory_50hz_edge_hold_3p2s.npz"),
        Path("data/augmented_npz/Punch_Swing_L_50hz_edge_hold_3p2s.npz"),
        Path("data/augmented_npz/riot_combo_50hz_edge_hold_3p2s.npz"),
    )

    return (
        Experiment(
            name="E1_baseline_raw_kick_10s_current",
            task="Tracking-Flat-T800-Exp-Current10s-v0",
            motion_files=(kick_raw,),
            seed=1001,
            hypothesis="Raw kick, current 10 s task. Tests baseline looping and edge discontinuity.",
        ),
        Experiment(
            name="E2_edge_hold_kick_3p2s_current",
            task="Tracking-Flat-T800-Exp-Current3p2s-v0",
            motion_files=(kick_edge,),
            seed=1002,
            hypothesis="Adds edge holds and short episode, keeping current done and reward settings.",
        ),
        Experiment(
            name="E3_edge_hold_kick_3p2s_relaxed_done",
            task="Tracking-Flat-T800-Exp-RelaxedDone3p2s-v0",
            motion_files=(kick_edge,),
            seed=1003,
            hypothesis="Tests whether current termination thresholds are too strict.",
        ),
        Experiment(
            name="E4_edge_hold_kick_3p2s_relaxed_done_low_vel_reward",
            task="Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0",
            motion_files=(kick_edge,),
            seed=1004,
            hypothesis="Tests whether body velocity rewards are over-constraining the kick.",
        ),
        Experiment(
            name="E5_multispeed_kick_3p2s_relaxed_done_low_vel_reward",
            task="Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0",
            motion_files=kick_speed,
            seed=1005,
            hypothesis="Tests whether speed diversity reduces single-speed overfitting.",
        ),
        Experiment(
            name="E6_multimotion_relaxed_done_low_vel_reward",
            task="Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0",
            motion_files=multi_motion,
            seed=1006,
            hypothesis="Tests whether a broader motion distribution improves stability.",
        ),
    )


def required_motion_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for experiment in experiments():
        files.extend(experiment.motion_files)
    return tuple(dict.fromkeys(files))
