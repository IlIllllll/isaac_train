# Single Motion Optimization

Archive directory: `experiments/single_motion_optimization_20260430_2019`

This archive is the control point for the six 5000-iteration comparisons. Generated motion npz files and videos are intentionally ignored by Git; matrix and result documents stay in this directory.

## Matrix

| Experiment | Task | Seed | Motion files |
| --- | --- | ---: | --- |
| E1_baseline_raw_kick_10s_current | `Tracking-Flat-T800-Exp-Current10s-v0` | 1001 | `data/npz/kick_Turn_50hz.npz` |
| E2_edge_hold_kick_3p2s_current | `Tracking-Flat-T800-Exp-Current3p2s-v0` | 1002 | `data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz` |
| E3_edge_hold_kick_3p2s_relaxed_done | `Tracking-Flat-T800-Exp-RelaxedDone3p2s-v0` | 1003 | `data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz` |
| E4_edge_hold_kick_3p2s_relaxed_done_low_vel_reward | `Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0` | 1004 | `data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz` |
| E5_multispeed_kick_3p2s_relaxed_done_low_vel_reward | `Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0` | 1005 | `data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_0p75.npz,data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_1p00.npz,data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_1p25.npz` |
| E6_multimotion_relaxed_done_low_vel_reward | `Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0` | 1006 | `data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz,data/augmented_npz/victory_50hz_edge_hold_3p2s.npz,data/augmented_npz/Punch_Swing_L_50hz_edge_hold_3p2s.npz,data/augmented_npz/riot_combo_50hz_edge_hold_3p2s.npz` |

## Workflow

1. Run `python scripts/prepare_single_motion_experiments.py --archive-dir experiments/single_motion_optimization_20260430_2019` to regenerate data and manifests.
2. Run `python scripts/run_single_motion_experiments.py --archive-dir experiments/single_motion_optimization_20260430_2019 --stage smoke --python "~/Desktop/IsaacLab/isaaclab.sh -p"` and check reward values/checkpoint creation.
3. Run `python scripts/run_single_motion_experiments.py --archive-dir experiments/single_motion_optimization_20260430_2019 --stage full --python "~/Desktop/IsaacLab/isaaclab.sh -p"` for 5000 iterations per experiment.
4. Run `python scripts/run_single_motion_experiments.py --archive-dir experiments/single_motion_optimization_20260430_2019 --stage play --python "~/Desktop/IsaacLab/isaaclab.sh -p"` to record both no-termination and normal-termination videos.
5. Fill `results.csv` and `notes.md` with manual video ratings and scalar summaries.
