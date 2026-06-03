# Single Motion Optimization Commands

Run from the repository root.

If the remote IsaacLab install needs the launcher wrapper, pass `--python "~/Desktop/IsaacLab/isaaclab.sh -p"` to `scripts/run_single_motion_experiments.py`.

```bash
python scripts/prepare_single_motion_experiments.py --archive-dir experiments/<archive-name>
```

## E1_baseline_raw_kick_10s_current

Smoke test:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-Current10s-v0 --motion_file data/npz/kick_Turn_50hz.npz --num_envs 128 --max_iterations 200 --seed 1001 --experiment_name single_motion_optimization --run_name E1_baseline_raw_kick_10s_current_smoke200 --headless
```

Full run:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-Current10s-v0 --motion_file data/npz/kick_Turn_50hz.npz --num_envs 128 --max_iterations 5000 --seed 1001 --experiment_name single_motion_optimization --run_name E1_baseline_raw_kick_10s_current --headless
```

Play with and without terminations after the checkpoint exists:

```bash
python scripts/run_single_motion_experiments.py --archive-dir experiments/<archive-name> --stage play --only E1_baseline_raw_kick_10s_current
```

## E2_edge_hold_kick_3p2s_current

Smoke test:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-Current3p2s-v0 --motion_file data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz --num_envs 128 --max_iterations 200 --seed 1002 --experiment_name single_motion_optimization --run_name E2_edge_hold_kick_3p2s_current_smoke200 --headless
```

Full run:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-Current3p2s-v0 --motion_file data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz --num_envs 128 --max_iterations 5000 --seed 1002 --experiment_name single_motion_optimization --run_name E2_edge_hold_kick_3p2s_current --headless
```

Play with and without terminations after the checkpoint exists:

```bash
python scripts/run_single_motion_experiments.py --archive-dir experiments/<archive-name> --stage play --only E2_edge_hold_kick_3p2s_current
```

## E3_edge_hold_kick_3p2s_relaxed_done

Smoke test:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-RelaxedDone3p2s-v0 --motion_file data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz --num_envs 128 --max_iterations 200 --seed 1003 --experiment_name single_motion_optimization --run_name E3_edge_hold_kick_3p2s_relaxed_done_smoke200 --headless
```

Full run:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-RelaxedDone3p2s-v0 --motion_file data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz --num_envs 128 --max_iterations 5000 --seed 1003 --experiment_name single_motion_optimization --run_name E3_edge_hold_kick_3p2s_relaxed_done --headless
```

Play with and without terminations after the checkpoint exists:

```bash
python scripts/run_single_motion_experiments.py --archive-dir experiments/<archive-name> --stage play --only E3_edge_hold_kick_3p2s_relaxed_done
```

## E4_edge_hold_kick_3p2s_relaxed_done_low_vel_reward

Smoke test:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0 --motion_file data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz --num_envs 128 --max_iterations 200 --seed 1004 --experiment_name single_motion_optimization --run_name E4_edge_hold_kick_3p2s_relaxed_done_low_vel_reward_smoke200 --headless
```

Full run:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0 --motion_file data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz --num_envs 128 --max_iterations 5000 --seed 1004 --experiment_name single_motion_optimization --run_name E4_edge_hold_kick_3p2s_relaxed_done_low_vel_reward --headless
```

Play with and without terminations after the checkpoint exists:

```bash
python scripts/run_single_motion_experiments.py --archive-dir experiments/<archive-name> --stage play --only E4_edge_hold_kick_3p2s_relaxed_done_low_vel_reward
```

## E5_multispeed_kick_3p2s_relaxed_done_low_vel_reward

Smoke test:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0 --motion_file data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_0p75.npz,data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_1p00.npz,data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_1p25.npz --num_envs 128 --max_iterations 200 --seed 1005 --experiment_name single_motion_optimization --run_name E5_multispeed_kick_3p2s_relaxed_done_low_vel_reward_smoke200 --headless
```

Full run:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0 --motion_file data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_0p75.npz,data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_1p00.npz,data/speed_augmented_npz/kick_Turn_50hz_edge_hold_3p2s_speed_1p25.npz --num_envs 128 --max_iterations 5000 --seed 1005 --experiment_name single_motion_optimization --run_name E5_multispeed_kick_3p2s_relaxed_done_low_vel_reward --headless
```

Play with and without terminations after the checkpoint exists:

```bash
python scripts/run_single_motion_experiments.py --archive-dir experiments/<archive-name> --stage play --only E5_multispeed_kick_3p2s_relaxed_done_low_vel_reward
```

## E6_multimotion_relaxed_done_low_vel_reward

Smoke test:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0 --motion_file data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz,data/augmented_npz/victory_50hz_edge_hold_3p2s.npz,data/augmented_npz/Punch_Swing_L_50hz_edge_hold_3p2s.npz,data/augmented_npz/riot_combo_50hz_edge_hold_3p2s.npz --num_envs 128 --max_iterations 200 --seed 1006 --experiment_name single_motion_optimization --run_name E6_multimotion_relaxed_done_low_vel_reward_smoke200 --headless
```

Full run:

```bash
python scripts/rsl_rl/train.py --task Tracking-Flat-T800-Exp-RelaxedDoneLowVel3p2s-v0 --motion_file data/augmented_npz/kick_Turn_50hz_edge_hold_3p2s.npz,data/augmented_npz/victory_50hz_edge_hold_3p2s.npz,data/augmented_npz/Punch_Swing_L_50hz_edge_hold_3p2s.npz,data/augmented_npz/riot_combo_50hz_edge_hold_3p2s.npz --num_envs 128 --max_iterations 5000 --seed 1006 --experiment_name single_motion_optimization --run_name E6_multimotion_relaxed_done_low_vel_reward --headless
```

Play with and without terminations after the checkpoint exists:

```bash
python scripts/run_single_motion_experiments.py --archive-dir experiments/<archive-name> --stage play --only E6_multimotion_relaxed_done_low_vel_reward
```
