# Configuration Guide

SONIC uses [Hydra](https://hydra.cc/) for hierarchical configuration. This guide
explains the config structure and the most important parameters to tune.

## Config Hierarchy

When you run a training command like:

```bash
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release
```

Hydra composes the final config from a chain of YAML files:

```
gear_sonic/config/
├── base.yaml                    # Global defaults (seed, num_envs, paths)
├── base/
│   ├── hydra.yaml               # Hydra output directory settings
│   └── structure.yaml           # Resolved experiment directory structure
├── algo/
│   └── ppo_im_phc.yaml          # PPO hyperparameters
├── manager_env/
│   ├── base_env.yaml            # Environment defaults (sim_dt, decimation, episode length)
│   ├── actions/tracking/base.yaml
│   ├── commands/tracking/base.yaml
│   │   └── terms/motion.yaml    # Motion library, body names, future frames
│   ├── rewards/tracking/
│   │   └── base_5point_local_feet_acc.yaml  # Reward composition
│   │       └── terms/*.yaml     # Individual reward terms with weights
│   ├── terminations/tracking/
│   │   └── base_adaptive_strict_ori_foot_xyz.yaml  # Termination composition
│   │       └── terms/*.yaml     # Individual termination conditions
│   ├── events/tracking/
│   │   └── level0_4.yaml        # Domain randomization events
│   └── observations/
│       ├── tokenizer/           # Encoder input observations
│       ├── policy/              # Policy (actor) observations
│       └── critic/              # Critic observations
├── actor_critic/
│   └── universal_token/         # Network architecture (encoders, decoders, quantizer)
├── aux_losses/
│   └── universal_token/         # Auxiliary loss terms
├── trainer/
│   └── trl_ppo_aux.yaml         # Trainer config (PPO with aux losses)
├── callbacks/                   # Training callbacks (save, eval, W&B, resample)
└── exp/manager/universal_token/all_modes/
    └── sonic_release.yaml       # Experiment config (overrides all of the above)
```

The experiment config (`sonic_release.yaml`) sits at the top and overrides
specific values from the base configs. You can further override any value
from the command line with `++key=value`.

## Overriding Config Values

Hydra uses `++` prefix to force-override values (even nested ones):

```bash
# Override a top-level value
python gear_sonic/train_agent_trl.py +exp=... num_envs=16

# Override a nested value (use dots for nesting)
python gear_sonic/train_agent_trl.py +exp=... \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=/path/to/data

# Override a reward weight
python gear_sonic/train_agent_trl.py +exp=... \
    ++manager_env.rewards.tracking_anchor_pos.weight=1.0
```

## Top Parameters to Tune

### Training scale

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `num_envs` | 4096 | `base.yaml` | Number of parallel environments. Reduce for debugging (`16`), increase for throughput. |
| `headless` | True | `base.yaml` | Set `False` to open the Isaac Lab viewer for visual debugging. |
| `seed` | 0 | `base.yaml` | Random seed for reproducibility. |

### PPO hyperparameters

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `algo.config.actor_learning_rate` | 2e-5 | `ppo_im_phc.yaml` | Actor learning rate. Lower for finetuning, higher for training from scratch. |
| `algo.config.critic_learning_rate` | 1e-3 | `ppo_im_phc.yaml` | Critic learning rate. Usually 10-100x the actor LR. |
| `algo.config.num_learning_epochs` | 5 | `ppo_im_phc.yaml` | PPO epochs per batch of experience. |
| `algo.config.num_mini_batches` | 4 | `ppo_im_phc.yaml` | Mini-batches per PPO epoch. |
| `algo.config.num_steps_per_env` | 24 | `sonic_release.yaml` | Rollout length (steps per env before PPO update). |
| `algo.config.gamma` | 0.99 | `ppo_im_phc.yaml` | Discount factor. |
| `algo.config.lam` | 0.95 | `ppo_im_phc.yaml` | GAE lambda. |
| `algo.config.clip_param` | 0.2 | `ppo_im_phc.yaml` | PPO clip parameter. |
| `algo.config.entropy_coef` | 0.01 | `ppo_im_phc.yaml` | Entropy bonus coefficient. |
| `algo.config.desired_kl` | 0.01 | `ppo_im_phc.yaml` | Target KL for adaptive learning rate schedule. |
| `algo.config.num_learning_iterations` | 100000 | `ppo_im_phc.yaml` | Total training iterations. |

### Simulation

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `manager_env.config.sim_dt` | 0.005 | `base_env.yaml` | Physics timestep (200 Hz). Smaller = more stable but slower. |
| `manager_env.config.decimation` | 4 | `base_env.yaml` | Policy runs every `decimation` sim steps (50 Hz policy at 200 Hz sim). |
| `manager_env.config.episode_length_s` | 10.0 | `base_env.yaml` | Episode length in seconds before timeout reset. |
| `manager_env.config.terrain_type` | trimesh | `sonic_release.yaml` | `plane` for flat ground, `trimesh` for rough terrain. |
| `manager_env.config.robot.type` | g1_model_12_dex | `sonic_release.yaml` | Robot type (must match `robot_mapping` in code). |

### Motion data

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `manager_env.commands.motion.motion_lib_cfg.motion_file` | — | `sonic_release.yaml` | Path to retargeted robot motion PKLs. |
| `manager_env.commands.motion.motion_lib_cfg.smpl_motion_file` | — | `sonic_release.yaml` | Path to SMPL motion PKLs (or `dummy`). |
| `manager_env.commands.motion.motion_lib_cfg.soma_motion_file` | — | `sonic_bones_seed.yaml` | Path to SOMA motion PKLs (4-encoder config only). |
| `manager_env.commands.motion.motion_lib_cfg.smpl_y_up` | true | `sonic_release.yaml` | Set `true` if SMPL data uses y-up coordinates. |
| `manager_env.commands.motion.motion_lib_cfg.target_fps` | 50 | `motion.yaml` | Target FPS for motion resampling. |
| `manager_env.commands.motion.motion_lib_cfg.asset.assetFileName` | g1_29dof_rev_1_0.xml | `motion.yaml` | MJCF file for motion library FK. Change for different robots. |

### Motion command

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `manager_env.commands.motion.num_future_frames` | 10 | `sonic_release.yaml` | Number of future reference frames provided to the policy. |
| `manager_env.commands.motion.dt_future_ref_frames` | 0.1 | `sonic_release.yaml` | Time spacing between future frames (seconds). |
| `manager_env.commands.motion.cat_upper_body_poses` | true | `sonic_release.yaml` | Augment lower-body motions with upper-body from different clips. |
| `manager_env.commands.motion.cat_upper_body_poses_prob` | 0.5 | `sonic_release.yaml` | Probability of upper-body augmentation per episode. |
| `manager_env.commands.motion.freeze_frame_aug` | true | `sonic_release.yaml` | Augment with frozen (static) reference frames. |

### Observation history

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `actor_prop_history_length` | 10 | `sonic_release.yaml` | Number of past proprioception frames stacked for actor. |
| `actor_actions_history_length` | 10 | `sonic_release.yaml` | Number of past actions stacked for actor. |
| `critic_prop_history_length` | 10 | `sonic_release.yaml` | Same, for critic. |
| `critic_actions_history_length` | 10 | `sonic_release.yaml` | Same, for critic. |

### Reward weights

All reward terms have a `weight` parameter. Positive weights encourage the behavior,
negative weights penalize it. The default weights for `base_5point_local_feet_acc`:

| Reward term | Weight | Description |
|-------------|--------|-------------|
| `tracking_anchor_pos` | 0.5 | Root position tracking |
| `tracking_anchor_ori` | 0.5 | Root orientation tracking |
| `tracking_relative_body_pos` | 1.0 | Body position tracking (anchor-relative) |
| `tracking_relative_body_ori` | 1.0 | Body orientation tracking (anchor-relative) |
| `tracking_body_linvel` | 1.0 | Body linear velocity tracking |
| `tracking_body_angvel` | 1.0 | Body angular velocity tracking |
| `tracking_vr_5point_local` | 2.0 | 5-point (wrists + head + feet) local tracking |
| `action_rate_l2` | -0.1 | Smooth actions (penalize jerk) |
| `joint_limit` | -10.0 | Stay within joint limits |
| `undesired_contacts` | -0.1 | Penalize non-foot ground contacts |
| `anti_shake_ang_vel` | -0.005 | Penalize wrist/head jitter |
| `feet_acc` | -2.5e-6 | Penalize foot acceleration (smooth stepping) |

Each reward term also has a `std` parameter controlling the Gaussian kernel
sharpness. Smaller `std` = stricter tracking (reward drops faster with error).

Override example:
```bash
++manager_env.rewards.tracking_anchor_pos.weight=2.0
++manager_env.rewards.tracking_anchor_pos.params.std=0.1
```

### Termination thresholds

Terminations end episodes early when tracking error exceeds a threshold. The
adaptive variants use a curriculum that tightens thresholds over training:

| Termination | Threshold | Description |
|-------------|-----------|-------------|
| `anchor_pos` | 0.15 m | Root position deviation |
| `anchor_ori_full` | 0.2 rad | Root orientation deviation |
| `ee_body_pos` | 0.15 m | End-effector position deviation |
| `foot_pos_xyz` | 0.2 m | Foot position deviation |
| `motion_time_out` | — | Episode ends when motion clip finishes |

Looser thresholds (larger values) make training easier initially. The adaptive
terminations automatically tighten as the policy improves.

### Adaptive motion sampling

The motion library supports adaptive sampling — motions the policy fails on are
sampled more frequently:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adaptive_sampling.enable` | true | Enable adaptive sampling. |
| `adaptive_sampling.bin_size` | 50 | Window size for failure rate tracking. |
| `adaptive_sampling.adp_samp_failure_rate_max_over_mean` | 200 | Max/mean failure rate ratio cap. Prevents one hard motion from dominating. |

### Saving and logging

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `algo.config.save_interval` | 500 | `ppo_im_phc.yaml` | Save checkpoint every N iterations. |
| `algo.config.eval_frequency` | 500 | `ppo_im_phc.yaml` | Run evaluation every N iterations. |
| `use_wandb` | false | `base.yaml` | Enable Weights & Biases logging. |
| `base_dir` | logs_rl | `base.yaml` | Root directory for training outputs. |

## Experiment Configs

| Config | Encoders | Use case |
|--------|----------|----------|
| `sonic_release` | G1, teleop, SMPL | Default — matches the released checkpoint |
| `sonic_bones_seed` | G1, teleop, SMPL, SOMA | Extended training with SOMA skeleton encoder |
| `sonic_h2` | G1, teleop, SMPL | H2 robot (31 DOF) |

## Common Recipes

### Debug a training run visually

```bash
python gear_sonic/train_agent_trl.py +exp=... \
    num_envs=4 headless=False \
    algo.config.num_learning_iterations=10
```

### Finetune with lower learning rate

```bash
python gear_sonic/train_agent_trl.py +exp=... \
    +checkpoint=sonic_release/last.pt \
    ++algo.config.actor_learning_rate=5e-6 \
    ++algo.config.desired_kl=0.005
```

### Train on flat ground only

```bash
python gear_sonic/train_agent_trl.py +exp=... \
    ++manager_env.config.terrain_type=plane
```

### Relax termination thresholds for hard motions

```bash
python gear_sonic/train_agent_trl.py +exp=... \
    ++manager_env.terminations.anchor_pos.params.threshold=0.3 \
    ++manager_env.terminations.ee_body_pos.params.threshold=0.3
```

### Increase tracking precision

```bash
python gear_sonic/train_agent_trl.py +exp=... \
    ++manager_env.rewards.tracking_relative_body_pos.params.std=0.1 \
    ++manager_env.rewards.tracking_anchor_pos.params.std=0.1
```
