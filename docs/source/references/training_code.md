# Training Code Structure

This page describes the Python training codebase under `gear_sonic/`, covering directory layout, the training pipeline, configuration system, key modules, and evaluation scripts.

---

## Directory Layout

```
gear_sonic/
├── train_agent_trl.py          # Main training entry point
├── eval_agent_trl.py           # Single-checkpoint evaluation
├── eval_exp.py                 # Checkpoint monitor (continuous eval)
├── config/                     # Hydra configuration hierarchy
│   ├── base.yaml               # Global defaults (seed, num_envs, paths)
│   ├── base_eval.yaml          # Eval-specific global defaults
│   ├── eval_exp.yaml           # Checkpoint monitor config
│   ├── base/                   # Hydra plumbing (output dirs, resolvers)
│   ├── algo/                   # PPO hyperparameters
│   ├── actor_critic/           # Actor-critic architecture configs
│   │   ├── encoders/           # Per-encoder MLP configs (g1, smpl, teleop)
│   │   ├── decoders/           # Decoder MLP configs (g1_kin, g1_dyn)
│   │   ├── critics/            # Critic backbone configs
│   │   ├── quantizers/         # FSQ quantizer config
│   │   └── universal_token/    # Assembled encoder+decoder+quantizer presets
│   ├── aux_losses/             # Auxiliary loss definitions
│   ├── callbacks/              # Training callback configs
│   ├── exp/                    # Experiment presets (compose all pieces)
│   ├── manager_env/            # Environment MDP component configs
│   ├── opt/                    # Logging options (wandb)
│   └── trainer/                # Trainer class selection
├── envs/                       # IsaacLab environment wrappers
│   ├── manager_env/
│   │   ├── modular_tracking_env_cfg.py   # Scene, sensors, robot articulation
│   │   ├── robots/             # Per-robot configs (g1.py, h2.py)
│   │   └── mdp/                # MDP components (see below)
│   ├── wrapper/
│   │   └── manager_env_wrapper.py  # RL-facing env wrapper
│   └── env_utils/              # Joint ordering utilities
├── trl/                        # Training modules (PPO, actor-critic, losses)
│   ├── trainer/
│   │   ├── ppo_trainer.py          # Base PPO trainer
│   │   └── ppo_trainer_aux_loss.py # PPO + auxiliary losses (SONIC)
│   ├── modules/
│   │   ├── actor_critic_modules.py     # Actor, Critic classes
│   │   ├── universal_token_modules.py  # UniversalTokenModule (SONIC ATM)
│   │   ├── base_module.py              # Shared MLP building blocks
│   │   └── data_utils.py              # Batch/data helpers
│   ├── losses/
│   │   └── token_losses.py     # Reconstruction & latent auxiliary losses
│   ├── callbacks/              # Runtime callbacks
│   │   ├── im_eval_callback.py     # Imitation evaluation metrics
│   │   ├── im_resample_callback.py # Adaptive motion resampling
│   │   ├── model_save_callback.py  # Checkpoint saving
│   │   ├── wandb_callback.py       # W&B logging
│   │   └── read_eval_callback.py   # Read eval results from disk
│   └── utils/                  # Math, rotation, scheduling utilities
├── utils/                      # Shared utilities
│   ├── motion_lib/             # Motion library loading (PKL format)
│   ├── mujoco_sim/             # MuJoCo sim-to-sim bridge
│   └── teleop/                 # VR teleoperation helpers
├── data/                       # Robot models, URDF/USD assets
├── data_process/               # Motion data conversion scripts
└── scripts/                    # MuJoCo sim loop, misc tools
```

---

## Training Pipeline

Running `python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release` executes the following steps:

### 1. Configuration Loading

The entry point uses `@hydra.main(config_path="config", config_name="base")`. The `+exp=...` argument selects an experiment preset that composes all sub-configs:

```
base.yaml                          # Global defaults
  └── +exp=manager/universal_token/all_modes/sonic_release
        ├── /algo: ppo_im_phc      # PPO hyperparameters
        ├── /actor_critic: universal_token/all_mlp_v1
        │     ├── encoders/g1_mf_mlp, smpl_mlp, teleop_mlp
        │     ├── decoders/g1_kin_mf_mlp, g1_dyn_mlp
        │     ├── quantizers/fsq
        │     └── critics/mlp
        ├── /manager_env: base_env  # Environment config
        │     ├── observations/{tokenizer, policy, critic}
        │     ├── rewards/tracking/base_5point_local_feet_acc
        │     ├── terminations/tracking/base_adaptive_strict_ori_foot_xyz
        │     └── events/tracking/level0_4
        ├── /aux_losses: universal_token/g1_recon_and_all_latent
        ├── /trainer: trl_ppo_aux
        └── /callbacks: model_save, wandb, read_eval, im_resample
```

### 2. Simulator and Accelerator Init

After config resolution, the script:
1. Parses TRL `PPOConfig` / `ScriptArguments` / `ModelConfig` from the config dict.
2. Creates a HuggingFace `Accelerator` for multi-GPU support (DDP).
3. Launches the IsaacLab `AppLauncher` to start the Isaac Sim runtime.
4. Saves `config.yaml` and `meta.yaml` to the experiment directory.

### 3. Environment Creation

`create_manager_env()` instantiates the IsaacLab `ManagerBasedRLEnv` from the composed environment config, then wraps it with `ManagerEnvWrapper`:

```
ManagerBasedRLEnv (IsaacLab)
  └── ManagerEnvWrapper
        ├── Observation spaces (policy, critic, tokenizer groups)
        ├── Motion command manager (motion_lib)
        ├── Action transform module (optional, for pretrained ATM)
        └── Keyboard / visualization hooks
```

### 4. Policy and Value Model Creation

The actor and critic are instantiated from the algo config. For SONIC training, the actor backbone is `UniversalTokenModule`:

```python
# Simplified from train_agent_trl.py
policy = custom_instantiate(config.algo.config.actor, env_config=env.config, ...)
value_model = custom_instantiate(config.algo.config.critic, env_config=env.config, ...)
```

The `Actor` wraps `UniversalTokenModule` as its backbone and adds a diagonal Gaussian distribution for exploration. The `Critic` wraps a separate MLP backbone.

### 5. PPO Training Loop

The `TRLAuxLossPPOTrainer.train()` method runs the main loop:

```
for iteration in range(num_learning_iterations):
    # 1. Rollout: collect num_steps_per_env transitions
    for step in range(num_steps_per_env):
        actions = policy.rollout(obs_dict)
        obs_dict, rewards, dones, infos = env.step(actions)
        store(obs, actions, rewards, values, log_probs)

    # 2. GAE: compute advantages and returns
    advantages = generalized_advantage_estimation(rewards, values, dones)

    # 3. PPO update: num_ppo_epochs over mini-batches
    for epoch in range(num_ppo_epochs):
        for mini_batch in shuffle_and_split(rollout_data):
            policy_loss = clipped_surrogate_objective(...)
            value_loss  = clipped_value_loss(...)
            aux_loss    = sum(coef_i * aux_loss_i)  # encoder reconstruction, etc.
            total_loss  = policy_loss + value_loss_coef * value_loss
                        + aux_loss_scale * aux_loss
            optimizer.step(total_loss)

    # 4. Post-update: sync running stats, adaptive sampling, callbacks
    update_scheduled_params(...)     # learning rate, domain randomization
    callbacks.on_step_end(...)       # checkpointing, evaluation, logging
```

---

## Configuration System

The configuration system uses [Hydra](https://hydra.cc/) with config groups and composition.

### Hierarchy

| Level | Path | Purpose |
|---|---|---|
| **Global** | `config/base.yaml` | Seed, num_envs, paths, wandb toggle |
| **Algorithm** | `config/algo/ppo_im_phc.yaml` | PPO hyperparameters, learning rates, epochs |
| **Actor-Critic** | `config/actor_critic/` | Network architecture (encoders, decoders, critic) |
| **Environment** | `config/manager_env/` | Observations, rewards, terminations, events |
| **Auxiliary Losses** | `config/aux_losses/` | Reconstruction and latent alignment losses |
| **Trainer** | `config/trainer/` | Trainer class selection (PPO or PPO+AuxLoss) |
| **Callbacks** | `config/callbacks/` | Checkpointing, evaluation, W&B logging |
| **Experiment** | `config/exp/` | Preset that composes all the above |

### Experiment Presets

Experiment configs live under `config/exp/` and use the `@package _global_` directive to set values at the root level. They compose all component configs via `defaults`:

```yaml
# config/exp/manager/universal_token/all_modes/sonic_release.yaml
defaults:
  - /algo: ppo_im_phc
  - /manager_env: base_env
  - override /actor_critic: universal_token/all_mlp_v1
  - override /manager_env/observations/tokenizer: unitoken_all_noz
  - override /manager_env/observations/policy: local_dir_hist
  - override /manager_env/rewards: tracking/base_5point_local_feet_acc
  - override /manager_env/terminations: tracking/base_adaptive_strict_ori_foot_xyz
  - override /manager_env/events: tracking/level0_4
  # ...
```

### Key Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_envs` | 4096 | Number of parallel simulation environments |
| `algo.config.num_learning_iterations` | 100000 | Total training iterations |
| `algo.config.num_steps_per_env` | 32 | Rollout horizon per iteration |
| `algo.config.num_learning_epochs` | 5 | PPO epochs per iteration |
| `algo.config.num_mini_batches` | 4 | Mini-batches per PPO epoch |
| `algo.config.actor_learning_rate` | 2e-5 | Actor learning rate |
| `algo.config.critic_learning_rate` | 1e-3 | Critic learning rate |
| `algo.config.clip_param` | 0.2 | PPO clipping parameter |
| `algo.config.init_noise_std` | 0.05 | Initial exploration noise std |
| `algo.config.save_interval` | 500 | Checkpoint save frequency (iterations) |

---

## Universal Token Module

The `UniversalTokenModule` implements SONIC's action transform module (ATM) -- the core architecture that maps diverse motion inputs into a shared token space.

### Architecture

```
                  ┌─────────────┐
  G1 obs    ───►  │  G1 Encoder │──┐
                  └─────────────┘  │
                  ┌─────────────┐  │    ┌─────────┐     ┌─────────────┐
  Teleop obs───►  │Teleop Encdr │──┼──► │   FSQ   │──►  │ G1 Dynamic  │──► joint actions
                  └─────────────┘  │    │Quantizer│     │   Decoder   │
                  ┌─────────────┐  │    └─────────┘     └─────────────┘
  SMPL obs  ───►  │ SMPL Encoder│──┘          │
                  └─────────────┘             │         ┌─────────────┐
                                              └───────► │G1 Kinematic │──► (aux loss only)
                                                        │   Decoder   │
                                                        └─────────────┘
```

**Encoders** map different observation modalities into a shared latent space. Each encoder is an MLP that takes modality-specific tokenizer observations and outputs a fixed-size latent vector. During training, one encoder is sampled per environment according to `encoder_sample_probs`.

**FSQ Quantizer** discretizes the continuous latent into a finite set of tokens using Finite Scalar Quantization. Each latent dimension is independently quantized to one of `fsq_level_list` discrete levels. This produces a compact, discrete token representation.

**Decoders** reconstruct outputs from the quantized tokens plus proprioception:
- **G1 Dynamic Decoder** (`g1_dyn`): Produces joint-space actions fed to the actuators. This is the only decoder used at deployment time.
- **G1 Kinematic Decoder** (`g1_kin`): Reconstructs future motion frames from tokens. Used only during training to compute reconstruction auxiliary losses.

### Latent Residual Mode

For downstream tasks (e.g., object manipulation), an external policy can inject corrections into the token space without retraining the base ATM:

| Mode | Behavior |
|---|---|
| `post_quantization` (default) | Residual added after FSQ quantization |
| `pre_quantization` | Residual added before FSQ; the sum gets quantized |
| `pre_quantization_replace` | Latent is replaced entirely by the residual |

### Encoder Sampling

During training, each environment is randomly assigned an encoder per episode according to `encoder_sample_probs`. The `encoder_index` observation tells the module which encoder produced the current token. At deployment, only one encoder is active (selected by the observation configuration).

---

## Environment Structure

The training environment is built on IsaacLab's `ManagerBasedRLEnv` and uses a modular MDP design where each component is configured independently via YAML.

### MDP Components

All MDP components live in `gear_sonic/envs/manager_env/mdp/`:

| Module | Config path | Description |
|---|---|---|
| `observations.py` | `config/manager_env/observations/` | Observation terms for policy, critic, and tokenizer groups |
| `actions.py` | `config/manager_env/actions/` | Joint position action space |
| `rewards.py` | `config/manager_env/rewards/` | Reward terms (tracking, regularization) |
| `terminations.py` | `config/manager_env/terminations/` | Episode termination conditions |
| `events.py` | `config/manager_env/events/` | Domain randomization events |
| `commands.py` | `config/manager_env/commands/` | Motion command generation (motion library) |
| `curriculum.py` | `config/manager_env/curriculum/` | Curriculum schedules |
| `terrain.py` | (inline) | Terrain generation |
| `recorders.py` | `config/manager_env/recorders/` | Video recording |

### Observation Groups

Observations are split into groups, each with its own config file:

| Group | Purpose | Example terms |
|---|---|---|
| **policy** | Direct input to the policy MLP | joint_pos, joint_vel, base_ang_vel, gravity_dir, last_actions |
| **critic** | Privileged observations for the value function | All policy obs + base_lin_vel, body_pos, body_ori |
| **tokenizer** | Input to the UniversalTokenModule encoders | Multi-future joint commands, SMPL joints, VR targets, anchor orientations |

### Reward Terms

Reward configs compose individual terms from `config/manager_env/rewards/terms/`. Key tracking rewards:

| Term | Description |
|---|---|
| `tracking_relative_body_pos` | Track reference body positions (5-point: root, wrists, feet) |
| `tracking_relative_body_ori` | Track reference body orientations |
| `tracking_anchor_pos` | Track root anchor position |
| `tracking_anchor_ori` | Track root anchor orientation |
| `tracking_body_linvel` | Track reference body linear velocities |
| `tracking_body_angvel` | Track reference body angular velocities |
| `action_rate_l2` | Penalize action jerk |
| `feet_acc` | Penalize foot acceleration (smoothness) |

### ManagerEnvWrapper

`ManagerEnvWrapper` bridges the IsaacLab environment with the RL training loop. It handles:
- Flattening observation dicts for the policy
- Applying the optional pretrained action transform module
- Motion replay mode
- Debug visualization and keyboard controls

---

## Evaluation Scripts

### eval_agent_trl.py -- Single Checkpoint

Loads a single checkpoint and runs evaluation in Isaac Sim. Automatically reads the training `config.yaml` from the checkpoint directory to reconstruct the full configuration.

```bash
# Interactive visualization
python gear_sonic/eval_agent_trl.py +checkpoint=path/to/model.pt +headless=False ++num_envs=1

# Headless with video rendering
python gear_sonic/eval_agent_trl.py +checkpoint=path/to/model.pt +headless=True \
    ++num_envs=16 +run_once=True \
    ++manager_env.config.save_rendering_dir=path/to/output \
    ++manager_env.config.render_results=True \
    +manager_env/recorders=render
```

Key features:
- Merges training config with eval overrides (`eval_overrides` in config)
- Removes train-only events and terminations automatically
- Supports `+run_once=True` to exit after all environments complete one episode
- Handles `+metrics_file` to render worst-performing motions from a prior eval

### eval_exp.py -- Checkpoint Monitor

`CheckpointEvaluator` continuously monitors an experiment directory for new checkpoints and evaluates them sequentially. It runs as a companion process alongside training.

```bash
python gear_sonic/eval_exp.py ++experiment_dir=path/to/experiment
```

For each new checkpoint, it:
1. Runs metrics evaluation (launches `eval_agent_trl.py` via subprocess)
2. Runs video rendering for the hardest motions
3. Logs results and videos to W&B (resuming the training run)
4. Marks each checkpoint as evaluated to avoid redundant work

Configuration (`config/eval_exp.yaml`):

| Parameter | Description |
|---|---|
| `experiment_dir` | Path to the training experiment directory |
| `scan_interval` | Seconds between checkpoint scans (default: 60) |
| `num_eval_envs` | Number of environments for metric evaluation |
| `num_render_videos` | Number of videos to render per checkpoint |
| `eval_frequency` | Only evaluate every N-th checkpoint (default: all) |
| `single_pass` | Evaluate pending checkpoints once and exit |

---

## Key Classes Reference

| Class | Module | Description |
|---|---|---|
| `Actor` | `trl/modules/actor_critic_modules.py` | Policy network: backbone + diagonal Gaussian. Maintains observation buffer for temporal models. |
| `Critic` | `trl/modules/actor_critic_modules.py` | Value function network: backbone + scalar output. Supports running mean/std normalization. |
| `UniversalTokenModule` | `trl/modules/universal_token_modules.py` | SONIC ATM: multi-encoder, FSQ quantizer, multi-decoder. Computes auxiliary reconstruction losses. |
| `TRLPPOTrainer` | `trl/trainer/ppo_trainer.py` | Base PPO trainer adapted from HuggingFace TRL. Handles rollout collection, GAE, and gradient updates. |
| `TRLAuxLossPPOTrainer` | `trl/trainer/ppo_trainer_aux_loss.py` | Extends `TRLPPOTrainer` with auxiliary loss support (reconstruction, latent alignment). |
| `PolicyAndValueWrapper` | `trl/trainer/ppo_trainer.py` | Wraps policy + value model into a single `nn.Module` for DDP-safe forward passes. |
| `ManagerEnvWrapper` | `envs/wrapper/manager_env_wrapper.py` | Bridges IsaacLab `ManagerBasedRLEnv` with the training loop. Handles obs flattening, action transforms, replay. |
| `CheckpointEvaluator` | `eval_exp.py` | Monitors experiment directory, evaluates new checkpoints, logs to W&B. |
