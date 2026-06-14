# Training Guide

This guide covers data processing, training, evaluation, and ONNX export for
SONIC whole-body controllers.

## Overview

SONIC uses a universal-token architecture to control a humanoid robot (Unitree
G1, 29 DOF) by imitating human motion capture data. Multiple parallel encoders
accept different motion input formats:

- **G1**: Robot joint trajectories
- **Teleop**: VR 3-point tracking targets (head + two wrists)
- **SMPL**: Parametric human body model joint positions
- **SOMA**: BVH-derived skeleton joint positions (optional 4th encoder)

All encoders project into a shared latent token space via FSQ (Finite Scalar
Quantization), and a single decoder produces joint actions regardless of input
modality. Training uses PPO with auxiliary losses in Isaac Lab simulation.

| Config | Encoders | Use case |
|--------|----------|----------|
| `sonic_release` | G1, teleop, SMPL | **Default** — matches the released checkpoint |
| `sonic_bones_seed` | G1, teleop, SMPL, SOMA | Extended training with SOMA skeleton encoder |

Use `sonic_release` for finetuning and evaluation. The `sonic_bones_seed`
config adds a fourth SOMA encoder (see [Training with SOMA](#training-with-soma-encoder)).

## Data Processing

### Step 1: Convert motion data

SONIC requires motion data in **motion_lib PKL format**. Convert Bones-SEED
CSV files:

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input /path/to/bones_seed/g1/csv/ \
    --output data/motion_lib_bones_seed/robot \
    --fps 30 \
    --fps_source 120 \
    --individual \
    --num_workers 16
```

### Step 2: Filter motions

Remove motions the G1 robot cannot perform (furniture interaction, vehicles,
acrobatics, elevated surfaces):

```bash
python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source data/motion_lib_bones_seed/robot \
    --dest data/motion_lib_bones_seed/robot_filtered \
    --workers 16
```

This removes ~8.7% of motions (~130K of 142K remain). Use `--dry-run` to
preview, or `--add-keywords` to add custom filters.

### Data layout

Place processed data at the repo root:

```
<repo_root>/
├── data/motion_lib_bones_seed/
│   ├── robot/              # Full motion library (142K PKLs)
│   └── robot_filtered/     # Filtered subset (~130K PKLs)
└── gear_sonic/
```

## Training

### Basic command

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>
```

For example, using the sample data from Hugging Face:

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=16 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
```

Or using the full dataset:

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

### Finetuning from the released checkpoint

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>
```

### Multi-GPU and multi-node training

We recommend training with **64+ GPUs** for reasonable convergence times.
Single-node (8 GPU) training works but is significantly slower.

```bash
# Single node (8 GPUs)
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=4096 headless=True

# Multi-node — use accelerate config for distributed setup
accelerate launch \
    --multi_gpu \
    --num_machines=8 \
    --num_processes=64 \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=4096 headless=True
```

For multi-node setup, see the
[Accelerate distributed training guide](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
and
[multi-node launcher docs](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch).

### W&B logging

Enabled by default. Key overrides:

```bash
WANDB_MODE=offline python gear_sonic/train_agent_trl.py ...   # offline mode
    wandb.wandb_project=my_project wandb.wandb_entity=my_team  # custom project
    use_wandb=false                                             # disable entirely
```

### Local debug run

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=16 headless=False \
    ++algo.config.num_learning_iterations=100
```

## Monitoring

### Key metrics

| Metric | Good range | Description |
|--------|-----------|-------------|
| `rewards/total` | 3.0+ | Total reward |
| `rewards/anchor_pos_err` | < 0.15 | Root position tracking error (m) |
| `rewards/body_pos_err` | < 0.10 | Body position tracking error (m) |
| `throughput/fps` | ~4000+ | Training throughput |

### Checkpoints

Saved every 2000 steps to:

```
logs_rl/TRL_G1_Track/<experiment_name>-<timestamp>/
├── model_step_002000.pt
├── config.yaml
└── ...
```

## Evaluation

### Visualize reference motions

Replay motions to verify data quality before training:

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    ++replay=True num_envs=4 headless=False
```

### Evaluate a checkpoint

Two eval modes: **metrics** (success rate, MPJPE) and **render** (video output).

For the released checkpoint, you must override motion paths since its
`config.yaml` has internal training paths. For your own checkpoints trained
with `sonic_release`, omit the motion overrides.

```bash
# --- Metrics ---
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=128 \
    "+manager_env/terminations=tracking/eval" \
    "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=512"
```

```bash
# --- Render videos ---
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=8 \
    ++manager_env.config.render_results=True \
    "++manager_env.config.save_rendering_dir=/tmp/renders" \
    ++manager_env.config.env_spacing=10.0 \
    "~manager_env/recorders=empty" "+manager_env/recorders=render"
```

For the **released checkpoint only**, append this override to either command
(its embedded config has internal training paths):

```bash
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered"
```

Videos are saved as `000000.mp4`, `000001.mp4`, etc. in `save_rendering_dir`.

### Expected eval metrics

*Training rewards* (W&B `Episode_Reward/`):

| Metric | Converged | Description |
|--------|-----------|-------------|
| `tracking_vr_5point_local` | > 0.80 | 5-point tracking quality |
| `tracking_relative_body_pos` | > 0.44 | Upper-body position tracking |
| `tracking_anchor_pos` | > 0.14 | Root position tracking |
| `time_out` | > 0.90 | Episode completion rate |

*Eval metrics* (from `eval_agent_trl.py`):

| Metric | Converged | Description |
|--------|-----------|-------------|
| `success_rate` | > 0.97 | Motions tracked without early termination |
| `mpjpe_l` | < 30 mm | Local per-joint position error |
| `mpjpe_g` | < 200 mm | Global per-joint position error |

A well-converged policy reaches >0.98 success rate and <29 mm mpjpe_l after
100K iterations.

## ONNX Export

Export a trained checkpoint to ONNX for C++ deployment:

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True ++num_envs=1 \
    +export_onnx_only=true
```

For the released checkpoint, append the motion path overrides shown in the
eval section above.

Output (in `exported/` next to the checkpoint):

| File | Description |
|------|-------------|
| `*_smpl.onnx` | SMPL encoder + decoder (pose estimation input) |
| `*_g1.onnx` | G1 encoder + decoder (robot joint input) |
| `*_teleop.onnx` | Teleop encoder + decoder (VR tracking input) |
| `*_encoder.onnx` | All encoders combined |
| `*_decoder.onnx` | Decoder only |

Use the encoder+decoder pair matching your input modality. See
[deployment code reference](../references/deployment_code.md) for C++ details.

## Training with SOMA encoder

The `sonic_bones_seed` config adds a fourth SOMA encoder for BVH-derived
skeleton joint positions.

### SOMA data preparation

```bash
# Extract SOMA joints from BVH
python gear_sonic/data_process/extract_soma_joints_from_bvh.py \
    --input /path/to/bones_seed/bvh/ \
    --output data/motion_lib_bones_seed/soma \
    --fps 30 --num_workers 16 --skip_existing

# Filter to match robot data
python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source data/motion_lib_bones_seed/soma \
    --dest data/motion_lib_bones_seed/soma_filtered \
    --workers 16
```

### Training

Use multi-node training (64+ GPUs recommended):

```bash
accelerate launch \
    --multi_gpu --num_machines=8 --num_processes=64 \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_bones_seed \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=data/motion_lib_bones_seed/soma_filtered
```

Data layout for 4-encoder training:

```
data/
├── motion_lib_bones_seed/
│   ├── robot_filtered/     # ~130K PKLs (G1 retargeted)
│   └── soma_filtered/      # ~130K PKLs (SOMA skeleton)
└── smpl_filtered/          # ~131K PKLs (SMPL human)
```
