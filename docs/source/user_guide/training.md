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
| `sonic_v1_1` | G1, teleop, SMPL | SONIC v1.1 with heading-normalized targets and wrist-pose augmentation |
| `sonic_bones_seed` | G1, teleop, SMPL, SOMA | Extended training with SOMA skeleton encoder |

Use `sonic_release` for finetuning and evaluation. The `sonic_bones_seed`
config adds a fourth SOMA encoder (see [Training with SOMA](#training-with-soma-encoder)).
Use `sonic_v1_1` with `sonic_v1_1/last.pt`.

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

### Adaptive sampling attribution

Adaptive sampling is enabled by default. A terminated environment is charged
to the motion ID and frame bin it tracked during physics, before Isaac Lab
resamples the environment. This prevents failures from being assigned to the
newly drawn motion.

After a few hundred iterations, `adp_samp/failure_rate_max` should separate
from `adp_samp/failure_rate_mean`; values that remain nearly identical are a
strong signal that failures are still being attributed after reset.

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

### ExecuTorch export (optional)

Add `+export_pt2=true` to the command above to also save the encoder and the
decoder as `torch.export` programs (`*_encoder.pt2`, `*_decoder.pt2`). This runs
in the training environment and needs no extra packages.

Then turn them into ExecuTorch `.pte` files on the device you deploy to. Each
`.pte` holds a TensorRT engine, so the robot does not build one at startup. A
TensorRT engine only runs on the GPU type and TensorRT version it was built with,
which is why this step runs on the device. Use a separate Python 3.11+
environment, because Torch-TensorRT and ExecuTorch need newer PyTorch and NumPy
versions than training:

```bash
python3 -m venv ~/pte_env && source ~/pte_env/bin/activate
pip install "torch_tensorrt[executorch]==2.15.*"
python gear_sonic/scripts/pt2_to_pte.py \
    <experiment_dir>/exported/model_step_000123_encoder.pt2 \
    <experiment_dir>/exported/model_step_000123_decoder.pt2
```

Use the step number of your own export in the file names. This was tested on
Jetson AGX Thor with `.pt2` files from PyTorch 2.7 (the training environment) and
Torch-TensorRT 2.15, so the command pins that version. The wheels need JetPack 7 or
newer (CUDA 13). On an older JetPack, the pin makes `pip` fail instead of quietly
installing an older Torch-TensorRT that lacks options the script uses. Loading a
`.pt2` from PyTorch 2.7 prints a warning that the file format is deprecated; the
conversion still works.

For the released checkpoint, the exported encoder takes the inputs of
`gear_sonic_deploy/policy/release/observation_config.yaml` without the 2 root height
observations (`motion_root_z_position_10frame_step5` and `motion_root_z_position`),
which its encoders do not read. Set those 2 to `enabled: false` in a copy of that file
and pass the copy to the deployment with `--obs-config`. Do not use
`observation_config_sonic_release.yaml` for it: that file has the same number of
values but in a different order, and the deployment only checks the number. Other
checkpoints need the observation config that matches their own inputs.

The C++ side has a loader for these files: `PolicyEngine` and `EncoderEngine` use
it when the model path ends in `.pte`. No program in the repo can run one yet,
though. `g1_deploy_onnx_ref` always links TensorRT 10 for its ONNX engines and the
planner, while a `.pte` uses the TensorRT 11 that comes with `torch_tensorrt`, and
they cannot share a process. So `g1_deploy_onnx_ref` is always built without
ExecuTorch for now, and it reports an error for a `.pte`. Running `.pte` files needs
the deployment to move to the same TensorRT build that `torch_tensorrt` installs
(11.4 today).

To work on the loader, build `run_tests` with ExecuTorch support. Set up the shell
as for the default build (`source gear_sonic_deploy/scripts/setup_env.sh`), then
activate the environment above and keep it: the tests load the libraries of the
installed packages from there. The ExecuTorch CMake package needs CMake 3.28 or
newer, and the tests need a GPU. The option only changes `run_tests`, and this
`run_tests` stays in its build folder, so the programs in `target/release` stay the
default ones.

```bash
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
cmake -S gear_sonic_deploy -B gear_sonic_deploy/build/executorch -DCMAKE_BUILD_TYPE=Release \
    -DUSE_EXECUTORCH=ON -DCMAKE_PREFIX_PATH="$SITE/executorch;$SITE/torch_tensorrt_executorch_runtime"
cmake --build gear_sonic_deploy/build/executorch --target run_tests -j
gear_sonic_deploy/build/executorch/run_tests --gtest_filter='ExecuTorch*'
```

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
