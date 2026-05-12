# Installation (Training)

This guide walks through setting up the SONIC training environment for whole-body humanoid control.

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA 12.x (L40, A100, or H100 recommended)
- **OS**: Ubuntu 22.04+
- **Python**: 3.11 (required by Isaac Lab; sim/teleop/deploy scripts work on 3.10+)
- **Isaac Lab**: 2.3+ (required for simulation environments)

## Install Isaac Lab

SONIC training uses Isaac Lab for physics simulation. Follow the official
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
to install Isaac Lab.

After installation, verify:

```bash
python -c "import isaaclab; print(isaaclab.__version__)"
```

## Install gear_sonic (Training)

From the repository root:

```bash
pip install -e "gear_sonic/[training]"
```

This installs the training dependencies (Hydra, W&B, HuggingFace TRL, etc.)
on top of the Isaac Lab environment.

## Download Model and Data from Hugging Face

SONIC model checkpoints and SMPL motion data are hosted on
[Hugging Face](https://huggingface.co/nvidia/GEAR-SONIC).

```bash
pip install huggingface_hub
python download_from_hf.py --training
```

This downloads:

- **PyTorch checkpoint** (`sonic_release/last.pt`) for finetuning
- **SMPL motion data** (`data/smpl_filtered/`) for the SMPL encoder

## Prepare Robot Motion Data

SONIC trains on the [Bones-SEED](https://huggingface.co/datasets/bones-studio/seed) motion capture dataset
(142K+ motion sequences retargeted to the Unitree G1).

### Step 1: Download and convert

Download the **G1 retargeted CSVs** (29 DOF, 120 FPS) from
[Bones-SEED on HuggingFace](https://huggingface.co/datasets/bones-studio/seed), then convert:

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input /path/to/bones_seed/g1/csv/ \
    --output data/motion_lib_bones_seed/robot \
    --fps 30 --fps_source 120 --individual --num_workers 16
```

### Step 2: Filter motions

Remove motions the G1 robot cannot perform:

```bash
python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source data/motion_lib_bones_seed/robot \
    --dest data/motion_lib_bones_seed/robot_filtered --workers 16
```

This removes ~8.7% of motions (~130K of 142K remain). See the
[Training Guide](../user_guide/training.md) for details.

Your data directory should look like:

```
<repo_root>/
├── data/
│   ├── motion_lib_bones_seed/
│   │   └── robot_filtered/     # Filtered G1 motions (~130K PKLs)
│   └── smpl_filtered/           # SMPL motion data (from Hugging Face)
└── sonic_release/               # Released checkpoint (from Hugging Face)
```

> **Note**: Data processing scripts (`gear_sonic/data_process/`) do **not** require
> Isaac Lab and can be run on any machine with `pip install -e gear_sonic/`.

## Verify Installation

First, run the pre-flight check to verify all dependencies:

```bash
python check_environment.py --training
```

Then run a quick smoke test with a small number of environments:

```bash
# Interactive (with viewer)
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=16 headless=False \
    ++algo.config.num_learning_iterations=5

# Headless (server / no display)
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=16 headless=True \
    ++algo.config.num_learning_iterations=5
```

After a minute of initialization you should see training metrics (rewards, errors)
printing to the console.

## Full Training

Once installation is verified, see the [Training Guide](../user_guide/training.md)
for full training commands (64+ GPU recommended), evaluation, ONNX export, and
SOMA encoder setup.
