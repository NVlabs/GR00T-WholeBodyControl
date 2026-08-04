# Downloading Model Checkpoints

Pre-trained GEAR-SONIC checkpoints (ONNX format) are hosted on Hugging Face:

**[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)**

## Quick Download

### Install the dependency

```bash
pip install huggingface_hub
```

### Run the download script

From the repo root:

```bash
# Deployment (ONNX models + planner → gear_sonic_deploy/)
python download_from_hf.py

# Low-latency teleoperation checkpoint (ONNX models + planner → gear_sonic_deploy/)
python download_from_hf.py --low-latency

# Training (checkpoint + SMPL data → sonic_release/ + data/smpl_filtered/)
python download_from_hf.py --training

# Low-latency PyTorch checkpoint + config only
python download_from_hf.py --training --low-latency

# Sample data only (1 walking sequence for quick testing)
python download_from_hf.py --sample

# Training checkpoint only (skip 30GB SMPL download)
python download_from_hf.py --training --no-smpl
```

This downloads the **latest** policy encoder + decoder + kinematic planner into
`gear_sonic_deploy/`, preserving the same directory layout the deployment binary expects.

---

## Options

| Flag | Description |
|------|-------------|
| `--training` | Download training checkpoint + SMPL motion data (~30 GB) |
| `--low-latency` | Download the low-latency teleoperation checkpoint. For deployment, ONNX files go to `gear_sonic_deploy/policy/low_latency/`; with `--training`, the PyTorch checkpoint and configs go to `low_latency/`. |
| `--sample` | Download sample motion data only (~4 MB) |
| `--no-planner` | Skip the kinematic planner download |
| `--no-smpl` | With `--training`, skip SMPL data (checkpoint only) |
| `--output-dir PATH` | Override the destination directory |
| `--token TOKEN` | HF token (alternative to `hf auth login`) |

### Examples

```bash
# Policy + planner (default)
python download_from_hf.py

# Policy only
python download_from_hf.py --no-planner

# Low-latency teleoperation policy only
python download_from_hf.py --low-latency --no-planner

# Download into a custom directory
python download_from_hf.py --output-dir /data/gear-sonic
```

---

## Low-Latency Teleoperation Checkpoint

The checkpoint published under `low_latency/` in
[`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC) is configured
for responsive whole-body teleoperation. Its SMPL encoder uses **4 future
reference frames**, compared with **10 frames** in the default release. At
50 Hz (20 ms per frame), this reduces SMPL reference lookahead from
approximately **200 ms to 80 ms**.

This is the controller's reference lookahead, not a measurement of total
end-to-end system latency. The checkpoint does not replace the default
top-level deployment policy.

Download the deployment ONNX files:

```bash
python download_from_hf.py --low-latency
```

This creates:

```
gear_sonic_deploy/
└── policy/low_latency/
    ├── model_encoder.onnx
    ├── model_decoder.onnx
    └── observation_config.yaml
```

### C++ deployment inference

Run the low-latency ONNX controller in simulation:

```bash
cd gear_sonic_deploy
./deploy.sh \
    --cp policy/low_latency/model \
    --obs-config policy/low_latency/observation_config.yaml \
    sim
```

Run it for VLA or teleoperation on the real robot:

```bash
cd gear_sonic_deploy
./deploy.sh \
    --cp policy/low_latency/model \
    --obs-config policy/low_latency/observation_config.yaml \
    --input-type zmq_manager \
    real
```

`deploy.sh` expects `--cp` to be the shared model prefix; it appends
`_encoder.onnx` and `_decoder.onnx` internally. The low-latency PyTorch
checkpoint is available as `low_latency/last.pt`:

```bash
python download_from_hf.py --training --low-latency
```

### Python inference and evaluation

For Python-side checkpoint evaluation in Isaac Lab, download the PyTorch
checkpoint and sample motions:

```bash
python download_from_hf.py --training --low-latency
python download_from_hf.py --sample
```

Then run the low-latency checkpoint with `eval_agent_trl.py`:

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=low_latency/last.pt \
    +headless=False \
    ++num_envs=1 \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
```

For the Python VLA tmux launcher, pass the same low-latency C++ deploy files
through launcher flags:

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/low_latency/model \
    --deploy-obs-config policy/low_latency/observation_config.yaml \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

The launcher still runs the ONNX controller through the C++ deployment pane;
the Python process coordinates the VLA client, camera client, keyboard control,
and optional data exporter.

---

## Manual download via CLI

If you prefer the Hugging Face CLI:

```bash
pip install huggingface_hub[cli]

# Policy only
hf download nvidia/GEAR-SONIC \
    model_encoder.onnx \
    model_decoder.onnx \
    observation_config.yaml \
    --local-dir gear_sonic_deploy

# Everything (policy + planner)
hf download nvidia/GEAR-SONIC --local-dir gear_sonic_deploy
```

---

## Manual download via Python

```python
from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/GEAR-SONIC"

encoder = hf_hub_download(repo_id=REPO_ID, filename="model_encoder.onnx")
decoder = hf_hub_download(repo_id=REPO_ID, filename="model_decoder.onnx")
config  = hf_hub_download(repo_id=REPO_ID, filename="observation_config.yaml")
planner = hf_hub_download(repo_id=REPO_ID, filename="planner_sonic.onnx")

print("Policy encoder :", encoder)
print("Policy decoder :", decoder)
print("Obs config     :", config)
print("Planner        :", planner)
```

---

## SONIC Training Checkpoint

The SONIC release training checkpoint and config are also available on Hugging Face, for evaluation or fine-tuning:

### Download via CLI

```bash
hf download nvidia/GEAR-SONIC \
    sonic_release/last.pt \
    sonic_release/config.yaml \
    --local-dir models
```

### Download via Python

```python
from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/GEAR-SONIC"

checkpoint = hf_hub_download(repo_id=REPO_ID, filename="sonic_release/last.pt")
config = hf_hub_download(repo_id=REPO_ID, filename="sonic_release/config.yaml")

print("Checkpoint :", checkpoint)
print("Config     :", config)
```

### Evaluate the checkpoint

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=models/sonic_release/last.pt \
    +num_envs=1 headless=False
```

---

## Sample Motion Data (Quick Start)

A small sample dataset (1 walking sequence) is included for quick testing without downloading the full Bones-SEED dataset. It contains all three data types needed for training: robot retargeted, SOMA skeleton, and SMPL.

### Download via CLI

```bash
# Sample data only
hf download nvidia/GEAR-SONIC \
    --include "sample_data/*" \
    --local-dir .

# Sample data + training checkpoint
hf download nvidia/GEAR-SONIC \
    --include "sample_data/*" \
    --include "sonic_release/*" \
    --local-dir .
```

This creates:

```
sample_data/
├── robot_filtered/210531/    # G1 retargeted motion (for motion tracking)
│   ├── walk_forward_amateur_001__A001.pkl
│   └── walk_forward_amateur_001__A001_M.pkl
├── soma_filtered/210531/     # SOMA skeleton motion
│   ├── walk_forward_amateur_001__A001.pkl
│   └── walk_forward_amateur_001__A001_M.pkl
└── smpl_filtered/            # SMPL human motion
    ├── walk_forward_amateur_001__A001.pkl
    └── walk_forward_amateur_001__A001_M.pkl
```

### Test training with sample data

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=16 headless=True \
    manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
    manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
```

For full-scale training, download the complete [Bones-SEED](https://huggingface.co/datasets/bones-studio/seed) dataset and follow the [Training Guide](../user_guide/training.md).

---

## SMPL Motion Data (Bones-SEED Filtered)

The SMPL retargeted motion data used for training (131K sequences, filtered from the Bones-SEED dataset) is available as a split tar archive (~30GB total).

### Download and extract

```bash
# Download all parts
hf download nvidia/GEAR-SONIC --include "bones_seed_smpl/*" --local-dir .

# Reassemble and extract
cat bones_seed_smpl/bones_seed_smpl.tar.part_* | tar xf - -C data/
```

This extracts to `data/smpl_filtered/` with 131K `.pkl` files.

Then point training to it:

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

---

## Available files

```
nvidia/GEAR-SONIC/
├── model_encoder.onnx                # Policy encoder (ONNX, for deployment)
├── model_decoder.onnx                # Policy decoder (ONNX, for deployment)
├── observation_config.yaml           # Observation configuration (deployment)
├── planner_sonic.onnx                # Kinematic planner (ONNX)
├── low_latency/
│   ├── model_encoder.onnx            # Low-latency policy encoder (ONNX)
│   ├── model_decoder.onnx            # Low-latency policy decoder (ONNX)
│   ├── observation_config.yaml       # Low-latency observation configuration
│   ├── last.pt                       # Low-latency training checkpoint
│   ├── config.yaml                   # Low-latency training config
│   └── model_config.yaml             # Low-latency model config
├── bones_seed_smpl/                  # SMPL motion data (131K sequences, ~30GB split tar)
│   ├── bones_seed_smpl.tar.part_aa
│   ├── ...
│   └── bones_seed_smpl.tar.part_ag
├── sonic_release/
│   ├── last.pt                       # Training checkpoint (for eval/fine-tuning)
│   └── config.yaml                   # Training config
└── sample_data/                      # Sample motion data (1 walking sequence)
    ├── robot_filtered/               # G1 retargeted motion
    ├── soma_filtered/                # SOMA skeleton motion
    └── smpl_filtered/                # SMPL human motion
```

The download script places deployment files into the layout the deployment binary expects:

```
gear_sonic_deploy/
├── policy/release/
│   ├── model_encoder.onnx
│   ├── model_decoder.onnx
│   └── observation_config.yaml
├── policy/low_latency/
│   ├── model_encoder.onnx
│   ├── model_decoder.onnx
│   └── observation_config.yaml
└── planner/target_vel/V2/
    └── planner_sonic.onnx
```

---

## Authentication

The repository is **public** — no token required for downloading.

If you hit rate limits or need to access private forks:

```bash
# Option 1: CLI login (recommended — token is saved once)
hf login

# Option 2: environment variable
export HF_TOKEN="hf_..."
python download_from_hf.py

# Option 3: pass token directly
python download_from_hf.py --token hf_...
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Next steps

After downloading, follow the [Quick Start](quickstart.md) guide to run the
deployment stack in MuJoCo simulation or on real hardware.
