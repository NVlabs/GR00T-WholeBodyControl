# Downloading Model Checkpoints

Pre-trained GEAR-SONIC checkpoints (ONNX format) are hosted on Hugging Face:

**[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)**

## Quick Download

### Install the dependency

```bash
pip install huggingface_hub
```

### Run the download script

From the repo root, run:

```bash
python download_from_hf.py
```

This downloads the **latest** policy encoder + decoder + kinematic planner into
`gear_sonic_deploy/`, preserving the same directory layout the deployment binary expects.

---

## Options

| Flag | Description |
|------|-------------|
| `--no-planner` | Skip the kinematic planner download |
| `--output-dir PATH` | Override the destination directory |
| `--token TOKEN` | HF token (alternative to `huggingface-cli login`) |

### Examples

```bash
# Policy + planner (default)
python download_from_hf.py

# Policy only
python download_from_hf.py --no-planner

# Download into a custom directory
python download_from_hf.py --output-dir /data/gear-sonic
```

---

## Manual download via CLI

If you prefer the Hugging Face CLI:

```bash
pip install huggingface_hub[cli]

# Policy only
huggingface-cli download nvidia/GEAR-SONIC \
    model_releasev1-1_encoder.onnx \
    model_releasev1-1_decoder.onnx \
    --local-dir gear_sonic_deploy

# Everything (policy + planner + media)
huggingface-cli download nvidia/GEAR-SONIC --local-dir gear_sonic_deploy
```

---

## Manual download via Python

```python
from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/GEAR-SONIC"

encoder = hf_hub_download(
    repo_id=REPO_ID,
    filename="policy/release/model_releasev1-1_encoder.onnx",
    local_dir="gear_sonic_deploy",
)
decoder = hf_hub_download(
    repo_id=REPO_ID,
    filename="policy/release/model_releasev1-1_decoder.onnx",
    local_dir="gear_sonic_deploy",
)
config = hf_hub_download(
    repo_id=REPO_ID,
    filename="policy/release/observation_config.yaml",
    local_dir="gear_sonic_deploy",
)

print("Policy encoder :", encoder)
print("Policy decoder :", decoder)
print("Obs config     :", config)
```

---

## Available files

```
nvidia/GEAR-SONIC/
├── model_releasev1-1_encoder.onnx   # Policy encoder
├── model_releasev1-1_decoder.onnx   # Policy decoder
└── planner_sonic.onnx               # Kinematic planner
```

The download script places them into the layout the deployment binary expects:

```
gear_sonic_deploy/
├── policy/release/
│   ├── model_releasev1-1_encoder.onnx
│   └── model_releasev1-1_decoder.onnx
└── planner/target_vel/V2/
    └── planner_sonic.onnx
```

---

## Authentication

The repository is **public** — no token required for downloading.

If you hit rate limits or need to access private forks:

```bash
# Option 1: CLI login (recommended — token is saved once)
huggingface-cli login

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