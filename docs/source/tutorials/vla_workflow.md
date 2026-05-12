# VLA Workflow: Collect, Fine-tune, Deploy

This tutorial walks through the end-to-end workflow for training and deploying a
VLA policy on the Unitree G1 with SONIC whole-body control:

1. **Collect** teleop demonstrations using the SONIC stack
2. **Fine-tune** the Isaac-GR00T N1.7 model on your collected data
3. **Deploy** the finetuned policy for autonomous inference

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Collect     │     │  2. Fine-tune   │     │  3. Deploy      │
│  (VR Teleop +   │ ──► │  (Isaac-GR00T   │ ──► │  (PolicyServer  │
│   Data Export)  │     │   N1.7)         │     │   + SONIC)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

For examples of whole-body manipulation tasks accomplished with this workflow, see the
[VLA result videos on the GEAR-SONIC project page](https://nvlabs.github.io/GEAR-SONIC/#connection-to-vla-foundation-model).

## How It Works: SONIC Latent Actions

Instead of predicting raw joint angles, the VLA predicts **SONIC latent motion
tokens** — a compact 64-dimensional representation learned by the SONIC whole-body
controller. SONIC then decodes these latents into full-body joint commands at 50 Hz.

```
┌─────────────┐    latent tokens     ┌─────────────┐    joint commands    ┌───────┐
│  VLA Model  │ ──────────────────►  │   SONIC     │ ──────────────────►  │ Robot │
│ (2.5 Hz)    │   (64-dim × 40)      │  Decoder    │    (50 Hz)           │       │
│             │                       │  (C++)      │                      │       │
└─────────────┘                       └─────────────┘                      └───────┘
```

This means the VLA only needs to reason about *what* to do — SONIC handles the
*how*: balance, locomotion, and smooth whole-body coordination
all come for free from the pretrained controller. The result is a system that can
walk, reach, grasp, and manipulate simultaneously.

The full action space per inference step is 78-dimensional: 64-dim motion token +
7-dim left hand joints + 7-dim right hand joints.

## Step 1: Data Collection

Collect teleop demonstrations using VR whole-body teleoperation. The data exporter
records robot state, camera images, and teleop actions as a LeRobot dataset.

See the [Data Collection tutorial](data_collection.md) for full setup instructions
(camera server, VR teleop, recording controls).

**Quick start:**

```bash
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --task-prompt "pick up the soda can and place it in the bin"
```

**Output:** A LeRobot v2.1 dataset directory, e.g.:

```text
outputs/2026-04-03-14-30-00-G1-robot01/
├── data/
│   └── train-00000.parquet
├── videos/
│   └── observation.images.ego_view/
│       └── episode_000000.mp4
└── meta/
    ├── info.json
    ├── modality.json
    ├── episodes.jsonl
    └── tasks.jsonl
```

```{tip}
Collect at least 50–100 demonstrations of the target task for reliable fine-tuning.
Use `--dataset-name` to append multiple sessions into the same dataset, or merge
sessions afterwards with `process_dataset.py`.
```

## Step 2: Fine-tuning with Isaac-GR00T

Fine-tune the GR00T N1.7 base model on your collected dataset using the
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) training pipeline.

### Prerequisites

- Clone and install [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T):
  ```bash
  git clone https://github.com/NVIDIA/Isaac-GR00T.git
  cd Isaac-GR00T
  uv sync --all-extras
  ```
- Multi-GPU machine (4+ GPUs recommended)
- The collected dataset accessible from the training machine

### Launch Fine-tuning

```bash
export NUM_GPUS=4
uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /path/to/your/collected_dataset \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --num-gpus $NUM_GPUS \
    --output-dir /path/to/output \
    --save-total-limit 5 \
    --save-steps 5000 \
    --max-steps 20000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4
```

### Key Parameters

| Flag | Description |
|------|-------------|
| `--base-model-path` | HuggingFace model ID or local path to pretrained weights |
| `--dataset-path` | Path to the LeRobot dataset from Step 1 |
| `--embodiment-tag` | Must match the dataset's embodiment (`UNITREE_G1_SONIC`) |
| `--modality-config-path` | Python file defining the modality configuration |
| `--num-gpus` | Number of GPUs for distributed training |
| `--max-steps` | Total training steps (20k is a good starting point) |
| `--global-batch-size` | Total batch size across all GPUs |
| `--save-steps` | Checkpoint save interval |
| `--use-wandb` | Enable Weights & Biases logging |

### Monitoring

With `--use-wandb`, training metrics (loss, learning rate, etc.) are logged to your
W&B project. Monitor the training loss curve — it should decrease steadily and
plateau before `--max-steps`.

### Output

Checkpoints are saved to `--output-dir`:

```text
/path/to/output/
├── checkpoint-5000/
├── checkpoint-10000/
├── checkpoint-15000/
├── checkpoint-20000/
├── config.json
└── processor_config.json
```

Use the final checkpoint (or the best-performing one based on your evaluation) for
deployment in Step 3.

## Step 3: Deploy for Inference

Deploy the finetuned model using the Isaac-GR00T PolicyServer and the SONIC
inference stack. See the [VLA Inference tutorial](vla_inference.md) for full
details on the inference pipeline, keyboard controls, and configuration.

### Start the PolicyServer

On the GPU machine, from the Isaac-GR00T repository:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/output/checkpoint-20000 \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### Run Inference

On the inference machine (from the GR00T-WholeBodyControl repository):

```bash
python gear_sonic/scripts/launch_inference.py \
    --policy-host <gpu_machine_ip> \
    --policy-port 5550 \
    --camera-host 192.168.123.164 \
    --prompt "pick up the soda can and place it in the bin"
```

## Summary

| Step | Where | Key Command |
|------|-------|-------------|
| Collect | GR00T-WholeBodyControl | `launch_data_collection.py` |
| Fine-tune | Isaac-GR00T | `launch_finetune.py` |
| Deploy | Both repos | `run_gr00t_server.py` + `launch_inference.py` |

For iterating on a task, repeat Steps 1–3: collect more data, fine-tune again
(or continue from an existing checkpoint), and redeploy.
