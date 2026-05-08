# VLA Inference

This guide covers running a trained Isaac-GR00T VLA policy on the Unitree G1 robot
using the Sonic whole-body control stack.

## Overview

The inference pipeline consists of:

1. **Isaac-GR00T PolicyServer** — loads the VLA model and serves actions over ZMQ
2. **VLA inference client** (`run_vla_inference.py`) — reads camera + robot state,
   queries the PolicyServer, and publishes actions to the C++ control loop
3. **C++ deploy** (`gear_sonic_deploy`) — executes whole-body control on the robot
4. **Camera server** — provides camera images over ZMQ (runs as a systemd service)
5. **Data exporter** (optional) — records episodes during inference

```
┌──────────────────────┐
│  Isaac-GR00T         │
│  PolicyServer        │
│  (GPU machine)       │
└──────┬───────────────┘
       │ ZMQ REQ/REP
       ▼
┌─────────────────────┐    ZMQ TCP    ┌──────────────────────┐
│  VLA Inference      │ ◄─────────── │  Camera Server       │
│  (run_vla_inference)│              │  (on robot)          │
└────┬───────────┬────┘              └──────────────────────┘
     │           │
     │ ZMQ PUB   │ ZMQ SUB
     │ (actions) │ (state)
     ▼           ▼
┌─────────────────────┐
│  C++ Deploy         │
│  (gear_sonic_deploy)│
└─────────────────────┘
```

## Prerequisites

### 1. Isaac-GR00T PolicyServer

The PolicyServer runs on a machine with a GPU. It loads your finetuned VLA model
and serves inference over ZMQ.

Install [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and start the server:

```bash
# On the GPU machine (from the Isaac-GR00T repo)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### 2. Inference Environment

On the inference machine (can be the same as the PolicyServer or a separate PC):

```bash
bash install_scripts/install_inference.sh
```

This creates `.venv_inference` with the Isaac-GR00T PolicyClient and all
inference dependencies.

### 3. Camera Server

The camera server should be running as a systemd service on the robot.
See [Data Collection](data_collection.md) for camera server setup.

### 4. C++ Deploy

The `gear_sonic_deploy` binary must be built. See the main README.

## Action Space

The Sonic embodiment (`unitree_g1_sonic`) uses a 78-dimensional action
space: 64-dim motion token + 7-dim left hand joints + 7-dim right hand joints.

## Quick Start — tmux Launcher

The easiest way to run inference is with the all-in-one tmux launcher:

```bash
# Real robot
python gear_sonic/scripts/launch_inference.py \
    --prompt "pick up the apple" \
    --camera-host 192.168.123.164

# Simulation
python gear_sonic/scripts/launch_inference.py --sim \
    --prompt "pick up the apple"

# Without data recording
python gear_sonic/scripts/launch_inference.py \
    --no-data-exporter \
    --prompt "pick up the apple"
```

The launcher creates a tmux session with four panes:

| Pane | Component | Description |
|------|-----------|-------------|
| 0 (top-left) | C++ Deploy | Whole-body controller |
| 1 (bottom-left) | Keyboard Publisher | Type keyboard commands here |
| 2 (top-right) | VLA Inference | Policy client + action loop |
| 3 (bottom-right) | Data Exporter | Records episodes (optional) |

### Keyboard Controls

Type these keys in the **Keyboard Publisher** pane (pane 1):

| Key | Action |
|-----|--------|
| `k` | Start / stop the C++ control loop |
| `i` | Send initial pose and switch to POSE mode |
| `p` | Pause / resume policy inference |
| `[` | Toggle left hand open/closed (initial pose) |
| `]` | Toggle right hand open/closed (initial pose) |
| `t <text>` | Change the inference prompt (e.g., `t pick up the cup`) |
| `c` | Start recording an episode (data exporter) |
| `s` | Stop recording — success (data exporter) |
| `f` | Stop recording — failure / discard (data exporter) |

### Typical Workflow

1. Wait for all panes to initialize
2. Click on **pane 0** (C++ Deploy) and press Enter to confirm deployment
3. Switch to **pane 1** (Keyboard Publisher)
4. Press `k` to start the C++ control loop (starts in PLANNER mode)
5. Press `i` to send the initial pose (switches to POSE mode)
   > **Note:** The initial motion token in `gear_sonic/utils/inference/initial_poses.py`
   > is specific to the SONIC checkpoint used during training. If you change the
   > SONIC checkpoint, you must update `LATENT_INITIAL_MOTION_TOKEN` to a safe
   > standing pose from the new checkpoint's latent space.
6. Press `p` to unpause the inference loop
7. The robot will begin executing VLA-predicted actions
8. Press `p` to pause, `k` to stop the control loop when done

## Manual Setup (Without tmux)

If you prefer to run each component in separate terminals:

### Terminal 1 — Isaac-GR00T PolicyServer (GPU machine)

```bash
# From the Isaac-GR00T repo
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/your/finetuned_model \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port 5550
```

### Terminal 2 — C++ Deploy

```bash
cd gear_sonic_deploy
./deploy.sh --input-type zmq_manager real
```

### Terminal 3 — VLA Inference

```bash
source .venv_inference/bin/activate
python gear_sonic/scripts/run_vla_inference.py \
    --host <policy_server_ip> \
    --port 5550 \
    --embodiment-tag unitree_g1_sonic \
    --prompt "pick up the apple" \
    --camera-host 192.168.123.164
```

### Terminal 4 — Data Exporter (optional)

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the apple" \
    --camera-host 192.168.123.164
```

## Configuration Reference

### VLA Inference (`run_vla_inference.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | PolicyServer host |
| `--port` | `5550` | PolicyServer port |
| `--embodiment-tag` | `unitree_g1_sonic` | Embodiment tag |
| `--prompt` | `demo` | Language prompt |
| `--action-publish-rate` | `50` | Action publish rate (Hz) |
| `--action-horizon` | `40` | Actions per inference chunk |
| `--rate` | `2.5` | Inference rate (Hz) |
| `--camera-host` | `localhost` | Camera server host |
| `--camera-port` | `5555` | Camera server port |
| `--verbose-timing` | `false` | Always print loop timing |

### tmux Launcher (`launch_inference.py`)

The launcher exposes all the above flags plus deploy and data exporter options.
Run `python gear_sonic/scripts/launch_inference.py --help` for the full list.

## Remote PolicyServer

When running the PolicyServer on a separate GPU machine:

```bash
# On the inference machine, point to the remote server
python gear_sonic/scripts/launch_inference.py \
    --policy-host <gpu_machine_ip> \
    --policy-port 5550 \
    --camera-host 192.168.123.164 \
    --prompt "pick up the apple"
```

Make sure port 5550 (or your chosen port) is accessible between the two machines.

## Latency Compensation

The inference loop automatically compensates for network and compute latency.
When a new action chunk arrives, the system calculates how many actions in the
chunk are already "stale" based on the time elapsed since inference started,
and skips to the appropriate action index. This is controlled by `--action-publish-rate`
and `--action-horizon`.
