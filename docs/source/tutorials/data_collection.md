# Data Collection for VLA

Record teleop demonstrations as [LeRobot](https://github.com/huggingface/lerobot) datasets for post-training with [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T). The data exporter runs alongside the SONIC deployment and VR teleop stack, capturing robot state, SMPL teleop poses, and camera images at a configurable frequency.

```{admonition} Deployment model
:class: important
Everything runs **offboard on your workstation** except the **camera server**, which runs **onboard the robot computer** (e.g., Jetson Orin) where the physical cameras are connected. The camera server publishes JPEG frames over ZMQ to the workstation.
```

```{admonition} Supported cameras
:class: note
The tested and supported camera setup uses **Luxonis OAK cameras** (OAK-D, OAK-1, etc.). This includes a head/ego-view OAK camera and optional OAK wrist cameras. Other camera drivers (RealSense, USB webcam) are included in the codebase but have not been tested recently.
```

```{admonition} Prerequisites
:class: note
1. **Completed the [Quick Start](../getting_started/quickstart.md)** — you can run the sim2sim loop (includes [installing the deployment](../getting_started/installation_deploy.md) and [downloading model checkpoints](../getting_started/download_models.md)).
2. **Completed the [VR Teleop Setup](../getting_started/vr_teleop_setup.md)** — PICO hardware is calibrated and `.venv_teleop` is ready.
3. **Camera server running on the robot** — see [Camera Server Setup](#camera-server-setup-on-robot) below. For simulation, the MuJoCo sim loop publishes camera images automatically — no camera server needed.
```

---

## One-Time Setup (Workstation)

On your **workstation** (where you run the C++ deployment, teleop, and data exporter), run the install script from the repo root to create a dedicated virtual environment with all data collection dependencies (LeRobot, PyAV, OpenCV, etc.):

```sh
bash install_scripts/install_data_collection.sh
```

This creates `.venv_data_collection` using Python 3.10 via `uv`. It installs `gear_sonic[data_collection]` which includes `lerobot`, `av`, `opencv-python`, and other required packages. It also installs `espeak` (system package) for voice feedback during recording.

```{tip}
This environment is separate from `.venv_teleop` and `.venv_sim` — the data exporter has heavier ML dependencies that are not needed for teleop or simulation.
```

---

## Camera Server Setup (On-Robot)

The camera server is the **only component that runs on the robot computer** (e.g., Jetson Orin). Everything else — the C++ deployment, PICO teleop streamer, data exporter, and camera viewer — runs on your workstation.

The camera server captures frames from the OAK cameras physically connected to the robot and publishes them over ZMQ to the workstation.

### Step 1: Clone the repo on the robot

SSH into your robot computer and clone this repository:

```sh
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
```

### Step 2: Run the install script

The install script handles everything: creates the virtual environment, installs all
dependencies (including the DepthAI SDK for OAK cameras), detects connected cameras,
and optionally installs a systemd service so the camera server starts automatically
on boot.

```sh
bash install_scripts/install_camera_server.sh
```

The script will:

1. Create `.venv_camera` with `gear_sonic[camera]` (DepthAI, ZMQ, msgpack, OpenCV, tyro).
2. Detect connected OAK cameras and list their MxIDs.
3. Prompt you for each camera position (ego view, and optionally left/right wrist) and its device ID.
4. Ask whether to install the camera server as a **systemd service** (recommended). If you answer **y**, it generates the unit file, installs, enables, and starts the service automatically.

After the script finishes, verify the service is running:

```sh
sudo systemctl status composed_camera_server.service
journalctl -u composed_camera_server.service -f
```

```{note}
Other camera drivers (RealSense, USB webcam) are included in the codebase but have not been tested recently for data collection. If you need RealSense, install `pyrealsense2` into the venv after setup. See the driver files in `gear_sonic/camera/drivers/` for details.
```

### Manual setup (alternative)

If you prefer not to use the install script, or need to reconfigure:

**Finding camera device IDs:**

Each OAK camera has a unique MxID. List all connected OAK devices:

```sh
source .venv_camera/bin/activate
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

Example output:

```text
[XLinkDeviceState.X_LINK_BOOTED, MxId: 18443010E1ABC12300, ...]
```

**Starting the camera server manually:**

```sh
source .venv_camera/bin/activate

# Single camera (ego view only)
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera oak \
    --ego-view-device-id <YOUR_MXID> \
    --port 5555

# Multiple cameras (ego view + wrist cameras)
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera oak --ego-view-device-id <EGO_MXID> \
    --left-wrist-camera oak --left-wrist-device-id <LEFT_WRIST_MXID> \
    --right-wrist-camera oak --right-wrist-device-id <RIGHT_WRIST_MXID> \
    --port 5555
```

Run `python -m gear_sonic.camera.composed_camera --help` for all options including `--fps`, `--use-mjpeg`, and `--mjpeg-quality`.

**Manual systemd setup:**

```sh
# 1. Edit the service file to match your camera setup
nano systemd/composed_camera_server.service

# 2. Copy to systemd, enable, and start
sudo cp systemd/composed_camera_server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable composed_camera_server.service
sudo systemctl start composed_camera_server.service
```

Once the systemd service is running, the camera server starts automatically whenever the robot boots — no manual intervention needed.

### Connecting from the workstation

On your workstation, the data exporter and camera viewer connect to the robot's camera server over the network. Pass the robot's IP address (the G1 robot's default IP is `192.168.123.164`):

```sh
# Data exporter
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the cup" \
    --camera-host 192.168.123.164 --camera-port 5555

# Camera viewer (to verify the feed)
python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host 192.168.123.164 --camera-port 5555
```

The tmux launcher also accepts `--camera-host`:

```sh
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --task-prompt "pick up the cup"
```

### ZMQ message format

The camera server publishes a single msgpack-encoded payload per frame cycle containing all camera images:

```python
{
    "timestamps": {"ego_view": 1712345678.123, "left_wrist": 1712345678.125},
    "images": {"ego_view": "<base64-jpeg>", "left_wrist": "<base64-jpeg>"}
}
```

Images are JPEG-compressed (quality 80) and either base64-encoded strings or raw JPEG bytes (when MJPEG on-device encoding is enabled). The data exporter's `ComposedCameraClientSensor` handles both formats automatically.

---

## Architecture

The data exporter receives data from three ZMQ sources. The C++ deployment, PICO teleop, and data exporter all run **offboard on the workstation**. The camera server runs **onboard the robot** and streams frames to the workstation over the network.

```text
    Workstation (offboard)                           Robot (onboard)
┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────┐
│  C++ deploy          │  │  pico_manager         │  │  Camera       │
│  (zmq_output_handler)│  │  _thread_server.py    │  │  server       │
│                      │  │                       │  │  (OAK cameras)│
│  port 5557           │  │  port 5556            │  │  port 5555    │
│  topics: g1_debug,   │  │  topic: pose          │  │  (JPEG/ZMQ)   │
│          robot_config│  │  (SMPL body params)   │  │               │
└──────────┬───────────┘  └──────────┬────────────┘  └──────┬────────┘
           │                         │                       │
           └────────────┬────────────┘───────────────────────┘
                        │              (network)
               ┌────────▼────────┐
               │  run_data_      │
               │  exporter.py    │
               │  (workstation)  │
               │                 │
               │  LeRobot dataset│
               │  (parquet + mp4)│
               └─────────────────┘
```

| Source | Runs on | ZMQ Topic | Default Port | Provides |
|---|---|---|---|---|
| C++ deployment | Workstation | `g1_debug` | 5557 | Joint positions, velocities, IMU quaternion |
| C++ deployment | Workstation | `robot_config` | 5557 | One-shot robot configuration at startup |
| PICO teleop streamer | Workstation | `pose` | 5556 | SMPL body parameters (teleop target poses) |
| Camera server | Robot | *(raw TCP)* | 5555 | JPEG-compressed camera images (ego view + optional wrist views) |

---

## Running Data Collection

There are two ways to run the data collection stack: an **all-in-one tmux launcher** (recommended) or **manual multi-terminal setup**.

### Option A: All-in-One Tmux Launch (Recommended)

The launcher starts all components in a single tmux session with four panes:

```text
┌───────────────────────┬───────────────────────┐
│ Pane 0: C++ Deploy    │ Pane 2: Data Exporter │
│ (gear_sonic_deploy)   │ (.venv_data_collection)│
├───────────────────────┼───────────────────────┤
│ Pane 1: PICO Teleop   │ Pane 3: Camera Viewer │
│ (.venv_teleop)        │ (.venv_data_collection)│
└───────────────────────┴───────────────────────┘
```

```{note}
Requires `tmux` to be installed (`sudo apt install tmux`).
```

**For simulation** (the launcher starts `run_sim_loop.py` in a separate tmux window automatically):

```bash
python gear_sonic/scripts/launch_data_collection.py --sim
```

**For real robot** (camera server running on robot at `192.168.123.164`):

```bash
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --task-prompt "pick up the cup"
```

**With wrist cameras** (records ego view + left/right wrist camera streams):

```bash
python gear_sonic/scripts/launch_data_collection.py \
    --camera-host 192.168.123.164 \
    --task-prompt "pick up the cup" \
    --record-wrist-cameras
```

```{tip}
No need to activate a virtual environment first — the launcher automatically detects and uses `.venv_data_collection` if the required dependencies are not in the current Python.
```

The launcher auto-attaches to the tmux session.  Use `Ctrl+b` then arrow keys to switch between panes.

Common options:

| Flag | Default | Description |
|---|---|---|
| `--task-prompt` | `"demo"` | Language task description (e.g., `"pick up the cup"`) |
| `--dataset-name` | *(auto: timestamp)* | Dataset name; omit to auto-generate |
| `--sim / --no-sim` | `False` | Run deploy.sh in sim mode (also starts the sim loop) |
| `--camera-host` | `localhost` | Camera server host (e.g., `192.168.123.164` for real robot) |
| `--camera-port` | `5555` | Camera server port |
| `--no-camera-viewer` | *(viewer on)* | Disable the camera viewer pane |
| `--data-exporter-frequency` | `50` | Recording frequency (Hz) |
| `--deploy-checkpoint` | *(default)* | Custom checkpoint path for deploy.sh |
| `--deploy-obs-config` | *(default)* | Custom observation config for deploy.sh |
| `--deploy-planner` | *(default)* | Custom planner model path for deploy.sh |
| `--deploy-motion-data` | *(default)* | Custom motion data path for deploy.sh |
| `--record-wrist-cameras` | `False` | Record left/right wrist camera streams in the dataset |
| `--no-text-to-speech` | *(on)* | Disable voice feedback via espeak |

Run `python gear_sonic/scripts/launch_data_collection.py --help` for all options.

```{tip}
The launcher automatically enables **mouse support** in the tmux session — click to select panes, scroll with the mouse wheel, and drag to resize pane borders.
```

**Session management:**

| Action | Command |
|---|---|
| Switch panes | `Ctrl+b`, then arrow keys |
| Detach (keep running) | `Ctrl+b`, then `d` |
| Reattach | `tmux attach -t sonic_data_collection` |
| Kill session | `Ctrl+\` in any pane, or `tmux kill-session -t sonic_data_collection` |

### Option B: Manual Multi-Terminal Setup

If you prefer individual control over each process, run them in separate terminals:

**Terminal 1 — MuJoCo Simulator** *(skip for real robot)*:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
    --enable-image-publish --enable-offscreen --camera-port 5555
```

The `--enable-image-publish` and `--enable-offscreen` flags are required so the
sim renders camera images and streams them over ZMQ on the specified port.
The data exporter subscribes to this port the same way it subscribes to a
physical camera server.

For real robot deployment, skip this terminal and see [VR Whole-Body Teleop](vr_wholebody_teleop.md) instead.

**Terminal 2 — C++ Deployment** (from `gear_sonic_deploy/`):

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
# Wait until you see "Init done"
```

**Terminal 3 — PICO Teleop Streamer:**

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

**Terminal 4 — Data Exporter:**

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cup"
```

**Terminal 5 (optional) — Camera Viewer:**

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py
```

All options are provided via CLI flags — no interactive prompts.  Key flags:

| Flag | Default | Description |
|---|---|---|
| `--task-prompt` | `"demo"` | Language task description for this session |
| `--dataset-name` | *(auto: timestamp)* | Dataset name.  Omit to create a new one, or pass an existing name to append episodes |
| `--data-collection-frequency` | `50` | Recording frequency (Hz) |
| `--root-output-dir` | `outputs` | Parent directory for saved datasets |

```{tip}
Datasets are saved under `<root-output-dir>/<dataset-name>/`.  If `--dataset-name`
is not specified, a timestamped name is generated automatically.
```

### Recording Controls

There are two ways to control recording: **PICO VR controllers** (recommended during teleop) or **keyboard over ZMQ**.

**PICO VR Controllers (via `manager_state` topic):**

| Input | Action |
|---|---|
| **Left Grip + A** | **Toggle** recording — starts a new episode, or stops and saves the current one |
| **Left Grip + B** | **Discard** the current episode without saving |

These buttons work in any manager mode (POSE, PLANNER, etc.) and are independent of the mode-switching controls.

**Keyboard over ZMQ:**

| Key | Action |
|---|---|
| `c` | **Toggle** recording (same as Left Grip + A) |
| `x` | **Discard** episode (same as Left Grip + B) |

```{note}
Keyboard commands are sent via a separate ZMQ publisher (default port `5580`). The data exporter subscribes to this channel automatically. You can send keys from any ZMQ publisher on that port, or integrate with the C++ deployment's keyboard handler.
```

---

## Camera Viewer

A standalone camera viewer is available for monitoring camera feeds and recording raw video independently of the data exporter.

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
```

The viewer connects to the same ZMQ camera server used by the data exporter and displays all detected camera streams in a tiled OpenCV window.

**Controls** (OpenCV window must be focused):

| Key | Action |
|---|---|
| `R` | Start/stop video recording |
| `Q` | Quit |

Recordings are saved to `camera_recordings/rec_<timestamp>/` with one MP4 per camera stream. This is useful for:
- Verifying camera placement and image quality before starting data collection
- Recording reference videos alongside the LeRobot dataset
- Debugging camera server connectivity

Run `python gear_sonic/scripts/run_camera_viewer.py --help` for all options.

---

## CLI Options

All options can be viewed with `--help`:

```bash
python gear_sonic/scripts/run_data_exporter.py --help
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--task-prompt` | `"demo"` | Language task description for annotation |
| `--dataset-name` | *(auto: timestamp)* | Dataset name; omit to auto-generate, or reuse an existing name to append |
| `--data-collection-frequency` | `50` | Recording frequency in Hz |
| `--camera-host` | `localhost` | Camera server hostname |
| `--camera-port` | `5555` | Camera server port |
| `--sonic-zmq-host` | `localhost` | SMPL pose publisher host |
| `--sonic-zmq-port` | `5556` | SMPL pose publisher port |
| `--state-zmq-host` | `localhost` | Robot state publisher host |
| `--state-zmq-port` | `5557` | Robot state publisher port |
| `--root-output-dir` | `outputs` | Root directory for saved datasets |
| `--text-to-speech / --no-text-to-speech` | `True` | Voice feedback via espeak |

---

## Output Format

Datasets are saved in the [LeRobot v2.1](https://github.com/huggingface/lerobot) format under `<root-output-dir>/<dataset-name>/`:

```text
outputs/2026-04-03-14-30-00-G1-robot01/
├── data/
│   ├── train-00000.parquet      # Tabular data (joint states, actions, annotations)
│   └── ...
├── videos/
│   ├── observation.images.ego_view/
│   │   ├── episode_000000.mp4   # H264-encoded ego camera video
│   │   └── ...
│   ├── observation.images.left_wrist/   # (only with --record-wrist-cameras)
│   └── observation.images.right_wrist/  # (only with --record-wrist-cameras)
└── meta/
    ├── info.json                # Dataset metadata (fps, features, sizes)
    ├── modality.json            # GR00T modality configuration
    ├── episodes.jsonl           # Per-episode metadata
    └── tasks.jsonl              # Task prompt definitions
```

### Recorded Data Channels

Each frame contains:

| Feature | Shape | Description |
|---|---|---|
| `observation.state.joint_position` | `(N,)` | Actuated joint positions (rad) |
| `observation.state.joint_velocity` | `(N,)` | Actuated joint velocities (rad/s) |
| `observation.state.body_rotation_6d` | `(6,)` | Base orientation (6D rotation) |
| `observation.state.projected_gravity` | `(3,)` | Gravity vector in body frame |
| `observation.images.ego_view` | `(480, 640, 3)` | Ego camera image (saved as MP4 video) |
| `observation.images.left_wrist` | `(480, 640, 3)` | Left wrist camera (only with `--record-wrist-cameras`) |
| `observation.images.right_wrist` | `(480, 640, 3)` | Right wrist camera (only with `--record-wrist-cameras`) |
| `action.joint_position` | `(N,)` | Teleop target joint positions |
| `action.body_rotation_6d` | `(6,)` | Teleop target body rotation |
| `annotation.human.action.task_description` | string | Task prompt for this frame |

---

## Post-Processing Datasets

After recording, you can clean and merge datasets using the processing script.
All commands below run in the **data collection virtual environment**:

```bash
source .venv_data_collection/bin/activate
```

### Remove Stale SMPL Frames

Teleop pauses or ZMQ frame drops create frames where `teleop.smpl_pose` is all
zeros.  The processing script detects these and also removes consecutive
frozen (identical) lead-in frames that precede them:

```bash
# Clean a single dataset in-place
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_dataset

# Clean and write to a new directory (non-destructive)
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/my_dataset \
    --output-path outputs/my_dataset_cleaned
```

### Merge Multiple Datasets

Combine several recording sessions into a single dataset.  The script
validates that all sessions share the same `script_config` (robot
configuration) before merging:

```bash
# Merge by listing datasets on the command line
python gear_sonic/scripts/process_dataset.py \
    --dataset-path outputs/session1 outputs/session2 outputs/session3 \
    --output-path outputs/merged_dataset

# Or use a text file (one dataset path per line, # for comments)
python gear_sonic/scripts/process_dataset.py \
    --dataset-list datasets.txt \
    --output-path outputs/merged_dataset
```

SMPL cleaning is applied by default during merging.  When enabled, the script
removes entire frames where the SMPL teleop pose is stuck at zeros — this
happens during operator pauses or ZMQ packet-drop periods where the SMPL
stream stops updating.  Consecutive frozen (identical) frames that lead into
a zero block are also removed, since they represent stale data right before
the dropout.  To skip this cleaning and merge only, add `--no-remove-stale-smpl`.

---

## Next Steps: Fine-tune and Deploy

The output dataset is directly compatible with the [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) post-training pipeline. To fine-tune a VLA model on your collected data and deploy it for autonomous inference, see the [VLA Workflow tutorial](vla_workflow.md).
