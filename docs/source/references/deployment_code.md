# C++ Deployment Program Flow

This document describes how the main executables run, their arguments, and the logging/configuration options.

## Program Pipeline

High-level flow (matches current code):
- Input interfaces: `keyboard | gamepad | gamepad_manager | zmq | zmq_manager | ros2 | manager`
- Optional planner (when enabled) generates target animations
- Motion reader provides reference motions for non-planner mode
- Policy inference (TensorRT; optional encoder → decoder)
- Output publishing via `--output-type <zmq|ros2|all>`

## Available Commands

```sh
just build           # Build main project
just clean           # Clean build artifacts
just --list          # Show all available commands
```

## Run

### Frequency Test

Load an ONNX model and print input/output info. This is a sanity check for model loading; the reported frequency is not TensorRT inference speed.

```sh
# Basic usage with default settings (1000 iterations, random data)
just run freq_test policy/example/model_step_000000.onnx

# Custom iterations and data mode
just run freq_test policy/example/model_step_000000.onnx 5000 random
```

**Usage:** `just run freq_test <model_file> [iterations] [data_mode]`
- `model_file`: Path to ONNX model file (required)
- `iterations`: Number of inference iterations (default: 1000)
- `data_mode`: Input data type — `zeros|random|ones` (default: random)

### Policy Deployment

Deploy ONNX policy on G1 robot with motion reference control:

```sh
# Example command (real robot)
just run g1_deploy_onnx_ref enP8p1s0 policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type manager \
  --enable-motion-recording \
  --enable-csv-logs

# MuJoCo simulation (disables CRC validation)
python ../gear_sonic/scripts/run_sim_loop.py
just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type manager \
  --enable-motion-recording \
  --enable-csv-logs \
  --disable-crc-check
```

**Usage:** `just run g1_deploy_onnx_ref <network_interface> <model_file> <motion_data_path> [options...]`

**Required Arguments:**
- `network_interface`: Network interface for DDS communication (e.g., `eth0`, `enp5s0`, `enP8p1s0`, `lo`)
- `model_file`: Path to ONNX policy model file
- `motion_data_path`: Path to motion data directory containing reference motions

**Optional Arguments:**

**Model Configuration:**
- `--obs-config <path>`: Path to observation configuration YAML file
- `--encoder-file <path>`: Path to ONNX encoder model file (optional, for token-based policies)
- `--planner-file <path>`: Path to ONNX planner model file (required for ROS2, `gamepad_manager`, and `zmq_manager` planner mode)
- `--planner-precision <16|32>`: Floating point precision for planner (default: 32)
- `--policy-precision <16|32>`: Floating point precision for policy (default: 32)

**Output Mode:**
- `--output-type <type>`: Output interface for publishing control results
  - `zmq` — Publish via ZMQ (default)
  - `ros2` — Publish via ROS2 (only if built with ROS2 support)
  - `all` — Create all available output interfaces simultaneously

**Input Mode:**
- `--input-type <type>`: Input interface type (default: `keyboard`)
  - `keyboard` — Direct keyboard input
  - `gamepad` — Wireless controller
  - `gamepad_manager` — Gamepad + quick switching to ZMQ/ROS2
  - `zmq` — Network motion streaming
  - `zmq_manager` — Dynamic switching between planner and network motion streaming
  - `manager` — Dynamic switching between keyboard, gamepad, ZMQ, and ROS2
  - `ros2` — ROS2 topic control (requires planner, only if built with ROS2 support)

**ZMQ Configuration (when using `--input-type zmq`, `zmq_manager`, `demo_gamepad_manager`, or `manager`):**
- `--zmq-host <host>`: ZMQ server host (default: `localhost`)
- `--zmq-port <port>`: ZMQ server port (default: `5556`)
- `--zmq-topic <topic>`: ZMQ topic/prefix (default: `pose`)
- `--zmq-conflate`: Enable ZMQ CONFLATE mode
- `--zmq-verbose`: Enable verbose ZMQ subscriber logging
- `--zmq-out-port`: Port to which control results will be published when using `--output-type zmq` (default: `5557`)
- `--zmq-out-topic`: Topic to which control results will be published when using `--output-type zmq` (default: `g1_debug`)

**Simulation:**
- `--disable-crc-check`: Disable CRC validation (required for MuJoCo simulation)

**Hand & Compliance Control:**
- `--set-compliance <value>`: Set initial VR 3-point compliance (0.01 = rigid, 0.5 = compliant; default: `0.5,0.5,0.0`). Can specify 1 value (applied to both hands) or 3 comma-separated values (`left_wrist,right_wrist,head`). Runtime keyboard controls: `g/h` = left hand ±0.1, `b/v` = right hand ±0.1.
- `--max-close-ratio <value>`: Set initial hand max close ratio (0.2–1.0; default: 1.0 = full closure allowed). Runtime keyboard controls: `x/c` = ±0.1.

**Logging (CLI flags):**
- **Debug / analysis logs (write a single CSV file)**:
  - `--target-motion-logfile <path>`: Log the target motion tracked by the controller (visualize with `visualize_motion.py`)
  - `--planner-motion-logfile <path>`: Log planner-generated animation sequences
  - `--policy-input-logfile <path>`: Log policy input (observation) tensors
  - `--record-input-file <path>`: Record operator control inputs to CSV for later playback
  - `--playback-input-file <path>`: Play back previously recorded control inputs from CSV
- **State CSV logs (write a timestamped directory)**:
  - `--logs-dir <path>`: Base directory for state CSV logs (default: `logs/dd-mm-yy/hh-mm-ss`)
  - `--enable-csv-logs`: Enable robot state CSV logging (default: OFF)
  - `--enable-motion-recording`: Record the active motion stream(s) to `reference/recorded_motion/...` (default: OFF)

## Logging (Details)

The system provides multiple logging capabilities for debugging, analysis, and replay.

### Motion Logging

**Target Motion (`--target-motion-logfile <path>`):**
- Logs the motion the controller is tracking each control frame (~50 Hz)
- CSV columns: `pos_x, pos_y, pos_z, rot_qw, rot_qx, rot_qy, rot_qz, dof_0, dof_1, ... dof_28`
  - Global position (xyz)
  - Global rotation quaternion (w, x, y, z)
  - 29 joint angles (DoF)

**Planner Motion (`--planner-motion-logfile <path>`):**
- Logs animation sequences generated by the planner (~10 Hz planning updates)
- Each planner update produces a short sequence (e.g., ~100 frames) that is appended to the CSV
- Same CSV format as target motion
- Contains motion blending and replanning results

**Motion Recording (`--enable-motion-recording`):**
- Automatically records the currently active motion stream(s) into timestamped folders under `reference/recorded_motion/YYYYMMDD/`
  - **Streamed motion** (ZMQ pose topic): saved as `streamed_HHMMSS/`
  - **Planner motion** (planner-generated sequence): saved as `planner_motion_HHMMSS/`
- Each recording folder contains `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, etc.
- Useful for offline inspection / regression comparisons of closed-loop behavior

### Visualization

All motion CSV files (logged data and reference motions) can be visualized using the `visualize_motion.py` script:

```sh
# Visualize logged motion data (single CSV file)
python visualize_motion.py --csv_path target_motion.csv

# Visualize reference motion from motion data directory
python visualize_motion.py --motion_dir reference/example/high_jump_full_turn/
```

The visualizer script can connect to a running `g1_deploy` executable to visualize target/measured robot motions in real time:

```sh
python visualize_motion.py --realtime_debug_url tcp://localhost:5557
```

This displays three G1 robots: target animation (colored), target with zero translation (green), and measured sensor data (red).

**Configuration:**
- Default port: 5557 (change with `--zmq-out-port <port>`)
- Default topic: `g1_debug` (change with `--zmq-out-topic <topic>` on executable, `--realtime_debug_topic <topic>` on visualizer)
- For physical robots, replace `localhost` with the robot's IP address

**Playback Controls:**
- **Space**: Pause/resume playback
- **`.`** (period): Step forward one frame
- **`,`** (comma): Step backward one frame
- **`r`**: Reset to frame 0

### Policy Input Logging

**Policy Input (`--policy-input-logfile <path>`):**
- Logs the raw observation tensor fed to the neural network policy
- Output: a single CSV file (one row per control step, all observation values)
- Useful for debugging observation configuration and input drift

### Control Input Recording/Playback

**Recording (`--record-input-file <path>`):**
- Records control inputs (motion index, frame, operator state, planner state, movement commands)
- Logging starts when the control system is activated
- Tip: Wait a few seconds after lowering from gantry before starting control to give yourself setup time during playback

**Playback (`--playback-input-file <path>`):**
- Replays recorded control inputs for reproducible experiments
- Playback starts when the control system is activated
- Useful for testing policy changes with identical inputs

### Robot State CSV Logger

When enabled with `--enable-csv-logs`, the system logs detailed robot state at each control step (50 Hz).

**Output Directory:**
- Default: `logs/dd-mm-yy/hh-mm-ss` (auto-generated timestamp)
- Custom: Use `--logs-dir <path>` to specify directory

**Files Generated (split by signal type):**
- `base_quat.csv` — Base IMU quaternion (4 values: w, x, y, z)
- `base_ang_vel.csv` — Base angular velocity (3 values: x, y, z)
- `torso_quat.csv` — Torso IMU quaternion (4 values)
- `torso_ang_vel.csv` — Torso angular velocity (3 values)
- `q.csv` — Joint positions (29 joints)
- `dq.csv` — Joint velocities (29 joints)
- `action.csv` — Policy actions (29 joints)

**CSV Format:**
- Columns: `index,time_ms,...`
- `time_ms`: Milliseconds since first log (0.0 at start, fractional allowed)
- Synchronized across all files using the same index/timestamp

**Example:**

```sh
just run g1_deploy_onnx_ref enp5s0 policy/model.onnx reference/motions/ \
  --obs-config policy/obs_config.yaml \
  --enable-csv-logs \
  --logs-dir logs/my_experiment
```

## Observation Configuration

The system uses YAML configuration files to define which observations are fed to the policy. This allows flexible policy designs without code changes.

**Basic Structure (`--obs-config <path>`):**

```yaml
observations:
  - name: "body_joint_positions"
    enabled: true
  - name: "base_angular_velocity"
    enabled: true
  # ... other observations
```

**With Encoder (Token-Based Policies):**

For policies that use encoded tokens, add an `encoder:` section:

```yaml
observations:
  - name: "token_state"           # Encoder outputs (64-dim tokens)
    enabled: true
  - name: "base_angular_velocity" # Direct observations
    enabled: true

encoder:
  dimension: 64       # Token output dimension
  use_fp16: false     # TensorRT precision (optional)
  encoder_observations:
    - name: "motion_joint_positions_10frame_step5"
      enabled: true
    # ... observations fed to encoder
```

Then run with `--encoder-file <path>` to load the encoder model. If omitted, tokens can be set externally via ROS2/ZMQ.

**Complete Observation Reference:**

For the full list of all available observation names, dimensions, and example configurations, see [Observation Configuration](observation_config.md).

**Examples:**
- See `policy/observation_config_example.yaml`
