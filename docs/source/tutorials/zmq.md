# Streaming Motion Tracking

Stream motion data to the robot over ZMQ for reference motion tracking. This interface supports streaming either **SMPL-based poses** (e.g., from PICO) or **G1 whole-body joint positions** (qpos) from any external source (`--input-type zmq`).

```{admonition} Prerequisites
:class: note
Complete the [Quick Start](../getting_started/quickstart.md) to have the sim2sim loop running.
```

```{admonition} Emergency Stop
:class: danger
Press **`O`** at any time to immediately stop control and exit. Always keep a hand near the keyboard ready to press **`O`**.
```

## Launch

**Sim2Sim (MuJoCo):**

```bash
# Terminal 1 — MuJoCo simulator (from repo root)
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 — C++ deployment (from gear_sonic_deploy/)
bash deploy.sh sim --input-type zmq \
  --zmq-host <publisher-ip> \
  --zmq-port 5556 \
  --zmq-topic pose
```

**Real Robot:**

```bash
# From gear_sonic_deploy/
bash deploy.sh real --input-type zmq \
  --zmq-host <publisher-ip> \
  --zmq-port 5556 \
  --zmq-topic pose
```

## Step-by-Step

1. Press **`]`** to start the control system.
2. By default you are in **reference motion mode** — use **`T`** to play motions, **`N`** / **`P`** to switch, **`R`** to restart (same as the [keyboard interface](keyboard.md)).
3. Press **`ENTER`** to toggle into **ZMQ streaming mode**. The terminal will print `ZMQ STREAMING MODE: ENABLED`.
4. The policy now tracks motion frames arriving from the ZMQ publisher in real time. Playback starts automatically.
5. Press **`ENTER`** again to switch back to reference motions. The terminal will print `ZMQ STREAMING MODE: DISABLED`, and the encode mode resets to `0` (joint-based).
6. Use **`Q`** / **`E`** to adjust the heading (±0.1 rad per press) in either mode.
7. Press **`I`** to reinitialize the base quaternion and reset the heading to zero.
8. When done, press **`O`** to stop control and exit.

```{note}
**No planner support** — this interface uses pre-loaded and ZMQ-streamed reference motions only. For planner + ZMQ control (e.g., PICO VR teleoperation), use `--input-type zmq_manager` instead. See the [VR Whole-Body Teleop tutorial](vr_wholebody_teleop.md).
```

```{tip}
**Build your own streaming source.** The ZMQ stream protocol documented below is self-contained — any publisher that sends messages in this format can drive the robot. You can write your own motion capture retargeting pipeline, simulator bridge, or any other source that produces the required fields. No PICO hardware is needed.
```

## Using with PICO VR Teleop

You can use `--input-type zmq` with the PICO teleop streamer for a simple, streaming-only whole-body teleoperation setup. In this mode, the PICO streams full-body SMPL poses over ZMQ and the deployment side tracks them directly — no locomotion planner, no PICO-button mode switching. All control is done from the keyboard.

### Prerequisites

1. **Completed the [Quick Start](../getting_started/quickstart.md)** — you can run the sim2sim loop.
2. **PICO VR hardware is set up** — headset and controllers are connected, body tracking is working, and `.venv_teleop` is installed. See the [VR Teleop Setup](../getting_started/vr_teleop_setup.md) for installation and calibration.

### Launch (Sim2Sim)

Run three terminals:

**Terminal 1 — MuJoCo simulator** (from repo root):

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**Terminal 2 — C++ deployment** (from `gear_sonic_deploy/`):

```bash
bash deploy.sh sim --input-type zmq \
  --zmq-host localhost \
  --zmq-port 5556 \
  --zmq-topic pose
```

**Terminal 3 — PICO teleop streamer** (from repo root):

```bash
source .venv_teleop/bin/activate

# With visualization (recommended for first run):
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager --vis_smpl --vis_vr3pt

# Without visualization (headless):
# python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

### Launch (Real Robot)

Run two terminals (no MuJoCo):

**Terminal 1 — C++ deployment** (from `gear_sonic_deploy/`):

```bash
bash deploy.sh real --input-type zmq \
  --zmq-host <teleop-machine-ip> \
  --zmq-port 5556 \
  --zmq-topic pose
```

Replace `<teleop-machine-ip>` with `localhost` if the PICO streamer runs on the same machine, or the IP of the machine running Terminal 2.

**Terminal 2 — PICO teleop streamer** (from repo root):

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

### Step-by-Step

1. **Calibration pose**: Stand upright, feet together, upper arms at your sides, forearms bent 90° forward (L-shape at each elbow), palms facing inward.
2. On the PICO controllers, press **A + B + X + Y** simultaneously to initialize and calibrate the body tracking.
3. Press **A + X** on the PICO controllers to start streaming poses.
4. In Terminal 2 (C++ deployment), press **`]`** to start the control system.
5. In the MuJoCo window (sim only), press **`9`** to drop the robot to the ground.
6. Back in Terminal 2, press **`ENTER`** to enable ZMQ streaming. The terminal prints `ZMQ STREAMING MODE: ENABLED`. The robot begins tracking your PICO poses in real time.
7. Move your body — the robot mirrors your motions. Use the **Trigger** button on each PICO controller to close the corresponding robot hand.
8. To **pause** streaming (e.g., to reposition yourself), press **`ENTER`** again. The terminal prints `ZMQ STREAMING MODE: DISABLED`. The robot holds its last pose and stops tracking. You can move freely without affecting the robot.
9. To **resume**, press **`ENTER`** once more. The robot will snap to your current pose — **move back close to the robot's current pose before resuming** to avoid sudden jumps.
10. When done, press **`O`** to stop control and exit.

```{admonition} DANGER — Resuming from Pause
:class: danger
When you press **`ENTER`** to resume streaming after a pause, the robot will immediately try to reach your current physical pose. If your body is in a very different position from the robot, the robot may perform sudden, aggressive motions. **Always move back close to the robot's current pose before pressing `ENTER` to resume.**
```

### PICO Buttons in ZMQ Mode

In `--input-type zmq` mode, the C++ deployment side does **not** process PICO controller button combos directly. However, the buttons still affect the **Python streamer**, which controls what data gets published on the `pose` ZMQ topic. Since the deployment side tracks whatever arrives (or stops arriving) on that topic, several buttons still have an indirect effect on the robot.

| PICO Button | Effect |
|-------------|--------|
| **A + B + X + Y** | Calibrate body tracking in the streamer. Press once to initialize; press again to stop streaming (emergency stop on the streamer side). |
| **A + X** | Toggle Pose mode in the streamer — starts or stops publishing pose data. When stopped, the robot holds its last pose. **Works as pause/resume.** |
| **Menu (hold)** | Pauses pose streaming in the streamer while held. The robot holds its last pose until you release. **Works as pause.** Move back close to the robot's current pose before releasing. |
| **Trigger** | Hand grasp — processed by the streamer and sent as `left_hand_joints` / `right_hand_joints` in the stream. |
| **B + Y** | Toggle Pose mode in the streamer (same effect as A+X) — starts or stops publishing pose data. **Works as pause/resume.** |

All mode control on the deployment side is done from the keyboard:

| Key | Action |
|-----|--------|
| **`]`** | Start control system |
| **`ENTER`** | Toggle streaming on/off (pause/resume) |
| **`O`** | Emergency stop — stop control and exit |
| **`I`** | Reinitialize base quaternion and reset heading |
| **`Q`** / **`E`** | Adjust heading (±0.1 rad) |

```{note}
For the full PICO VR experience with planner support, locomotion modes, and PICO-controller-based mode switching, use `--input-type zmq_manager` instead. See the [VR Whole-Body Teleop tutorial](vr_wholebody_teleop.md).
```

## Controls

| Key | Action |
|-----|--------|
| **]** | Start control system |
| **O** | Stop control and exit (emergency stop) |
| **ENTER** | Toggle between reference motions and ZMQ streaming |
| **I** | Reinitialize base quaternion and reset heading |
| **Q** / **E** | Adjust delta heading left / right (±0.1 rad) |

*Reference motion mode only (streaming off):*

| Key | Action |
|-----|--------|
| **T** | Play current motion to completion |
| **R** | Restart current motion from beginning (pause at frame 0) |
| **P** / **N** | Previous / Next motion sequence |

## Stream Protocol Versions

The encode mode is determined automatically by the ZMQ stream protocol version. **SONIC uses Protocol v1 and v3.** Protocol v2 is available for custom applications.

### Encode Mode Logic

The encode mode only takes effect when the policy model has an **encoder** configured and loaded. At startup, each motion's encode mode is initialized based on encoder availability:

| `encode_mode` | Meaning |
|----------------|---------|
| `-2` | No encoder / token state configured in the model — encode mode has no effect |
| `-1` | Encoder config exists (token state dimension > 0) but no encoder model file provided |
| `0` | Encoder loaded, joint-based mode (default) |
| `1` | Encoder loaded, teleop / 3 points upper-body mode |
| `2` | Encoder loaded, SMPL-based mode |

When ZMQ streaming is active, the protocol version sets the encode mode on the streamed motion: v1 → `0`, v2/v3 → `2`. This only affects inference if the model actually has an encoder (`encode_mode >= 0`). If no encoder is configured (`-2`), the value is set but has no effect on the inference pipeline.

When switching back to reference motions (pressing **ENTER** to disable streaming), the encode mode resets to `0` (if the motion has an encoder, i.e. `encode_mode >= 0`).

### Common Fields (All Versions)

All versions require two common fields:

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `body_quat` | `[N, 4]` or `[N, num_bodies, 4]` | `f32` / `f64` | Body quaternion(s) per frame (w, x, y, z) |
| `frame_index` | `[N]` | `i32` / `i64` | Monotonically increasing frame indices for alignment |

```{warning}
Changing the protocol version mid-session is not allowed. If the publisher switches protocol versions while streaming, the interface will automatically disable ZMQ mode and return to reference motions for safety.

Error message: `Protocol version changed from X to Y during active ZMQ session!`
```

### Protocol v1 — Joint-Based (Encode Mode 0)

Streams raw G1 joint positions and velocities. Use this when your source provides direct qpos/qvel data (e.g., from another simulator or motion capture retargeting pipeline).

**Required fields:**

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `joint_pos` | `[N, 29]` | `f32` / `f64` | Joint positions in IsaacLab order (all 29 joints) |
| `joint_vel` | `[N, 29]` | `f32` / `f64` | Joint velocities in IsaacLab order (all 29 joints) |

- `N` = number of frames per message (batch size).
- All 29 joint values must be provided and meaningful.
- Frame counts of `joint_pos` and `joint_vel` must match.

**Common errors:**
- `Version 1 missing required fields (joint_pos, joint_vel)` — one or both fields are absent.
- `Frame count mismatch between joint_pos and joint_vel` — the `N` dimension differs.

### Protocol v2 — SMPL-Based (Encode Mode 2)

Streams SMPL body model data. This protocol is **not used by SONIC's built-in pipelines** — it is available for your own custom applications that produce SMPL representations, for example a plicy only observe the SMPL.

**Required fields:**

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `smpl_joints` | `[N, 24, 3]` | `f32` / `f64` | SMPL joint positions (24 joints × xyz) |
| `smpl_pose` | `[N, 21, 3]` | `f32` / `f64` | SMPL joint rotations in axis-angle (21 body poses × xyz) |

- `joint_pos` and `joint_vel` are **optional** in v2.

**Common errors:**
- `Version 2 missing required field 'smpl_joints'` or `'smpl_pose'` — required SMPL fields are absent.

### Protocol v3 — Joint + SMPL Combined (Encode Mode 2)

Combines both joint-level and SMPL data. This is what SONIC uses for whole-body teleoperation (e.g., PICO VR).

**Required fields:**

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `joint_pos` | `[N, 29]` | `f32` / `f64` | Joint positions in IsaacLab order |
| `joint_vel` | `[N, 29]` | `f32` / `f64` | Joint velocities in IsaacLab order |
| `smpl_joints` | `[N, 24, 3]` | `f32` / `f64` | SMPL joint positions (24 joints × xyz) |
| `smpl_pose` | `[N, 21, 3]` | `f32` / `f64` | SMPL joint rotations in axis-angle (21 body poses × xyz) |

```{important}
In Protocol v3, **only the 6 wrist joints need meaningful values** in `joint_pos` — the remaining 23 joints can be zero. The wrist joint indices (in IsaacLab order) are: **[23, 24, 25, 26, 27, 28]** (3 joints per wrist × 2 wrists). The `joint_vel` values for non-wrist joints can also be zero.

The SMPL fields (`smpl_joints`, `smpl_pose`) carry the primary motion data in v3; the wrist joints in `joint_pos` provide fine-grained wrist control that SMPL alone cannot capture.
```

- Frame counts across all four fields must be consistent.

**Common errors:**
- `Version 3 missing required field 'joint_pos'` or `'joint_vel'` — joint fields are absent (unlike v2, they are required in v3).
- `Version 3 frame count mismatch between smpl_joints (X) and joint_pos (Y)` — the `N` dimension differs across fields.

### Protocol Summary

| Protocol | Encode Mode | Used by SONIC | Required Fields |
|----------|-------------|---------------|-----------------|
| v1 | `0` (joint-based) | ✅ Yes | `joint_pos`, `joint_vel` |
| v2 | `2` (SMPL-based) | ❌ Custom only | `smpl_joints`, `smpl_pose` |
| v3 | `2` (SMPL-based) | ✅ Yes | `joint_pos`, `joint_vel`, `smpl_joints`, `smpl_pose` |

## Optional Stream Fields

The following optional fields can be included in any protocol version:

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `left_hand_joints` | `[7]` or `[1, 7]` | `f32` / `f64` | Left hand 7-DOF Dex3 joint positions |
| `right_hand_joints` | `[7]` or `[1, 7]` | `f32` / `f64` | Right hand 7-DOF Dex3 joint positions |
| `vr_position` | `[9]` or `[3, 3]` | `f32` / `f64` | VR 3-point tracking positions: left wrist, right wrist, head (xyz × 3) |
| `vr_orientation` | `[12]` or `[3, 4]` | `f32` / `f64` | VR 3-point orientations: left, right, head quaternions (wxyz × 3) |
| `catch_up` | scalar | `bool` / `u8` / `i32` | If `true` (default), resets playback when a large frame gap is detected |
| `heading_increment` | scalar | `f32` / `f64` | Incremental heading adjustment applied per message |

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--zmq-host` | `localhost` | ZMQ publisher host |
| `--zmq-port` | `5556` | ZMQ publisher port |
| `--zmq-topic` | `pose` | ZMQ topic prefix |
| `--zmq-conflate` | off | Keep only the latest message (drop stale frames) |
