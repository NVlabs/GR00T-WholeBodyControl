# Kinematic Planner ONNX Model Reference

This page provides a detailed specification of the **Kinematic Planner** ONNX model inputs and outputs. The kinematic planner is the core motion generation component of the GEAR-SONIC system: given the robot's current state and high-level navigation commands, it produces a sequence of future whole-body poses (MuJoCo `qpos` frames) that the low-level whole-body controller then tracks.

The ONNX model is part of the **C++ inference stack** and is called by the deployment runtime during operation. The C++ stack manages input construction, timing, and state management — certain combinations of inputs are invalid and are handled by the C++ layer to ensure safe operation. This page is intended for developers who want to understand the model interface at a deeper level or build custom integrations beyond the standard deployment pipeline.

```{admonition} Training Code & Technical Report
:class: note
The kinematic planner training code and technical report will be released soon. This page documents the ONNX model interface for deployment integration.
```

---

## Overview

The planner takes **11 input tensors** and produces **2 output tensors**. The 6 primary inputs are listed below; the remaining 5 are <a href="#advanced-inputs">advanced inputs</a> managed by the C++ stack and should not need to be modified in most cases.

**Primary inputs:**

| Tensor Name | Shape | Dtype | Default |
|-------------|-------|-------|---------|
| `context_mujoco_qpos` | `[1, 4, 36]` | `float32` | Required |
| `target_vel` | `[1]` | `float32` | `-1.0` (use mode default velocity) |
| `mode` | `[1]` | `int64` | Required |
| `movement_direction` | `[1, 3]` | `float32` | Required |
| `facing_direction` | `[1, 3]` | `float32` | Required |
| `height` | `[1]` | `float32` | `-1.0` (disable height control) |

**Outputs:**

| Tensor Name | Shape | Dtype |
|-------------|-------|-------|
| `mujoco_qpos` | `[1, N, 36]` | `float32` |
| `num_pred_frames` | scalar | `int64` |

Where:
- **K** = `max_tokens - min_tokens + 1` (model-dependent; the range of allowed prediction horizons)
- **N** = maximum number of output frames (padded); only the first `num_pred_frames` frames are valid

---

## Coordinate System

The model operates in **MuJoCo's Z-up coordinate convention**:

- **X** — forward
- **Y** — left
- **Z** — up

All position and direction vectors in the inputs and outputs follow this convention.

---

## Input Tensors

(context_mujoco_qpos)=
### `context_mujoco_qpos`

| Property | Value |
|----------|-------|
| **Shape** | `[1, 4, 36]` |
| **Dtype** | `float32` |
| **Description** | The planner's context input consisting of 4 consecutive MuJoCo `qpos` frames representing the recent states of the robot |

This is the primary context input. It provides 4 frames of the robot's recent joint configuration at the simulation framerate.
The 36 dimensions of each frame are the standard MuJoCo `qpos` vector for the Unitree G1 (29-DOF) model:

| Index | Field | Description |
|-------|-------|-------------|
| 0–2 | Root position | `(x, y, z)` in meters, Z-up world frame |
| 3–6 | Root quaternion | `(w, x, y, z)` orientation — MuJoCo convention |
| 7–35 | DOF positions | 29 joint angles in radians, following MuJoCo body tree order |

```{admonition} Coordinate Frame
:class: note
All inputs — including `context_mujoco_qpos`, `movement_direction`, `facing_direction`, `specific_target_positions`, and `specific_target_headings` — should be provided in the **world coordinate frame**. The root quaternion uses MuJoCo's `(w, x, y, z)` ordering at indices 3 to 6. The model handles canonicalization internally.
```

### `target_vel`

| Property | Value |
|----------|-------|
| **Shape** | `[1]` |
| **Dtype** | `float32` |
| **Description** | Desired locomotion speed override |

Controls the target movement speed. When set to **zero or below** (e.g., `-1.0`), the model uses the default velocity for the selected mode. When set to a **positive value**, it overrides the mode's default speed (in meters per second). Note that the actual achieved speed may differ from the target due to the critically damped spring model and motion dynamics.

| Value | Behavior |
|-------|----------|
| `<= 0.0` | Use the default velocity for the selected `mode` |
| `> 0.0` | Override with this target velocity (m/s) |


### `mode`

| Property | Value |
|----------|-------|
| **Shape** | `[1]` |
| **Dtype** | `int64` |
| **Description** | Index selecting the motion style/behavior |

Selects the motion style from the pre-loaded clip library. The mode index is clamped to the number of available clips at runtime. The default planner ships with the following modes:

**Locomotion set:**

| Index | Mode | Description |
|-------|------|-------------|
| 0 | `idle` | Standing still |
| 1 | `slowWalk` | Slow forward locomotion |
| 2 | `walk` | Normal walking speed |
| 3 | `run` | Running |

**Squat / ground set:**

| Index | Mode | Description |
|-------|------|-------------|
| 4 | `squat` | Squatting — requires `height` input (range ~0.4–0.8m) |
| 5 | `kneelTwoLeg` | Kneeling on both knees — requires `height` input (0.2m-0.4m) |
| 6 | `kneelOneLeg` | Kneeling on one knee — requires `height` input (0.2m-0.4m) |
| 7 | `lyingFacedown` | Lying face down — requires `height` input |
| 8 | `handCrawling` | Crawling on hands and knees |
| 14 | `elbowCrawling` | Crawling on elbows (more likely to overheat) |

**Boxing set:**

| Index | Mode | Description |
|-------|------|-------------|
| 9 | `idleBoxing` | Boxing stance (idle) |
| 10 | `walkBoxing` | Walking with boxing guard |
| 11 | `leftJab` | Left jab |
| 12 | `rightJab` | Right jab |
| 13 | `randomPunches` | Random punch sequence |
| 15 | `leftHook` | Left hook |
| 16 | `rightHook` | Right hook |

**Style walks:**

| Index | Mode | Description |
|-------|------|-------------|
| 17 | `happy` | Happy walking |
| 18 | `stealth` | Stealthy walking |
| 19 | `injured` | Limping walk |
| 20 | `careful` | Cautious walking |
| 21 | `objectCarrying` | Walking with hands reaching out |
| 22 | `crouch` | Crouched walking |
| 23 | `happyDance` | Dancing walk (only walk forward) |
| 24 | `zombie` | Zombie walk |
| 25 | `point` | Walking with hands pointing |
| 26 | `scared` | Scared walk |

### `movement_direction`

| Property | Value |
|----------|-------|
| **Shape** | `[1, 3]` |
| **Dtype** | `float32` |
| **Description** | Desired direction of movement in the MuJoCo world frame |

A 3D direction vector `(x, y, z)` in the Z-up world coordinate system indicating where the robot should move. It is recommended to pass a normalized vector for good practice, though the model normalizes internally. Speed is controlled by `target_vel` and `mode`, not by the magnitude of this vector.

- The planner uses the `(x, y)` components (horizontal plane) for computing the target root trajectory via a critically-damped spring model.
- When the magnitude is near zero (`< 1e-5`), the model falls back to using the `facing_direction` with a small scaling factor for in-place turning.


### `facing_direction`

| Property | Value |
|----------|-------|
| **Shape** | `[1, 3]` |
| **Dtype** | `float32` |
| **Description** | Desired facing (heading) direction in the MuJoCo world frame |

A 3D direction vector `(x, y, z)` indicating which direction the robot's torso should face. The target heading angle is computed as `atan2(y, x)` from this vector. Like `movement_direction`, this does not need to be normalized.

This is independent of `movement_direction` — the robot can walk in one direction while facing another (e.g., strafing).


### `height`

| Property | Value |
|----------|-------|
| **Shape** | `[1]` |
| **Dtype** | `float32` |
| **Description** | Desired root height for height-aware behaviors |

Controls the target pelvis height for modes that support variable height (e.g., `squat`, `kneelTwoLeg`, `kneelOneLeg`, `lyingFacedown`). When a positive value is provided, the model searches the reference clip's keyframes and selects the one whose root height is closest to the requested value, using it as the target pose for motion generation.

| Value | Behavior |
|-------|----------|
| `< 0.0` | Height control disabled; use the randomly-selected keyframe from the reference clip |
| `>= 0.0` | Find the closest height keyframe in the reference clip and use it as the target pose (meters) |


<div id="advanced-inputs"></div>

## Advanced Inputs

These inputs are managed internally by the C++ deployment stack and **should not be modified** under normal operation. They are documented here for completeness and for advanced users who need to build custom integrations.

### `random_seed`

| Property | Value |
|----------|-------|
| **Shape** | `[1]` |
| **Dtype** | `int64` |
| **Description** | Seed for controlling network randomness |


### `has_specific_target`

| Property | Value |
|----------|-------|
| **Shape** | `[1, 1]` |
| **Dtype** | `int64` |
| **Description** | Flag indicating whether specific waypoint targets are provided |

| Value | Behavior |
|-------|----------|
| `0` | Ignore `specific_target_positions` and `specific_target_headings`; use `movement_direction` / `facing_direction` |
| `1` | Use the provided specific target positions and headings as waypoint constraints |

When enabled, the spring model's target root position and heading are overridden by the values in `specific_target_positions` and `specific_target_headings`.


### `specific_target_positions`

| Property | Value |
|----------|-------|
| **Shape** | `[1, 4, 3]` |
| **Dtype** | `float32` |
| **Description** | 4 waypoint positions in MuJoCo world coordinates |

Each waypoint is a 3D position `(x, y, z)` in the Z-up world frame. The 4 waypoints correspond to 4 frames (one token's worth) of target root positions. Only used when `has_specific_target = 1`.


### `specific_target_headings`

| Property | Value |
|----------|-------|
| **Shape** | `[1, 4]` |
| **Dtype** | `float32` |
| **Description** | 4 waypoint heading angles in radians |

Target heading (yaw) angles for each of the 4 waypoint frames. These are absolute angles in the Z-up world frame, measured as rotation around the Z-axis. Only used when `has_specific_target = 1`. The last waypoint's heading (`[:, -1]`) is used as the primary target heading for the spring model.


### `allowed_pred_num_tokens`

| Property | Value |
|----------|-------|
| **Shape** | `[1, K]` where `K = max_tokens - min_tokens + 1` |
| **Dtype** | `int64` |
| **Description** | Binary mask controlling the allowed prediction horizon |

A binary mask where each element corresponds to a possible number of predicted tokens. Index `i` maps to `min_tokens + i` tokens. A value of `1` means that prediction length is allowed; `0` means it is disallowed.

Since each token represents 4 frames, the prediction horizon in frames is `num_tokens * 4`. In our default planner we have `min_tokens = 6` and `max_tokens = 16`:

| Index | Tokens | Frames |
|-------|--------|--------|
| 0 | 6 | 24 |
| 1 | 7 | 28 |
| 2 | 8 | 32 |
| 3 | 9 | 36 |
| 4 | 10 | 40 |
| 5 | 11 | 44 |
| 6 | 12 | 48 |
| 7 | 13 | 52 |
| 8 | 14 | 56 |
| 9 | 15 | 60 |
| 10 | 16 | 64 |

---

## Output Tensors

### `mujoco_qpos`

| Property | Value |
|----------|-------|
| **Shape** | `[1, N, 36]` |
| **Dtype** | `float32` |
| **Description** | Predicted motion sequence as MuJoCo `qpos` frames |

The primary output: a sequence of whole-body pose frames in the same 36-dimensional MuJoCo `qpos` format as the input (see {ref}`context_mujoco_qpos <context_mujoco_qpos>` for the dimension layout).

```{admonition} Important: Use num_pred_frames to Truncate
:class: warning
The output tensor `mujoco_qpos` is **not truncated** — it contains the full padded buffer. Only the first `num_pred_frames` frames are valid predictions. When consuming this output, always slice:
```

```python
valid_qpos = mujoco_qpos[:, :num_pred_frames, :]
```

The poses are in the **global MuJoCo world frame** (not canonicalized). The model internally handles canonicalization, inference, and coordinate conversion, then transforms the output back to the original world frame. The first 4 predicted frames are blended with the input context for smooth transitions.

The root quaternion in the output uses `(w, x, y, z)` ordering (MuJoCo convention).


### `num_pred_frames`

| Property | Value |
|----------|-------|
| **Shape** | scalar |
| **Dtype** | `int64` |
| **Description** | Number of valid predicted frames in the `mujoco_qpos` output |

This value equals `num_pred_tokens * 4`, where `num_pred_tokens` is the number of motion tokens the model decided to generate (constrained by `allowed_pred_num_tokens`). Use this value to slice the `mujoco_qpos` output.

---

## Internal Pipeline

1. **Canonicalization** — The input qpos is transformed to a body-relative frame by removing the first frame's heading rotation and horizontal position. This helps the model generalize across different starting orientations and positions.

2. **Spring Model** — A critically-damped spring model generates smooth target root trajectories and heading angles from the high-level commands, using mode-dependent average velocities from the training clips.

3. **Target Pose Selection** — Based on the `mode` and `random_seed`, a target pose is fetched from the pre-loaded clip library and aligned (rotated/translated) to match the spring model's predicted target position and heading.

4. **Motion Inference** — The core motion model fills in the motion between the context (current state) and target (desired future state), producing a natural transition.

5. **Post-processing** — The output is converted back to MuJoCo qpos in the original world frame, and the first 4 frames are blended with the input context for smooth transitions.

---

## Deployment Integration

This section describes how the C++ deployment stack uses the planner at runtime. Understanding this is useful for building custom integrations or modifying the replan behavior.

### Threading Model

The planner runs on a **dedicated thread at 10 Hz** (`planner_dt = 0.1s`), separate from the control loop (50 Hz) and input thread (100 Hz). The planner thread:

1. Reads the latest `MovementState` from a thread-safe buffer (written by the input interface).
2. Decides whether a replan is needed.
3. If so, calls `UpdatePlanning()` which runs TensorRT inference.
4. Stores the result in a shared buffer that the control thread picks up on its next tick.

### Initialization

When the planner is first enabled (e.g., pressing **ENTER** on the keyboard interface), the planner thread:

1. Reads the robot's current base quaternion and joint positions from the latest `LowState`.
2. Calls `Initialize()`, which:
   - Sets up a 4-frame context at the default standing height with zero-yaw orientation.
   - Runs an initial inference with `IDLE` mode and no movement.
   - Resamples the 30 Hz output to 50 Hz.
3. The control thread detects the new planner motion and switches `current_motion_` to the planner output.

### Context Construction

The planner requires a 4-frame context (`context_mujoco_qpos` of shape `[1, 4, 36]`). During operation, this context is sampled from the **current planner motion** (not the robot state):

- The context starts at `gen_frame = current_frame + motion_look_ahead_steps` (default look-ahead = 2 frames at 50 Hz).
- 4 frames are sampled at 30 Hz intervals from this starting point.
- Joint positions, body positions, and quaternions are linearly interpolated (quaternions via slerp) between 50 Hz motion frames to produce the 30 Hz context samples.

### Replan Logic

Not every planner tick triggers a replan. The decision follows this priority:

**1. Always replan when** (regardless of static/non-static mode):
- Locomotion mode changed
- Facing direction changed
- Height changed

**2. For non-static modes only**, also replan when any of the following is true:
- Movement speed changed
- Movement direction changed
- Periodic replan timer expired **and** movement speed is non-zero

Static modes (Idle, Squat, Kneel, Lying, Idle Boxing) **never** trigger replans from the second category — they only replan on mode/facing/height changes from the first category.

**Replan intervals** (periodic timer) vary by locomotion type to balance responsiveness and computational cost:

| Locomotion Type | Replan Interval |
|----------------|-----------------|
| Running | 0.1 s (every planner tick) |
| Crawling | 0.2 s |
| Boxing (punches, hooks) | 1.0 s |
| All others (walk, squat, styled, etc.) | 1.0 s |

The periodic timer only triggers a replan if the current movement speed is non-zero — a stationary robot in a non-static mode (e.g., Walk mode with speed 0) will not replan on the timer.

### Output Resampling (30 Hz → 50 Hz)

The planner model outputs frames at **30 Hz**. The deployment stack resamples them to **50 Hz** (the control loop rate) using linear interpolation:

- For each 50 Hz frame, compute the corresponding fractional 30 Hz frame index.
- Linearly interpolate joint positions and body positions between the two nearest 30 Hz frames.
- Slerp-interpolate body quaternions.
- Compute joint velocities by finite differencing the resampled positions (`(pos[t+1] - pos[t]) * 50`).

The resampled motion is stored in `planner_motion_50hz_` and has `num_pred_frames * 50/30` frames (rounded down).

### Animation Blending

When a new planner output arrives while the previous one is still playing, the control thread **blends** the old and new animations over an 8-frame cross-fade:

1. The old animation is rebased so `current_frame` maps to frame 0.
2. The new animation is aligned to start at `gen_frame - current_frame` in the rebased timeline.
3. Over 8 frames starting from the blend point, a linearly increasing weight `w_new` (0 → 1) is applied:
   - Joint positions/velocities: `w_old * old + w_new * new`
   - Body positions: `w_old * old + w_new * new`
   - Body quaternions: `slerp(old, new, w_new)`
4. After the blend region, the new animation takes over completely.
5. `current_frame` is reset to 0 on the blended result.

This ensures smooth transitions between successive planner outputs without visible discontinuities.

### TensorRT Acceleration

The planner runs via **TensorRT** with CUDA graph capture for low-latency inference:

1. At startup, the ONNX model is converted to a TensorRT engine (cached on disk).
2. A **CUDA graph** is captured during initialization — this records the entire inference pass (input copy → kernel launches → output copy) as a single replayable graph.
3. On each replan, inputs are copied to GPU via pinned memory (`TPinnedVector`), the CUDA graph is launched, and outputs are copied back.
4. FP16 precision is supported via `--planner-precision 16` (default is FP32).

### Planner Model Versions

The deployment stack supports multiple planner model versions, auto-detected from the model filename:

| Version | Inputs | Modes | Description |
|---------|--------|-------|-------------|
| V0 | 6 | 4 (Idle, Slow Walk, Walk, Run) | Basic locomotion only |
| V1 | 11 | 20 | Adds squat/kneel/boxing/styled walks + height control + waypoint targets |
| V2 | 11 | 27 | All V1 modes + additional styled walking modes |

The version is determined by the presence of `V0`, `V1`, or `V2` in the planner model filename. Version determines:
- The number of input tensors (6 vs 11)
- The valid range of `mode` values
- Whether `height`, `has_specific_target`, `specific_target_positions`, `specific_target_headings`, and `allowed_pred_num_tokens` inputs are used

---

## Model Properties

The exported ONNX model has the following properties:

- **ONNX opset version**: 17
- **Batch size**: 1 (fixed)

The model is distributed as a single `.onnx` file, along with a `.pt` file containing reference input/output tensors that can be used for validation.

```{admonition} Coming Soon
:class: note
The training code, export tooling, and a full technical report will be released soon. Stay tuned for updates.
```

---

## Usage Example

```python
import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("kinematic_planner.onnx")

# Primary inputs
inputs = {
    "context_mujoco_qpos": current_qpos_buffer.astype(np.float32),           # [1, 4, 36]
    "target_vel": np.array([-1.0], dtype=np.float32),                         # -1.0 = use mode default
    "mode": np.array([2], dtype=np.int64),                                    # 2 = walk
    "movement_direction": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),      # forward
    "facing_direction": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),        # face forward
    "height": np.array([-1.0], dtype=np.float32),                             # -1.0 = disabled

    # Advanced inputs (typically managed by the C++ stack)
    "random_seed": np.array([1234], dtype=np.int64),
    "has_specific_target": np.array([[0]], dtype=np.int64),
    "specific_target_positions": np.zeros([1, 4, 3], dtype=np.float32),
    "specific_target_headings": np.zeros([1, 4], dtype=np.float32),
    "allowed_pred_num_tokens": np.ones([1, 11], dtype=np.int64),              # K = 11 for default model
}

# Run inference
mujoco_qpos, num_pred_frames = session.run(None, inputs)

# Extract valid frames only
num_frames = int(num_pred_frames)
predicted_motion = mujoco_qpos[0, :num_frames, :]  # [num_frames, 36]

# Each row: [x, y, z, qw, qx, qy, qz, 29 joint angles in radians]
for frame in predicted_motion:
    root_pos = frame[:3]
    root_quat = frame[3:7]       # (w, x, y, z)
    joint_angles = frame[7:36]   # 29 DOF positions
```
