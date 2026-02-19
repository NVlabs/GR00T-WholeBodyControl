# Observation Configuration

This page is the complete reference for configuring observations in the deployment system. It covers the YAML configuration format, the encoder system, every available observation type, and how to create your own custom observations.

(obs-config-format)=
## Configuration Format

Observations are configured via a YAML file passed with `--obs-config <path>`. Each observation has a `name` (must match a registered observation) and an `enabled` flag.

### Basic Structure

```yaml
observations:
  - name: "motion_joint_positions"
    enabled: true
  - name: "motion_joint_velocities"
    enabled: true
  - name: "motion_anchor_orientation"
    enabled: true
  - name: "base_angular_velocity"
    enabled: true
  - name: "body_joint_positions"
    enabled: true
  - name: "body_joint_velocities"
    enabled: true
  - name: "last_actions"
    enabled: true
```

**Key rules:**

- Observations are concatenated **in the order listed** to form the policy input vector.
- Offsets are calculated automatically — no manual offset management needed.
- The **total dimension** of all enabled observations must match your ONNX model's input size.
- Disabled observations (`enabled: false`) are skipped entirely.
- Reordering entries changes the layout of the input tensor (offsets shift accordingly).

(obs-config-encoder)=
### With Encoder (Token-Based Policies)

For policies that use an encoder to compress observations into a compact token, add an `encoder:` section:

```yaml
observations:
  - name: "token_state"           # Encoder outputs (dimension set below)
    enabled: true
  - name: "base_angular_velocity" # Direct observations
    enabled: true
  - name: "body_joint_positions"
    enabled: true
  - name: "body_joint_velocities"
    enabled: true
  - name: "last_actions"
    enabled: true

encoder:
  dimension: 64       # Token output dimension
  use_fp16: false     # TensorRT precision for encoder (optional)
  encoder_observations:
    - name: "motion_joint_positions_10frame_step5"
      enabled: true
    - name: "motion_joint_velocities_10frame_step5"
      enabled: true
    - name: "motion_anchor_orientation_10frame_step5"
      enabled: true
    - name: "motion_root_z_position_10frame_step5"
      enabled: true
  encoder_modes:            # Optional: per-mode observation requirements
    - name: "g1"
      mode_id: 0
      required_observations:
        - motion_joint_positions_10frame_step5
        - motion_joint_velocities_10frame_step5
        - motion_anchor_orientation_10frame_step5
        - motion_root_z_position_10frame_step5
```

**Encoder fields:**

| Field | Description |
|---|---|
| `dimension` | Token output dimension (must match encoder ONNX model output). Set to 0 or omit to disable encoder. |
| `use_fp16` | Use FP16 precision for encoder TensorRT engine (default: false). |
| `encoder_observations` | Observations fed to the encoder (superset of all modes). Same name/enabled format as policy observations. |
| `encoder_modes` | *(Optional)* Per-mode observation requirements. Observations not in a mode's `required_observations` are zero-filled, saving computation. |

Run with `--encoder-file <path>` to load the encoder model. If omitted, `token_state` can be set externally via ROS2/ZMQ.

See `policy/observation_config_example.yaml` for a complete annotated example.

### Naming Convention

Multi-frame observations follow the pattern: `{base_name}_{N}frame_step{S}`

- **N** = number of frames gathered (temporal window size)
- **S** = step size between frames (in control ticks at 50 Hz, so step5 = 0.1 s apart)
- Without the suffix = single current frame only

For example, `motion_joint_positions_10frame_step5` gathers 10 frames of joint positions, sampled every 5 ticks (0.1 s), giving a 0.9 s look-ahead window. If future frames exceed the motion length, the last frame is repeated.

---

## Encoder & Token Observations

These observations relate to the encoder (tokenizer) system. See [With Encoder](obs-config-encoder) above for the YAML format.

| Name | Dim | Description |
|---|---|---|
| `token_state` | config | Encoder output tokens (dimension set by `encoder.dimension` in YAML). Populated by local encoder inference or externally via ZMQ/ROS2. |
| `encoder_mode` | 3 | Current encoder mode ID + 2 zero-padding values. |
| `encoder_mode_4` | 4 | Current encoder mode ID + 3 zero-padding values. |

---

## Motion Reference Observations

Gathered from the currently-active motion sequence (reference motions, planner output, or ZMQ stream). All joint data uses **IsaacLab joint ordering** (29 joints).

### Joint Positions (from motion)

| Name | Dim | Frames | Step | Description |
|---|---|---|---|---|
| `motion_joint_positions` | 29 | 1 | — | Current frame joint positions (rad) |
| `motion_joint_positions_3frame_step1` | 87 | 3 | 1 | 3-frame window, consecutive |
| `motion_joint_positions_5frame_step5` | 145 | 5 | 5 | 5-frame window, 0.1 s apart |
| `motion_joint_positions_10frame_step1` | 290 | 10 | 1 | 10-frame window, consecutive |
| `motion_joint_positions_10frame_step5` | 290 | 10 | 5 | 10-frame window, 0.1 s apart |
| `motion_joint_positions_lowerbody_10frame_step1` | 120 | 10 | 1 | Lower-body joints only (12 joints), consecutive |
| `motion_joint_positions_lowerbody_10frame_step5` | 120 | 10 | 5 | Lower-body joints only, 0.1 s apart |
| `motion_joint_positions_wrists_10frame_step1` | 60 | 10 | 1 | Wrist joints only (6 joints), consecutive |
| `motion_joint_positions_wrists_2frame_step1` | 12 | 2 | 1 | Wrist joints only, 2 consecutive frames |

```{note}
When upper-body control is active (e.g., via ZMQ/ROS2 teleoperation), the upper-body joint positions in these observations are replaced with the externally-provided targets.
```

### Joint Velocities (from motion)

| Name | Dim | Frames | Step | Description |
|---|---|---|---|---|
| `motion_joint_velocities` | 29 | 1 | — | Current frame joint velocities (rad/s). Zero when not playing. |
| `motion_joint_velocities_3frame_step1` | 87 | 3 | 1 | 3-frame window, consecutive |
| `motion_joint_velocities_5frame_step5` | 145 | 5 | 5 | 5-frame window, 0.1 s apart |
| `motion_joint_velocities_10frame_step1` | 290 | 10 | 1 | 10-frame window, consecutive |
| `motion_joint_velocities_10frame_step5` | 290 | 10 | 5 | 10-frame window, 0.1 s apart |
| `motion_joint_velocities_lowerbody_10frame_step1` | 120 | 10 | 1 | Lower-body joints only, consecutive |
| `motion_joint_velocities_lowerbody_10frame_step5` | 120 | 10 | 5 | Lower-body joints only, 0.1 s apart |
| `motion_joint_velocities_wrists_10frame_step1` | 60 | 10 | 1 | Wrist joints only, consecutive |

### Anchor Orientation (from motion)

Heading-corrected relative rotation from the robot's current base orientation to the reference motion orientation. Output is the first two columns of the 3×3 rotation matrix (6 values per frame).

| Name | Dim | Frames | Step | Description |
|---|---|---|---|---|
| `motion_anchor_orientation` | 6 | 1 | — | Current frame anchor orientation |
| `motion_anchor_orientation_10frame_step1` | 60 | 10 | 1 | 10-frame window, consecutive |
| `motion_anchor_orientation_10frame_step5` | 60 | 10 | 5 | 10-frame window, 0.1 s apart |

### Root Z Position (from motion)

| Name | Dim | Frames | Step | Description |
|---|---|---|---|---|
| `motion_root_z_position` | 1 | 1 | — | Current frame root height (m) |
| `motion_root_z_position_3frame_step1` | 3 | 3 | 1 | 3-frame window, consecutive |
| `motion_root_z_position_10frame_step1` | 10 | 10 | 1 | 10-frame window, consecutive |
| `motion_root_z_position_10frame_step5` | 10 | 10 | 5 | 10-frame window, 0.1 s apart |

---

## SMPL Observations

Gathered from SMPL data in the motion sequence (optional — requires motions with `smpl_joint.csv` / `smpl_pose.csv`).

### SMPL Joint Positions

3D positions per SMPL joint (24 joints × 3 = 72 per frame).

| Name | Dim | Frames | Step | Description |
|---|---|---|---|---|
| `smpl_joints` | 72 | 1 | — | Current frame, all 24 SMPL joints |
| `smpl_joints_2frame_step1` | 144 | 2 | 1 | 2 consecutive frames |
| `smpl_joints_5frame_step5` | 360 | 5 | 5 | 5-frame window, 0.1 s apart |
| `smpl_joints_10frame_step1` | 720 | 10 | 1 | 10-frame window, consecutive |
| `smpl_joints_10frame_step5` | 720 | 10 | 5 | 10-frame window, 0.1 s apart |
| `smpl_joints_lower_10frame_step1` | 270 | 10 | 1 | Lower-body SMPL joints only (9 joints), consecutive |

### SMPL Poses (Axis-Angle)

3D axis-angle per SMPL body part (21 poses × 3 = 63 per frame).

| Name | Dim | Frames | Step | Description |
|---|---|---|---|---|
| `smpl_pose` | 63 | 1 | — | Current frame, all 21 SMPL poses |
| `smpl_pose_5frame_step5` | 315 | 5 | 5 | 5-frame window, 0.1 s apart |
| `smpl_pose_10frame_step1` | 630 | 10 | 1 | 10-frame window, consecutive |
| `smpl_pose_10frame_step5` | 630 | 10 | 5 | 10-frame window, 0.1 s apart |
| `smpl_elbow_wrist_poses_10frame_step1` | 120 | 10 | 1 | Elbow + wrist poses only (4 parts), consecutive |

### SMPL Aliases

These use the same gatherers as the motion observations but are intended for SMPL-based policies:

| Name | Dim | Frames | Step | Description |
|---|---|---|---|---|
| `smpl_root_z_10frame_step1` | 10 | 10 | 1 | Root height, 10 consecutive frames |
| `smpl_anchor_orientation_10frame_step1` | 60 | 10 | 1 | Anchor orientation, 10 consecutive frames |
| `smpl_anchor_orientation_2frame_step1` | 12 | 2 | 1 | Anchor orientation, 2 consecutive frames |

---

## VR Tracking Observations

VR 3-point and 5-point tracking data. When an external source (ZMQ/ROS2) provides VR data, buffered values are used directly. Otherwise, positions and orientations are computed from the motion sequence's body data and normalised to the root body frame.

### VR 3-Point

| Name | Dim | Description |
|---|---|---|
| `vr_3point_local_target` | 9 | 3-point positions in root frame: `[left_wrist xyz, right_wrist xyz, head xyz]` |
| `vr_3point_local_target_compliant` | 9 | Same as above (identical during teleoperation) |
| `vr_3point_local_orn_target` | 12 | 3-point orientations in root frame: `[left quat wxyz, right quat wxyz, head quat wxyz]` |
| `vr_3point_compliance` | 3 | Compliance values: `[left_arm, right_arm, head]`. Keyboard-controlled (g/h/b/v keys), range [0.0, 0.5]. |

### VR 5-Point

| Name | Dim | Description |
|---|---|---|
| `vr_5point_local_target` | 15 | 5-point positions in root frame: `[left_wrist, right_wrist, head, left_ankle, right_ankle]` × xyz |
| `vr_5point_local_orn_target` | 20 | 5-point orientations in root frame: 5 quaternions × wxyz |

---

## Robot State History Observations

Gathered from the StateLogger ring buffer (measured sensor data from the real robot). These provide temporal context by sampling past states.

### Single-Frame (Current State)

| Name | Dim | Description |
|---|---|---|
| `base_angular_velocity` | 3 | IMU angular velocity (rad/s): `[roll_rate, pitch_rate, yaw_rate]` |
| `body_joint_positions` | 29 | Current joint positions from encoders (rad, IsaacLab order) |
| `body_joint_velocities` | 29 | Current joint velocities from encoders (rad/s, IsaacLab order) |
| `last_actions` | 29 | Previous policy output (normalised action values) |
| `gravity_dir` | 3 | Gravity direction in body frame (computed from base IMU quaternion) |

### Multi-Frame History (4 frames, step 1)

| Name | Dim | Description |
|---|---|---|
| `his_body_joint_positions_4frame_step1` | 116 | Joint positions: 4 consecutive ticks (29 × 4) |
| `his_body_joint_velocities_4frame_step1` | 116 | Joint velocities: 4 consecutive ticks |
| `his_last_actions_4frame_step1` | 116 | Past actions: 4 consecutive ticks |
| `his_base_angular_velocity_4frame_step1` | 12 | Angular velocity: 4 consecutive ticks (3 × 4) |
| `his_gravity_dir_4frame_step1` | 12 | Gravity direction: 4 consecutive ticks |

### Multi-Frame History (10 frames, step 1)

| Name | Dim | Description |
|---|---|---|
| `his_body_joint_positions_10frame_step1` | 290 | Joint positions: 10 consecutive ticks (29 × 10) |
| `his_body_joint_velocities_10frame_step1` | 290 | Joint velocities: 10 consecutive ticks |
| `his_last_actions_10frame_step1` | 290 | Past actions: 10 consecutive ticks |
| `his_base_angular_velocity_10frame_step1` | 30 | Angular velocity: 10 consecutive ticks (3 × 10) |
| `his_gravity_dir_10frame_step1` | 30 | Gravity direction: 10 consecutive ticks |

---

## Creating Custom Observations

You can add your own observation types by modifying the C++ source. The observation system is built around a **registry pattern** — you write a gatherer function, register it with a name and dimension, and then use that name in your YAML config.

All observation code lives in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` inside the `G1Deploy` class.

### Step 1: Write a Gatherer Function

A gatherer function reads from internal state (sensor data, motion data, etc.) and writes its output into a target buffer at a given offset. The signature is:

```cpp
bool MyObservation(std::vector<double>& target_buffer, size_t offset) {
    // Write your observation values into target_buffer starting at offset.
    // Return true on success, false on failure (will stop the control loop).
}
```

**Available data sources inside G1Deploy** (see member variables in `g1_deploy_onnx_ref.cpp` for the full list):

| Source | Description |
|---|---|
| `state_logger_` | Ring buffer of past robot states — IMU, joints, velocities, actions, hand states, token state |
| `current_motion_` / `current_frame_` | Currently-active motion sequence and playback cursor |
| `operator_state` | Operator control flags (`.play`, `.start`, `.stop`) |
| `vr_*_buffer_`, `left_hand_joint_buffer_`, etc. | Buffered input interface data — VR tracking, hand joints, compliance, upper-body targets |
| `heading_state_buffer_`, `movement_state_buffer_` | Thread-safe buffers for heading and planner movement commands |

**Example** — a custom observation that outputs the torso IMU angular velocity (3 values):

```cpp
bool GatherTorsoAngularVelocity(std::vector<double>& target_buffer, size_t offset) {
    if (!state_logger_) { return false; }

    auto hist = state_logger_->GetLatest(1);
    if (hist.empty()) { return false; }

    const auto& entry = hist[0];
    target_buffer[offset + 0] = entry.body_torso_ang_vel[0];
    target_buffer[offset + 1] = entry.body_torso_ang_vel[1];
    target_buffer[offset + 2] = entry.body_torso_ang_vel[2];
    return true;
}
```

### Step 2: Register in the Observation Registry

Add your observation to the `GetObservationRegistry()` method in `g1_deploy_onnx_ref.cpp`. Each entry is a tuple of `{name, dimension, gatherer_lambda}`:

```cpp
std::vector<ObservationRegistry> GetObservationRegistry() {
    return {
        // ... existing observations ...

        // Your custom observation:
        {"torso_angular_velocity", 3,
         [this](std::vector<double>& buf, size_t offset) {
             return GatherTorsoAngularVelocity(buf, offset);
         }},
    };
}
```

The **name** is the string you'll use in the YAML config. The **dimension** must be exact — the system validates that the total of all enabled observations matches the ONNX model input size.

### Step 3: Use in YAML Config

Once registered, your observation is available like any built-in one:

```yaml
observations:
  - name: "torso_angular_velocity"
    enabled: true
  # ... other observations ...
```

### Tips

- **Dimension must be fixed.** The observation dimension is set at registration time and cannot change at runtime. If you need variable-size data, pad to a fixed maximum.
- **Don't allocate in the hot path.** Gatherer functions run at 50 Hz in the control loop. Avoid `new`, `malloc`, or resizing vectors. Pre-allocate buffers in the constructor or use stack arrays.
- **Return `false` carefully.** Returning `false` from a gatherer stops the entire control loop. Only return `false` for unrecoverable errors. For missing optional data, write zeros and return `true`.
- **Thread safety.** Gatherers run on the control thread. Reading from `state_logger_` and `DataBuffer` objects is thread-safe. Accessing `current_motion_` and `current_frame_` is protected by `current_motion_mutex_` (already held when `GatherObservations()` is called).
- **Multi-frame pattern.** If your observation needs temporal windows, follow the existing `GatherHis*` or `GatherMotion*MultiFrame` patterns — they accept `num_frames` and `step_size` parameters and register multiple variants (e.g., `my_obs`, `my_obs_4frame_step1`, `my_obs_10frame_step5`).
- **Encoder observations.** Custom observations can also be used as encoder inputs. Register them in the same registry — they'll be available for both `observations:` and `encoder_observations:` in the YAML config.
- **Rebuild after changes.** After modifying the C++ source, rebuild with `just build` from the `gear_sonic_deploy/` directory.

---

## Example Configurations

### Minimal (154D — default policy)

```yaml
observations:
  - name: "motion_joint_positions"       # 29D
    enabled: true
  - name: "motion_joint_velocities"      # 29D
    enabled: true
  - name: "motion_anchor_orientation"    # 6D
    enabled: true
  - name: "base_angular_velocity"        # 3D
    enabled: true
  - name: "body_joint_positions"         # 29D
    enabled: true
  - name: "body_joint_velocities"        # 29D
    enabled: true
  - name: "last_actions"                 # 29D
    enabled: true
# Total: 154D
```

### Token-Based Policy with Encoder

```yaml
observations:
  - name: "token_state"                  # 64D (from encoder)
    enabled: true
  - name: "base_angular_velocity"        # 3D
    enabled: true
  - name: "body_joint_positions"         # 29D
    enabled: true
  - name: "body_joint_velocities"        # 29D
    enabled: true
  - name: "last_actions"                 # 29D
    enabled: true

encoder:
  dimension: 64
  use_fp16: false
  encoder_observations:
    - name: "motion_joint_positions_10frame_step5"   # 290D
      enabled: true
    - name: "motion_joint_velocities_10frame_step5"  # 290D
      enabled: true
    - name: "motion_anchor_orientation_10frame_step5" # 60D
      enabled: true
    - name: "motion_root_z_position_10frame_step5"   # 10D
      enabled: true
```

### VR Teleoperation Policy

```yaml
observations:
  - name: "token_state"                         # 64D
    enabled: true
  - name: "vr_3point_local_target"              # 9D
    enabled: true
  - name: "vr_3point_local_orn_target"          # 12D
    enabled: true
  - name: "vr_3point_compliance"                # 3D
    enabled: true
  - name: "base_angular_velocity"               # 3D
    enabled: true
  - name: "body_joint_positions"                # 29D
    enabled: true
  - name: "body_joint_velocities"               # 29D
    enabled: true
  - name: "last_actions"                        # 29D
    enabled: true
```

See [Configuration Format](obs-config-format) above for YAML syntax details.
