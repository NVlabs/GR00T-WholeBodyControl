# Motion Representation

## Overview

MotionBricks represents motion as a normalized feature vector per frame. The representation separates **root motion** (global position and heading of the robot's pelvis) from **body motion** (joint rotations, positions, velocities, and foot contacts). This separation lets the root model and the pose/tokenizer module operate on different subsets of the same representation.

Throughout the paper and codebase, two interchangeable subsets are used:

- **Global** (`GlobalRootGlobalJoints`, 414 dims) — **root model** mostly operates with this representation for precise global root control. This is also what the data loader returns directly.
- **Local** (`LocalRootGlobalJoints`, 413 dims) — used by the **pose/tokenizer module**.

The two subsets share the same 409-dim body features and differ only in how the root is parameterized (5 global vs 4 local dims). They convert losslessly to each other via `dual_rep.global_to_local` / `dual_rep.local_to_global`. In the training loop, batches come out of the loader in the global representation and are converted to local on the fly before being passed to the pose/tokenizer module. Concretely, the `"motion"` tensor in every batch dict is always the **global** motion — per-sample conversion to local happens inside the training step.

The current configuration uses the **DualRootGlobalJoints** representation on the **G1Skeleton34** skeleton (Unitree G1 with 34 joints). The full feature vector is 418-dimensional per frame, composed of the 414-dim global subset and the 413-dim local subset that share the 409-dim body features.

See the MotionBricks paper for the full derivation of the representation.

## Feature Breakdown

All body features are defined in the **global (world) frame**. The `local_` prefix on `local_vel` is a naming holdover — in `DualRootGlobalJoints` (`removing_heading=False`), the velocity is NOT heading-rotated, and its normalization statistics are computed from the same world-frame values.

### Body Features (409 dimensions)

Shared by both global and local root representations.

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| `ric_data` | 99 | Global joint positions with the **projected (XZ) root position** subtracted per frame, for 33 non-root joints. Not heading-canonicalized. |
| `global_rot_data` | 204 | **Global (world-frame)** 6D continuous rotations for all 34 joints.|
| `local_vel` | 102 | **Global-frame** per-joint velocity, computed as finite differences of world positions. |
| `foot_contacts` | 4 | Binary contact states for left ankle, left toe, right ankle, right toe. |

### Global Root Features (5 dimensions)

Used by the global representation subset (consumed by the root model).

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| `global_root_pos` | 3 | XYZ position in world frame (during training and inference, first frame's root XZ is placed at origin). |
| `global_root_heading` | 2 | Root heading as (cos, sin) of the Y-axis rotation angle. |

### Local Root Features (4 dimensions)

Used by the local representation subset (consumed by the pose/tokenizer module). Derived from the global root during the `global_to_local` conversion.

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| `local_root_rot_vel` | 1 | Angular velocity around the Y-axis. |
| `local_root_vel` | 2 | Root translational velocity in the XZ plane, expressed in the root's heading-aligned frame. |
| `global_root_y` | 1 | Root height (Y-axis position in world frame). |

### Combined Dimensions

| Representation | Formula | Total |
|---------------|---------|-------|
| Global subset (`GlobalRootGlobalJoints`) | 5 (global root) + 409 (body) | **414** |
| Local subset (`LocalRootGlobalJoints`) | 4 (local root) + 409 (body) | **413** |
| Full dual (`DualRootGlobalJoints`) | 5 + 4 + 409 | **418** |

The root model uses the **global subset** (414 dims). The pose/tokenizer module uses the **local subset** (413 dims).

## Skeleton: G1Skeleton34

The skeleton defines the kinematic tree. G1Skeleton34 has 34 joints: 32 active joints from the Unitree G1 robot plus 2 dummy toe joints for foot contact detection.

```
pelvis (root)
  |-- left_hip_pitch -- left_hip_roll -- left_hip_yaw -- left_knee
  |     \-- left_ankle_pitch -- left_ankle_roll -- left_toe_base*
  |-- right_hip_pitch -- right_hip_roll -- right_hip_yaw -- right_knee
  |     \-- right_ankle_pitch -- right_ankle_roll -- right_toe_base*
  |-- waist_yaw -- waist_roll -- waist_pitch
        |-- left_shoulder_pitch -- left_shoulder_roll -- left_shoulder_yaw -- left_elbow
        |     \-- left_wrist_roll -- left_wrist_pitch -- left_wrist_yaw -- left_hand_roll
        |-- right_shoulder_pitch -- right_shoulder_roll -- right_shoulder_yaw -- right_elbow
              \-- right_wrist_roll -- right_wrist_pitch -- right_wrist_yaw -- right_hand_roll
```

*Dummy toe joints (not actuated on the real robot).

### MuJoCo Joint Mapping

The MuJoCo model has 29 hinge joints (excluding the free-floating root and toe joints). The output qpos vector is 36-dimensional:

| Indices | Content |
|---------|---------|
| 0-2 | Root translation (x, y, z) |
| 3-6 | Root quaternion (w, x, y, z) |
| 7-35 | 29 joint angles (1 DOF per hinge joint) |

The `mujoco_qpos_converter` class handles the mapping between the 34-joint motion representation and the 29-DOF MuJoCo model, including coordinate system transformation (motion space: Y-up, Z-forward; MuJoCo space: Z-up, X-forward).

## Coordinate Systems

| Space | Up | Forward | Handedness |
|-------|-----|---------|------------|
| Motion | Y | Z | Right-handed |
| MuJoCo | Z | X | Right-handed |

The coordinate transformation between the two:
- Motion X = MuJoCo Y
- Motion Y = MuJoCo Z
- Motion Z = MuJoCo X

## Normalization

All features are z-score normalized before being fed to models:

```
normalized = (feature - mean) / sqrt(std^2 + eps)
```

where `eps = 1e-5` for numerical stability. The `mean.npy` and `std.npy` files are computed per-dimension over the training dataset and stored alongside each model checkpoint in the `stats/motion/` directory.

## Feature Computation Pipeline

MotionBricks does **not** apply a fixed heading canonicalization to its features. Instead, each motion segment is placed at the origin with a heading that is *randomly rotated* at training time and *explicitly chosen by the caller* at inference time. This way the model sees motion in all orientations, so there is nothing to gain from pre-canonicalizing to a fixed frame.

The pipeline, mirroring the order in `compute_motion_features` in `motionlib/core/motion_reps/tools/motion_features.py`, is:

```
Raw motion (local / global joint rotations + root translation)
    │
    ▼
Compute ROOT features (compute_heading_info + compute_heading_features):
  • Global root:  (root XYZ, heading cos/sin)
  • Local root :  (XZ linear velocity, Y angular velocity, root height)
    │
    ▼
Compute BODY features in the WORLD frame (compute_position_features):
  • ric_data        (world joint positions − per-frame root XZ)
  • local_vel       (world-frame finite-difference velocity)
  • foot_contacts   (from position/velocity thresholds)
  • global_rot_data (world-frame 6D rotations)
    │
    ▼
Concatenate per frame → [T, 418];  obtain normalization stats (z-score)
    │
    ▼
The data loader returns the NORMALIZED GLOBAL rep [T, 414].
    │
    ▼
At training / inference time, for each motion segment:
  1. Call `change_first_heading(..., first_heading_angle)`
     - TRAINING:  first_heading_angle ~ Uniform(0, 2π)   → random heading
     - INFERENCE: first_heading_angle = 0                → deterministic
     Effect: rotates every frame so that the first frame faces the target
             heading, AND places the first frame's root XZ at the origin
             (Y / root height is preserved).
  2. If feeding the pose / tokenizer module: convert to LOCAL via
     `dual_rep.global_to_local(...)` (lossless; invertible via `local_to_global`).
```

The inverse pipeline (used at inference time) converts features back to joint positions and rotations, which are then mapped to MuJoCo qpos via the `mujoco_qpos_converter`.
