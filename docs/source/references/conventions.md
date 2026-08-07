# Coordinate Frame and Rotation Conventions

This page documents the coordinate frame, quaternion, and rotation conventions
used throughout the SONIC codebase. Getting these wrong causes silent bugs —
the robot will move but in the wrong direction or with wrong orientation.

## Coordinate Frames

### Isaac Lab / MuJoCo (simulation)

- **Z-up**: Gravity is along -Z. Ground plane is XY.
- **Right-handed**: X forward, Y left, Z up.
- This is the convention used during training and evaluation.

### SMPL / BVH (human motion data)

- **Y-up**: Gravity is along -Y. Ground plane is XZ.
- When loading SMPL or BVH data, set `smpl_y_up: true` in the motion library
  config. The motion library automatically converts Y-up to Z-up internally.

### Summary

| System | Up axis | Convention |
|--------|---------|------------|
| Isaac Lab | Z | Z-up, right-handed |
| MuJoCo | Z | Z-up, right-handed |
| SMPL body model | Y | Y-up |
| BVH motion files | Y | Y-up |
| Retargeted PKL data | Z | Z-up (already converted) |

## Quaternion Convention

### Scalar-first (wxyz) — default throughout SONIC

The SONIC codebase uses **scalar-first (wxyz)** quaternions everywhere:

```
q = [w, x, y, z]
```

This applies to:

- `gear_sonic/trl/utils/torch_transform.py` — all rotation utilities
- `gear_sonic/isaac_utils/rotations.py` — Isaac Lab rotation helpers (use `w_last=False`)
- Isaac Lab APIs (`body_quat_w`, `root_quat_w`, etc.)
- Motion library internal storage
- Retargeted PKL data (`root_rot` field)

### Scalar-last (xyzw) — scipy only

[SciPy's Rotation class](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
uses **scalar-last (xyzw)** convention:

```
q = [x, y, z, w]
```

This is only used in the **data processing scripts** (`data_process/`) when
calling `scipy.spatial.transform.Rotation`. The scripts convert to wxyz
before saving:

```python
# In data processing (scipy xyzw → wxyz for storage)
root_quat_xyzw = Rotation.from_euler("xyz", euler_angles).as_quat()  # scipy: xyzw
root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]                    # convert to wxyz
```

### The `w_last` parameter

Functions in `gear_sonic/isaac_utils/rotations.py` accept a `w_last` boolean:

```python
quat_rotate(q, v, w_last=False)   # q is wxyz (scalar-first) — this is the default
quat_rotate(q, v, w_last=True)    # q is xyzw (scalar-last)
```

**Always use `w_last=False`** unless you're interfacing with scipy or a system
that explicitly uses xyzw.

### Quick reference

| System | Convention | Order | Identity |
|--------|-----------|-------|----------|
| SONIC (torch_transform.py) | wxyz | `[w, x, y, z]` | `[1, 0, 0, 0]` |
| Isaac Lab | wxyz | `[w, x, y, z]` | `[1, 0, 0, 0]` |
| SciPy | xyzw | `[x, y, z, w]` | `[0, 0, 0, 1]` |
| MuJoCo | wxyz | `[w, x, y, z]` | `[1, 0, 0, 0]` |
| ROS | xyzw | `[x, y, z, w]` | `[0, 0, 0, 1]` |

### Converting between conventions

```python
# wxyz → xyzw
q_xyzw = q_wxyz[..., [1, 2, 3, 0]]

# xyzw → wxyz
q_wxyz = q_xyzw[..., [3, 0, 1, 2]]
```

## Rotation Representations

The codebase uses multiple rotation representations depending on context:

| Representation | Shape | Used in |
|---------------|-------|---------|
| Quaternion (wxyz) | `(..., 4)` | Simulation, motion library, observations |
| Axis-angle | `(..., 3)` | `pose_aa` field in motion PKLs |
| Rotation matrix | `(..., 3, 3)` | Forward kinematics, 6D rotation encoding |
| 6D rotation | `(..., 6)` | Some observation terms (first 2 columns of rotation matrix) |
| Euler angles | `(..., 3)` | CSV motion data input (converted immediately) |

### Axis-angle in motion data

The `pose_aa` field in retargeted PKL files stores per-body **local** rotations
as axis-angle vectors. The direction is the rotation axis, the magnitude is
the angle in radians:

```python
pose_aa  # (T, num_bodies, 3) — axis-angle per body, MuJoCo body order
```

## Joint Ordering

Isaac Lab and MuJoCo traverse the kinematic tree in different orders. The
codebase provides bidirectional index mappings per robot:

```python
from gear_sonic.envs.manager_env.robots.g1 import (
    G1_ISAACLAB_TO_MUJOCO_DOF,   # Reorder DOFs: IsaacLab → MuJoCo
    G1_MUJOCO_TO_ISAACLAB_DOF,   # Reorder DOFs: MuJoCo → IsaacLab
)

# Convert joint positions from IsaacLab order to MuJoCo order:
mujoco_joints = isaaclab_joints[..., G1_ISAACLAB_TO_MUJOCO_DOF]
```

Motion PKL data (`dof`, `pose_aa`) is stored in **MuJoCo order**. Isaac Lab
simulation uses **IsaacLab order**. The training pipeline handles the conversion
automatically via `order_converter.py`.

See [Training on New Embodiments](../user_guide/new_embodiments.md) for how to
define these mappings for a new robot.
