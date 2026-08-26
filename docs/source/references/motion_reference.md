# Motion Reference Data

This page describes the motion reference data format used by the C++ deployment stack, how to create your own reference motions, and how to verify and deploy them.

The deployment stack plays back pre-loaded **reference motions** — sequences of joint positions, velocities, and full body kinematics that the policy tracks. These motions are stored as CSV files in a structured folder hierarchy. You can generate them from any source (motion capture, simulation, retargeting pipeline, etc.) as long as the output matches the format described below.

---

## Folder Structure

Each motion dataset is a directory containing one subfolder per motion clip. The C++ reader (`MotionDataReader`) auto-discovers all subfolders at startup.

```
reference/my_motions/
├── motion_name_1/
│   ├── joint_pos.csv          # Joint positions
│   ├── joint_vel.csv          # Joint velocities
│   ├── body_quat.csv          # Body quaternions
│   ├── body_pos.csv           # Body positions
│   ├── metadata.txt           # Body part indexes
│   ├── body_lin_vel.csv       # Body linear velocities
│   ├── body_ang_vel.csv       # Body angular velocities
│   ├── smpl_joint.csv         # SMPL joint positions
│   ├── smpl_pose.csv          # SMPL body poses
│   └── info.txt               # Detailed motion information
└── motion_name_2/
    └── ...
```

The C++ reader scans the base directory for subfolders, reads each subfolder as one motion, and validates frame-count consistency across all files in that folder. 
---

## File Formats

The C++ reader loads whichever files are present and skips missing files gracefully. However, **in practice**, most policies require:
- `joint_pos.csv`, `joint_vel.csv` — for joint-based motion tracking
- `body_quat.csv` — for anchor orientation observations (the control loop will stop if this is missing when gathering observations)
- `body_pos.csv` — for heading computation and VR 3-point observations
- `metadata.txt` — for body part index alignment when body data is present

A motion must have **at least one valid data source** (joint, body, or SMPL) to load at startup.

### `joint_pos.csv`

Joint positions in **IsaacLab order** (29 joints). Each row is one timestep at 50 Hz. The first row is a header.

| Column | Description |
|--------|-------------|
| `joint_0` … `joint_28` | Joint angles in radians (IsaacLab ordering) |

Shape: `(timesteps, 29)`

### `joint_vel.csv`

Joint velocities in **IsaacLab order** (29 joints). Each row is one timestep at 50 Hz. Frame count must match `joint_pos.csv`.

| Column | Description |
|--------|-------------|
| `joint_vel_0` … `joint_vel_28` | Joint angular velocities in rad/s (IsaacLab ordering) |

Shape: `(timesteps, 29)`

### `body_pos.csv`

Body part positions in the **world frame**. Each body contributes 3 columns (x, y, z). The number of bodies varies per motion. Needed for heading computation and VR 3-point observations.

| Column | Description |
|--------|-------------|
| `body_0_x`, `body_0_y`, `body_0_z` | Position of body 0 (root/pelvis) in meters |
| `body_1_x`, `body_1_y`, `body_1_z` | Position of body 1 in meters |
| … | … |

Shape: `(timesteps, num_bodies * 3)`

**We assume the root/pelvis is always at column group 0** (the first 3 columns).

### `body_quat.csv`

Body part orientations as quaternions in the **world frame**. Each body contributes 4 columns. The quaternion ordering is **(w, x, y, z)**. **Required for most policies** — the `motion_anchor_orientation` observation (used by most policies) will fail and stop the control system if this file is missing.

| Column | Description |
|--------|-------------|
| `body_0_w`, `body_0_x`, `body_0_y`, `body_0_z` | Quaternion of body 0 (root/pelvis) |
| `body_1_w`, `body_1_x`, `body_1_y`, `body_1_z` | Quaternion of body 1 |
| … | … |

Shape: `(timesteps, num_bodies * 4)`

**We assume the root/pelvis is always at column group 0** (the first 4 columns).

```{note}
The number of bodies in `body_quat.csv` can differ from `body_pos.csv`. The C++ reader tracks them independently (`num_bodies` vs `num_body_quaternions`). However, the root body (first column group) must be present for heading computation to work. You can use zero if you don't need root pos.
```

### `metadata.txt`

Contains the **body part indexes** array, which maps each column group in `body_pos.csv` / `body_quat.csv` to the corresponding IsaacLab body index. This is needed when body data is present.

```
Metadata for: motion_name
==============================

Body part indexes:
[ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]

Total timesteps: 497
```

The C++ reader parses `Body part indexes:` followed by a line of space-separated integers in brackets. For example, `[0, 4, 10, 18, ...]` means column group 0 → IsaacLab body 0 (pelvis/root), column group 1 → body 4, etc.

For a **root-only** motion (only 1 body), use:

```
Body part indexes:
[0]
```

### `body_lin_vel.csv` / `body_ang_vel.csv`

Body part linear and angular velocities in the world frame. Same layout as `body_pos.csv` (3 columns per body). The number of bodies must match `body_pos.csv`.

### `smpl_joint.csv`

SMPL joint positions (typically 24 joints × 3 coordinates). Each row is one timestep.

Shape: `(timesteps, num_smpl_joints * 3)`

### `smpl_pose.csv`

SMPL body poses in axis-angle representation (typically 21 poses × 3 coordinates). Each row is one timestep.

Shape: `(timesteps, num_smpl_poses * 3)`

```{note}
The **current reference motion tracking pipeline uses joint-based tracking only** (encoder mode 0). To enable SMPL-based reference tracking (encoder mode 2), you would need to modify the code to detect the presence of SMPL data and switch the encoder mode accordingly.
```

### `info.txt`

Human-readable summary with shapes, dtypes, and value ranges. Not read by the C++ stack — purely for documentation.

---

## Creating Your Own Reference Motions

You can generate reference motions from any source — the only requirement is producing CSV files in the format above. Common approaches:

1. **Motion capture retargeting** — retarget human mocap to the G1 model, export joint positions/velocities and body kinematics.
2. **Simulation recording** — record joint states from an IsaacLab or MuJoCo simulation at 50 Hz.
3. **Procedural generation** — programmatically create joint trajectories.

### Minimal Files Needed

The **minimum** set of files to create a working motion for SONIC policy:

1. **`joint_pos.csv`** — 29 joint positions (IsaacLab order), header + one row per timestep
2. **`joint_vel.csv`** — 29 joint velocities (IsaacLab order), header + one row per timestep
3. **`body_quat.csv`** — Root quaternion (w, x, y, z), header + one row per timestep
4. **`body_pos.csv`** — Root position (x, y, z), header + one row per timestep. You can use all zeros if you don't need position tracking.
5. **`metadata.txt`** — Body part indexes (just `[0]` for root-only)

**Example files:**

`joint_pos.csv`:
```
joint_0,joint_1,joint_2,...,joint_28
0.128441,0.102713,0.020116,...,0.045231
0.130124,0.104532,0.021045,...,0.046112
...
```

`joint_vel.csv`:
```
joint_vel_0,joint_vel_1,...,joint_vel_28
0.143671,0.143864,...,0.012345
...
```

`body_quat.csv` (root quaternion only):
```
body_0_w,body_0_x,body_0_y,body_0_z
0.999123,0.000456,0.001234,0.040567
...
```

`body_pos.csv` (root position, can be all zeros):
```
body_0_x,body_0_y,body_0_z
0.000000,0.000000,0.000000
...
```

`metadata.txt`:
```
Metadata for: my_motion
==============================

Body part indexes:
[0]

Total timesteps: 100
```

This gives you a **root-only** motion (1 body = pelvis/root) that most policies can track.


### Provided Conversion Script

A convenience script `reference/convert_motions.py` is included for converting **joblib pickle** (`.pkl`) files to this format. This is just one possible source — you can use any tool or pipeline that produces the correct CSV output.

```bash
cd gear_sonic_deploy
python reference/convert_motions.py <pkl_file> [output_dir]
```

The pickle should be a dictionary where each key is a motion name and each value contains `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `_body_indexes`, and `time_step_total`.

---

## Verifying Reference Motions

### MuJoCo Visualization

Use the included visualizer to check that the motion looks correct on the G1 model:

```bash
cd gear_sonic_deploy
python visualize_motion.py --motion_dir reference/my_motions/motion_name_1/
```

**Controls:**
- **Space**: Pause / resume playback
- **R**: Reset to frame 0
- **,** / **.**: Step backward / forward one frame
- **-** / **=**: Previous / next motion (if multiple loaded)

Verify that:
- The robot stands upright and does not clip through the floor
- Joint angles look reasonable (no extreme poses)
- The motion plays smoothly without sudden jumps
- Body positions track the expected trajectory

---

## Using Reference Motions

### With `deploy.sh`

Pass the motion directory via `--motion-data`:

```bash
./deploy.sh --motion-data reference/my_motions/ sim
```

Or use the default motions (configured in `deploy.sh`):

```bash
./deploy.sh sim
```

### At Runtime

Once deployed, use the keyboard or gamepad to browse and play motions:

- **T**: Play current motion
- **N / P**: Next / Previous motion
- **R**: Restart from frame 0

See the [Keyboard tutorial](../tutorials/keyboard.md) for the full control reference.

---

## Validation Rules

The C++ reader enforces the following during loading:

- **Frame count consistency**: All CSV files within a motion folder must have the same number of rows (excluding headers). Mismatches cause the motion to be skipped with an error.
- **Joint count consistency**: `joint_pos.csv` and `joint_vel.csv` must have the same number of columns.
- **Body count consistency**: `body_lin_vel.csv` and `body_ang_vel.csv` must have the same number of body columns as `body_pos.csv`.
- **At least one data source**: A motion must have at least some valid data (joint, body, or SMPL) to be loaded.
- **Metadata parsing**: The `metadata.txt` file must contain a `Body part indexes:` line followed by a bracketed list of integers for the motion to have correct body-part alignment.

If a motion fails validation, it is skipped and a warning is printed. The deployment continues with the remaining valid motions.

---

## Notes

- All data is at **50 Hz** (0.02 s per timestep), matching the control loop frequency.
- Joint ordering follows **IsaacLab convention** (not MuJoCo). The C++ stack handles the conversion internally when sending motor commands.
- Body quaternions use **(w, x, y, z)** ordering.
- The first body (column group 0) must correspond to the root/pelvis for heading computation and anchor orientation observations to work correctly.
- While the C++ reader can load motions without `body_quat.csv`, the control loop will fail during observation gathering if the policy observes `motion_anchor_orientation` (which most policies do).
- CSV files must have a **header row** as the first line — the C++ reader skips the first line of every CSV.
- Values are parsed as `double` precision internally.
