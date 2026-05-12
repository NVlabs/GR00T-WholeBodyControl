# Training on New Embodiments

SONIC's training pipeline is designed around the Unitree G1 (29 DOF) but can be
extended to other humanoid robots. This guide walks through every file you need
to touch, using the Unitree H2 (31 DOF) as a concrete example.

## What You Need

To train SONIC on a new robot, you need:

1. **Robot model files** — URDF or USD (for Isaac Lab) and MJCF/XML (for the motion library)
2. **Retargeted motion data** — Human motions retargeted to your robot's skeleton (PKL format)
3. **Robot configuration** — Joint/body definitions, actuator parameters, action scales
4. **Experiment config** — Hydra YAML connecting everything together

## Files You Need to Add or Modify

Here is every file that needs attention, in the order you should work through them:

| File | Action | Purpose |
|------|--------|---------|
| `gear_sonic/data/assets/robot_description/urdf/<robot>/` | **Add** | URDF + mesh files for Isaac Lab simulation |
| `gear_sonic/data/assets/robot_description/mjcf/<robot>.xml` | **Add** | MuJoCo XML for motion library forward kinematics |
| `gear_sonic/envs/manager_env/robots/<robot>.py` | **Add** | Robot config: joints, actuators, mappings, action scales |
| `gear_sonic/envs/manager_env/robots/__init__.py` | **Modify** | Import your new robot module |
| `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` | **Modify** | Add robot to `robot_mapping` dict (~line 998) |
| `gear_sonic/trl/utils/order_converter.py` | **Modify** | Add converter class for joint/body reordering |
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_<robot>.yaml` | **Add** | Experiment config |
| Config YAMLs (terminations, rewards, commands) | **Check** | Body names must exist on your robot |

## Step 1: Robot Model Files

Place your URDF and meshes under `gear_sonic/data/assets/robot_description/`:

```
gear_sonic/data/assets/robot_description/
├── urdf/h2/
│   ├── h2.urdf
│   └── meshes/          # STL/OBJ mesh files
└── mjcf/
    └── h2.xml           # MuJoCo XML
```

The **URDF** is loaded by Isaac Lab for physics simulation. The **MJCF** is used
by the motion library to compute forward kinematics on reference motion data.
Both must represent the same robot with consistent joint names and tree structure.

Make sure your URDF mesh paths are correct (relative paths like `meshes/pelvis.stl`
work best). If your URDF uses `package://` paths, update them to match the
directory layout.

## Step 2: Robot Configuration

Create `gear_sonic/envs/manager_env/robots/<robot>.py`. This is the most
important file — it defines how your robot integrates with the training pipeline.

### Joint and body ordering

Isaac Lab and MuJoCo traverse the kinematic tree in different orders. You must
define bidirectional index mappings. Get these by loading your URDF in Isaac Lab
and your MJCF in MuJoCo, printing the joint/body lists, and computing the
reorder indices.

```python
# All bodies in IsaacLab traversal order (including root "pelvis")
H2_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    # ... all 32 bodies for H2
]

# Index arrays: position i in the output = position mapping[i] in the input
H2_ISAACLAB_TO_MUJOCO_DOF = [...]   # len = num_dof (31 for H2)
H2_MUJOCO_TO_ISAACLAB_DOF = [...]
H2_ISAACLAB_TO_MUJOCO_BODY = [...]  # len = num_bodies (32 for H2)
H2_MUJOCO_TO_ISAACLAB_BODY = [...]

H2_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": H2_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": H2_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": H2_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": H2_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": H2_MUJOCO_TO_ISAACLAB_BODY,
}
```

**Getting the mappings right is critical.** If they are wrong, the policy will
receive scrambled observations and produce scrambled actions. Verify by loading a
known pose in both simulators and checking that joint values match after reordering.

### Actuator parameters (KP/KD tuning)

The actuator stiffness (KP) and damping (KD) are critical for sim-to-real
transfer and training stability. SONIC uses implicit PD actuators in Isaac Lab.

```python
# Derive from motor specs — these need tuning for your robot
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz natural frequency
DAMPING_RATIO = 2.0                      # Overdamped for stability

# Per-motor stiffness: KP = armature * omega^2
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
# Per-motor damping: KD = 2 * zeta * armature * omega
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
```

**Tuning guidance:**

- Start with the real motor's **armature** (rotor inertia) from the datasheet.
- The **natural frequency** controls responsiveness. 10 Hz is a good starting point
  for humanoids. Increase for stiffer/faster tracking, decrease for compliance.
- The **damping ratio** should be >= 1.0 (critically damped or overdamped) to avoid
  oscillation. 2.0 works well for SONIC.
- **Different joint groups need different gains.** Hip/knee motors are much stronger
  than wrist motors. Group joints by motor type (see G1/H2 configs for examples).
- If training is unstable (robot explodes or falls immediately), your KP/KD values
  are likely wrong. Try reducing KP or increasing KD.
- The **effort limits** (max torque) per joint should match the real motor specs.

### Articulation config

```python
H2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="gear_sonic/data/assets/robot_description/urdf/h2/h2.urdf",
        fix_base=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        ...
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.04),       # Standing height — must match your robot
        joint_pos={
            ".*_knee_joint": -0.363,  # Slight knee bend for stability
            # ... default standing pose for all joints
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_.*"],
            effort_limit={...},       # Max torque per joint (Nm)
            stiffness={...},          # KP values
            damping={...},            # KD values
            armature={...},           # Rotor inertia
        ),
        # ... one group per motor type (arms, waist, feet, etc.)
    },
)
```

**Important init_state notes:**

- `pos` z-value is the spawn height. Set this so the robot starts standing with
  feet slightly above ground. Too low = feet clip through ground on first frame.
- `joint_pos` should be a stable standing pose. Get this from your robot's real
  default calibration pose or a MuJoCo keyframe.

### Action scale

Action scale maps normalized policy outputs to joint position targets. Compute
from effort limit and stiffness:

```python
H2_ACTION_SCALE = {}
for joint_name in joint_names:
    H2_ACTION_SCALE[joint_name] = effort_limit[joint_name] / stiffness[joint_name]
```

Larger action scale = larger joint movements per policy output. If the robot
moves too aggressively, reduce the action scale.

### Register in __init__.py

Add your module to `gear_sonic/envs/manager_env/robots/__init__.py` so it's
importable.

### Register in modular_tracking_env_cfg.py

Add your robot to the `robot_mapping` dict (around line 998):

```python
from gear_sonic.envs.manager_env.robots import g1, h2  # Add your import

robot_mapping = {
    "g1_model_12_dex": {...},
    "h2": {
        "robot_cfg": h2.H2_CFG,
        "action_scale": h2.H2_ACTION_SCALE,
        "isaaclab_to_mujoco_mapping": h2.H2_ISAACLAB_TO_MUJOCO_MAPPING,
    },
}
```

The string key (e.g., `"h2"`) is what you'll use as `robot.type` in the
experiment config.

## Step 3: Order Converter

In `gear_sonic/trl/utils/order_converter.py`, add a converter class. This is used
by the evaluation and export pipeline:

```python
class H2Converter(IsaacLabMuJoCoConverter):
    def __init__(self):
        from gear_sonic.envs.manager_env.robots.h2 import (
            H2_ISAACLAB_JOINTS, H2_ISAACLAB_TO_MUJOCO_BODY,
            H2_ISAACLAB_TO_MUJOCO_DOF, H2_MUJOCO_TO_ISAACLAB_BODY,
            H2_MUJOCO_TO_ISAACLAB_DOF,
        )
        self.JOINT_NAMES = H2_ISAACLAB_JOINTS
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): H2_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): H2_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): H2_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): H2_MUJOCO_TO_ISAACLAB_BODY,
        }

    # Bodies used for VR tracking and foot contact — update for your robot
    VR_3POINTS_BODY_NAMES = ["torso_link", "left_wrist_pitch_link", "right_wrist_pitch_link"]
    FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
```

Use lazy imports (inside `__init__`) to avoid circular dependencies.

## Step 4: Body Name Compatibility

This is a common source of errors. The training configs reference specific body
names that must exist on your robot. Check **all** of these:

### Command config (`config/manager_env/commands/terms/motion.yaml`)

```yaml
anchor_body: "pelvis"                           # Root body
vr_3point_body: ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
reward_point_body: ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link",
                    "left_ankle_roll_link", "right_ankle_roll_link"]
body_names: [                                   # 14 tracked bodies
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
    "right_wrist_yaw_link",
]
```

### Termination configs (`config/manager_env/terminations/terms/`)

- `ee_body_pos_adaptive.yaml`: references `left_ankle_roll_link`, `right_ankle_roll_link`,
  `left_wrist_yaw_link`, `right_wrist_yaw_link`
- `foot_pos_xyz.yaml`: references `left_ankle_roll_link`, `right_ankle_roll_link`

### Reward configs (`config/manager_env/rewards/terms/`)

- `undesired_contacts.yaml`: regex pattern excluding specific bodies from contact
  penalty — references ankle and wrist link names
- `anti_shake_ang_vel.yaml`: references `left_wrist_yaw_link`, `right_wrist_yaw_link`,
  `head_link`

### What to do if names differ

If your robot uses different names for equivalent bodies (e.g., H2 has
`head_pitch_link` instead of G1's `head_link`), you have two options:

1. **Override in experiment config** (recommended): Add overrides in your
   `sonic_<robot>.yaml` for the specific fields that differ.

2. **Create robot-specific config variants**: Copy the affected term YAML files
   and create robot-specific versions (e.g., `anti_shake_ang_vel_h2.yaml`).

For H2, most G1 body names happen to exist (both are Unitree humanoids), but
`head_link` does not — H2 has `head_yaw_link` instead. Override in the
experiment config:

```yaml
manager_env:
  rewards:
    anti_shake_ang_vel:
      params:
        body_names: ["left_wrist_yaw_link", "right_wrist_yaw_link", "head_yaw_link"]
```

**Tip:** Run training with `num_envs=1` first. If a body name doesn't exist, Isaac
Lab will raise a clear error telling you which name failed. Fix it and retry.

## Step 5: Motion Data

SONIC expects retargeted motion data as PKL files (joblib format). Each file
contains a dict keyed by motion name:

```python
{
    "motion_name": {
        "root_trans_offset": np.ndarray,  # (T, 3) — root translation
        "pose_aa": np.ndarray,            # (T, num_bodies, 3) — axis-angle per body
        "dof": np.ndarray,                # (T, num_dof) — joint positions in MuJoCo order
        "root_rot": np.ndarray,           # (T, 4) — root quaternion (wxyz)
        "smpl_joints": np.ndarray,        # (T, 24, 3) — SMPL joint positions (optional)
        "fps": int,                       # Frame rate (typically 30)
    }
}
```

**Important data format notes:**

- `num_bodies` and `num_dof` must match your robot (e.g., 32 bodies / 31 DOF for H2).
- `dof` values must be in **MuJoCo joint order**, not IsaacLab order.
- `pose_aa` must be in **MuJoCo body order**.
- Mirrored variants (filename ending in `_M.pkl`) double your effective dataset
  size and improve symmetry.
- The `smpl_joints` field is used by the SMPL encoder. Set it to zeros if you
  don't have SMPL data.

The motion library loads PKL files **recursively** from a directory:

```
data/h2_motions/
├── session_01/
│   ├── walk_forward_001.pkl
│   └── walk_forward_001_M.pkl
└── session_02/
    └── ...
```

### Source motion data

The recommended source is [Bones-SEED](https://huggingface.co/datasets/bones-studio/seed)
— a large-scale human motion dataset (142K+ motions, ~288 hours) that provides:

- **Raw BVH files** — full-body human motion capture
- **G1 retargeted CSVs** — already retargeted to the Unitree G1 (29 DOF)

For a new robot, you need to **retarget** the raw human motions to your robot's
skeleton. This is the most labor-intensive step.

### Retargeting options

1. **[SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)** (recommended) —
   NVIDIA's BVH-to-humanoid motion retargeting library built with Newton and
   NVIDIA Warp. Supports any humanoid robot via JSON configuration. Includes a
   viewer for inspecting source and retargeted motions side by side. This is
   the same tool used to produce the Bones-SEED G1 retargeted data.

2. **[GMR](https://github.com/YanjieZe/GMR)** (General Motion Retargeting) —
   retargets human motions to arbitrary humanoid robots in real time on CPU.
   Supports any URDF. A lighter-weight alternative.

3. **This repo's data processing** (`gear_sonic/data_process/`) — converts
   retargeted CSVs/BVHs into the PKL format SONIC expects. Use this as the
   final step after retargeting:

   ```bash
   # Convert retargeted CSVs to motion library PKLs
   python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
       --input /path/to/retargeted_csvs/ \
       --output data/my_robot_motions/robot \
       --fps 30 --fps_source 120 --individual --num_workers 16

   # Filter out motions that are physically impossible for your robot
   python gear_sonic/data_process/filter_and_copy_bones_data.py \
       --source data/my_robot_motions/robot \
       --dest data/my_robot_motions/robot_filtered
   ```

### SMPL data (optional but recommended)

The SMPL encoder gives the policy an additional human-skeleton input signal.
You need SMPL retargeted data matching the same motion keys as your robot data.

- For Bones-SEED motions, pre-computed SMPL data is available on
  [Hugging Face](https://huggingface.co/nvidia/GEAR-SONIC):
  `python download_from_hf.py --training`
- If you use custom motions, extract SMPL joints from the BVH files:

  ```bash
  python gear_sonic/data_process/extract_soma_joints_from_bvh.py \
      --input /path/to/bvh_files/ \
      --output data/my_robot_motions/soma \
      --fps 30 --num_workers 16
  ```

- If you don't have SMPL data, set `smpl_motion_file: dummy` in the config.
  The training pipeline will generate minimal placeholder SMPL data from the
  robot motions. This works but produces weaker SMPL encoder performance.

## Step 6: Experiment Config

Create `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_<robot>.yaml`.
Start by copying `sonic_release.yaml` and modify:

```yaml
# @package _global_
defaults:
  - /algo: ppo_im_phc
  - /manager_env: base_env
  # ... same defaults as sonic_release.yaml

project_name: TRL_H2_Track                     # Change project name

manager_env:
  config:
    robot:
      type: h2                                  # Must match robot_mapping key
  commands:
    motion:
      motion_lib_cfg:
        motion_file: null                       # Provide on command line
        asset:
          assetFileName: "h2.xml"               # Your MJCF filename
```

**Fields to review and potentially override:**

- `robot.type` — must match the key in `robot_mapping`
- `motion_lib_cfg.asset.assetFileName` — your MJCF file
- `reward_point_body` / `reward_point_body_offset` — key bodies for reward computation
- `vr_3point_body` / `vr_3point_body_offset` — if doing VR teleoperation
- `upper_body_augment_prefixes` — remove if your motion data uses different naming
- Body names in reward/termination overrides — see Step 4

## Step 7: Train

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_h2 \
    num_envs=16 headless=False \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/h2_motions>
```

Start with `num_envs=16 headless=False` to visually verify the robot loads and
motions play correctly. Then scale up to `num_envs=4096 headless=True` for
full training.

## Example: H2 (included)

The codebase includes full H2 support as a reference:

| Component | File |
|-----------|------|
| Robot config | `gear_sonic/envs/manager_env/robots/h2.py` |
| URDF + meshes | `gear_sonic/data/assets/robot_description/urdf/h2/` |
| MJCF | `gear_sonic/data/assets/robot_description/mjcf/h2.xml` |
| Experiment config | `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml` |
| Order converter | `gear_sonic/trl/utils/order_converter.py` (`H2Converter`) |
| Robot mapping | `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` |

## Checklist

When adding a new robot, verify each of these:

- [ ] URDF + meshes in `gear_sonic/data/assets/robot_description/urdf/<robot>/`
- [ ] MJCF in `gear_sonic/data/assets/robot_description/mjcf/<robot>.xml`
- [ ] Robot config in `gear_sonic/envs/manager_env/robots/<robot>.py`:
  - [ ] Joint/body name lists
  - [ ] IsaacLab ↔ MuJoCo index mappings (verified correct!)
  - [ ] `ArticulationCfg` with tuned KP/KD/effort for each motor group
  - [ ] Correct init_state (standing height + default joint angles)
  - [ ] Action scale dict
- [ ] Robot imported in `robots/__init__.py`
- [ ] Robot added to `robot_mapping` in `modular_tracking_env_cfg.py`
- [ ] Order converter class in `order_converter.py`
- [ ] Experiment config YAML with correct `robot.type` and `assetFileName`
- [ ] All body names in config YAMLs exist on your robot (check with `num_envs=1`)
- [ ] Human motion source data (e.g., [Bones-SEED](https://huggingface.co/datasets/bones-studio/seed) BVH/CSV files)
- [ ] Motions retargeted to your robot's skeleton (e.g., via [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter))
- [ ] Retargeted data converted to PKL format (MuJoCo joint/body order)
- [ ] Motions filtered for physical feasibility (`filter_and_copy_bones_data.py`)
- [ ] Mirrored motion variants (`_M.pkl`) for symmetric training
- [ ] SMPL data matching the same motion keys (or `smpl_motion_file: dummy`)
