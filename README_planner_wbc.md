# Planner-to-GR00T WBC Integration

## Overview

This feature connects a precomputed upper-body joint trajectory to NVIDIA GR00T Whole-Body Control without using the teleoperation or IK pipeline.

The original `decoupled_wbc` package remains unchanged. All integration work is contained in `planner_wbc`.

## Data Flow

```text
NPZ joint trajectory
    ↓
run_planner_policy_loop.py
    ↓  ControlPolicy/upper_body_pose
run_g1_control_loop.py
    ↓
GR00T WBC
    ├── upper body: planner joint targets
    └── lower body: RL locomotion policy
    ↓
MuJoCo or robot
```

## Main Files

```text
planner_wbc/control/main/teleop/configs/configs.py
planner_wbc/control/main/teleop/run_planner_policy_loop.py
planner_wbc/control/dataset/retar_dualarm_18.npz
```

`run_g1_control_loop.py` and the WBC policy were not functionally changed.

## Trajectory Format

The current dataset contains:

```text
qpos:        (600, 17)
joint_names: 3 waist joints + 14 arm joints
```

The publisher maps joints by name instead of using fixed column positions.

With `enable_waist=False`:

- the three waist joints are ignored;
- the fourteen arm joints are published;
- hand joints remain at the robot model's default pose.

All published values are absolute joint positions in radians.

## Published Message

The planner publishes to:

```text
ControlPolicy/upper_body_pose
```

```python
{
    "target_upper_body_pose": target_upper_body_pose,
    "timestamp": current_time,
    "target_time": target_time,
}
```

No end-effector transforms, IK data, fake navigation commands, or fake hand commands are published.

## Timing

```text
Planner rate:             20 Hz
Initial transition time:  2 seconds
Trajectory looping:       disabled
```

The existing WBC control path handles interpolation. After reaching the final frame, the planner holds the final pose.

## Running in Simulation

Enter Docker:

```bash
cd ~/GR00T-WholeBodyControl/decoupled_wbc
./docker/run_docker.sh --root
```

Terminal 1:

```bash
cd ~/Projects/GR00T-WholeBodyControl
source /opt/ros/humble/setup.bash
python planner_wbc/control/main/teleop/run_g1_control_loop.py
```

Terminal 2:

```bash
cd ~/Projects/GR00T-WholeBodyControl
source /opt/ros/humble/setup.bash
python planner_wbc/control/main/teleop/run_planner_policy_loop.py
```

## Current Scope

Implemented:

- trajectory loading and validation;
- joint-name-based mapping;
- arm trajectory publishing;
- default hand-pose preservation;
- safe initial transition;
- final-pose holding;
- MuJoCo visualization through the existing control loop.

Not yet implemented:

- live motion-planner input;
- real-robot safety validation;
- velocity and acceleration limiting;
- trajectory resampling;
- forward/reverse ping-pong playback.
