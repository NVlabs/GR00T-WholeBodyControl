# Planner WBC Commands

Run host commands outside Docker. Run robot programs inside the Docker container.

## Start Docker

Run on the host:

```bash
cd ~/GR00T-WholeBodyControl/planner_wbc
./docker/run_docker.sh --root
```

Inside Docker, prepare each terminal:

```bash
cd ~/Projects/GR00T-WholeBodyControl
```

## Simulation test

Terminal 1, inside Docker:

```bash
cd ~/Projects/GR00T-WholeBodyControl
python planner_wbc/control/main/teleop/run_g1_control_loop.py
```

Press `]` in Terminal 1 to activate the lower-body policy.

Terminal 2, inside Docker:

```bash
cd ~/Projects/GR00T-WholeBodyControl
python planner_wbc/control/main/teleop/run_planner_policy_loop.py \
  --trajectory-path control/dataset/pickup.npz \
  --loop-trajectory
```

## Real-robot deployment

Before starting, support the robot, clear the area, and keep the physical emergency stop ready. Do not run another robot controller at the same time.

### 1. Check the robot network

Run on the host:

```bash
ip -br addr | grep 192.168.123
```

The computer should have an address such as `192.168.123.222/24` on the robot-facing interface.

### 2. Start the real-robot control loop

Terminal 1, inside Docker:

```bash
cd ~/Projects/GR00T-WholeBodyControl
python planner_wbc/control/main/teleop/run_g1_control_loop.py --interface real
```

Wait until both ONNX policies load and robot state is being received. Then press `]` in Terminal 1 to activate the lower-body policy.

### 3. First test: play the trajectory once

Terminal 2, inside Docker:

```bash
cd ~/Projects/GR00T-WholeBodyControl
python planner_wbc/control/main/teleop/run_planner_policy_loop.py \
  --trajectory-path control/dataset/pickup.npz
```

### 4. Continuous trajectory test

After the one-way test succeeds:

```bash
python planner_wbc/control/main/teleop/run_planner_policy_loop.py \
  --trajectory-path control/dataset/pickup.npz \
  --loop-trajectory
```

`--loop-trajectory` plays forward and backward continuously.

## Available trajectories

Use one of these values with `--trajectory-path`:

```text
control/dataset/pickup.npz
control/dataset/retar_dualarm_18.npz
control/dataset/retar_handover38.npz
```

Example:

```bash
python planner_wbc/control/main/teleop/run_planner_policy_loop.py \
  --trajectory-path control/dataset/retar_dualarm_18.npz \
  --loop-trajectory
```

## Stop safely

1. In the planner terminal, press `Ctrl+C` to stop publishing the trajectory.
2. Support the robot.
3. In the control-loop terminal, press `o` to deactivate the lower-body policy.
4. Press `Ctrl+C` to stop the control loop.

## Useful control-loop keys

```text
]   Activate lower-body policy
o   Deactivate lower-body policy
z   Set navigation command to zero
`   Emergency-stop/exit handler
```

The planner publisher has no keyboard shortcuts; stop it with `Ctrl+C`.
