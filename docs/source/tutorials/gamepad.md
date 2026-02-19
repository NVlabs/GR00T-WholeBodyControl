# Motion Tracking and Kinematic Planner with Gamepad Controls

Control the robot using a Unitree wireless gamepad for reference motion playback and planner-based locomotion (using `--input-type gamepad`).

```{admonition} Prerequisites
:class: note
Complete the [Installation Guide](../getting_started/installation_deploy) and build the project before proceeding.
```

```{warning}
The gamepad interface requires a physical Unitree wireless gamepad connected to the robot. It is **not available in sim2sim** — use the [keyboard interface](keyboard.md) to test in simulation first.
```

```{admonition} Emergency Stop
:class: danger
Press **Select** at any time to immediately stop control and exit. Always keep a hand ready to press **Select**.
```

## Launch

```bash
# From gear_sonic_deploy/
bash deploy.sh real --input-type gamepad
```

## Step-by-Step: Normal Mode (Reference Motion Tracking)

Normal Mode plays back pre-loaded reference motions. This is the default mode when the program starts.

1. Press **Start** to start the control system.
2. Press **A** to play the current reference motion — the robot executes it to completion.
4. Press **R1** to switch to the next motion sequence, or **L1** for the previous one.
5. Press **A** again to play the new motion.
6. To stop mid-motion and return to the first frame, press **B** — the robot pauses without terminating the policy.
7. Use **D-pad Left / Right** to nudge the heading (±0.1 rad per press).
8. Press **X** or **Y** to reinitialize the base quaternion and reset the heading to zero, i.e. robot will think the current facing is the facing at the first frame of the reference.
9. When done, press **Select** to stop control and exit.

## Step-by-Step: Planner Mode (Real-time Motion Generation)

Planner Mode gives you analog stick control — steer with the left stick, aim the facing direction with the right stick, and cycle through locomotion modes.

1. From Normal Mode, press **F1** to switch to Planner Mode. The terminal will print `Planner enabled`.
2. The robot starts in Slow Walk mode (mode 1). Push the **left stick forward** to walk — movement direction is computed from the stick angle relative to the current facing direction.
3. Steer the **right stick left / right** to smoothly rotate the facing direction (continuous, ±0.02 rad per frame).
4. Press **R1** to cycle to the next movement mode (Idle → Slow Walk → Walk → Run → Squat → Kneel Two Legs → Kneel → Idle → …). Press **L1** to cycle backward.
5. Hold **R2** to increase speed (for standing modes 1–3) or height (for squat modes 4–6). Hold **L2** to decrease. Speed/height changes by ±0.02 per frame while held.
6. When the left stick is in the dead zone, standing modes automatically switch to Idle; squat/kneel modes hold their pose with speed 0.
7. To pause, press **B** — the robot resets to Idle immediately. 
8. Press **F1** again to return to Normal Mode, or **Select** to stop and exit.

## Control Reference

### System Controls (Both Modes)

| Button | Action |
|--------|--------|
| **Start** | Start control system |
| **Select** | Stop control and exit (emergency stop) |
| **F1** | Toggle between Normal / Planner modes |
| **X** or **Y** | Reinitialize base quaternion and reset heading |
| **D-pad Left / Right** | Adjust delta heading (±0.1 rad) |

### Normal Mode Buttons

| Button | Action |
|--------|--------|
| **A** | Play current motion to completion |
| **B** | Restart current motion from beginning (pause at frame 0) |
| **L1** / **R1** | Previous / Next motion sequence |

### Planner Mode Buttons

**Movement:**

| Input | Action |
|-------|--------|
| **Left Stick** | Movement direction (computed from stick angle + facing direction) |
| **Right Stick** | Facing direction (continuous rotation, ±0.02 rad/frame) |

**Mode & Speed:**

| Button | Action |
|--------|--------|
| **L1** / **R1** | Previous / Next movement mode (cycles 0–6) |
| **L2** (hold) | Decrease speed (modes 1–3) or height (modes 4–6), ±0.02/frame |
| **R2** (hold) | Increase speed (modes 1–3) or height (modes 4–6), ±0.02/frame |
| **A** | Play / resume motion |

**Emergency:**

| Button | Action |
|--------|--------|
| **B** | Pause (reset to Idle) |
| **Select** | Stop control and exit |

## Movement Modes

The gamepad cycles through 7 modes with **L1** / **R1**. Use **L2** / **R2** (hold) to adjust speed or height.

| ID | Mode | L2/R2 Adjusts | Range |
|----|------|---------------|-------|
| 0 | Idle | — | — |
| 1 | Slow Walk | speed | 0.2–0.8 m/s |
| 2 | Walk | speed | 0.8–1.5 m/s |
| 3 | Run | speed | 1.5–3.0 m/s |
| 4 | Squat | height | 0.1–0.8 m |
| 5 | Kneel (two legs) |  — |  — |
| 6 | Kneel |  — |  — |

```{tip}
For lateral (side-stepping) movement, we recommend keeping the target velocity at around **0.4 m/s**. Higher velocities during strafing can cause the robot's feet to collide due to the cross-legged foot placement required for lateral steps.
```

```{note}
When entering Squat (mode 4) from an adjacent mode, the height automatically initializes to 0.8 m.
```
