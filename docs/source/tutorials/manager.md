# Interface Manager (All-In-One)

Dynamically switch between keyboard, gamepad, and ZMQ input interfaces at runtime using hotkeys (`--input-type manager`). The manager owns all interfaces simultaneously and delegates to the currently active one. You can switch at any time without restarting the program.

```{admonition} Prerequisites
:class: note
Complete the [Quick Start](../getting_started/quickstart.md) to have the sim2sim loop running.
```

```{admonition} Emergency Stop
:class: danger
Press **`O`** at any time to immediately stop control and exit — this works regardless of which interface is active, including when the gamepad is selected. Always keep a hand near the keyboard ready to press **`O`**.
```

## Launch

**Sim2Sim (MuJoCo):**

```bash
# Terminal 1 — MuJoCo simulator (from repo root)
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 — C++ deployment (from gear_sonic_deploy/)
bash deploy.sh sim --input-type manager
```

**Real Robot:**

```bash
# From gear_sonic_deploy/
bash deploy.sh real --input-type manager
```

If you plan to use the ZMQ interface, add ZMQ flags:

```bash
bash deploy.sh sim --input-type manager \
  --zmq-host <publisher-ip> \
  --zmq-port 5556 \
  --zmq-topic pose
```

## Step-by-Step

1. The manager starts in **Keyboard** mode by default.
2. Press **`]`** to start the control system (keyboard mode). Use the robot as described in the [Keyboard tutorial](keyboard.md).
3. To switch to gamepad, press **`Shift+2`** (types `@`). The terminal prints `Switched to: GAMEPAD (safety reset triggered)`. The robot returns to reference motion at frame 0 and the planner is disabled.
4. Use the gamepad controls as described in the [Gamepad tutorial](gamepad.md).
5. To switch to ZMQ streaming, press **`Shift+3`** (types `#`). A safety reset is triggered and the terminal prints `Switched to: ZMQ`. Use the ZMQ controls as described in the [ZMQ tutorial](zmq.md).
6. To switch back to keyboard at any time, press **`Shift+1`** (types `!`).
7. Press **`O`** to stop control and exit from any interface.

## Switching Interfaces

| Hotkey | Interface | Notes |
|--------|-----------|-------|
| **Shift+1** (`!`) | Keyboard | Default. Full keyboard controls (Normal + Planner modes). See [Keyboard tutorial](keyboard.md). |
| **Shift+2** (`@`) | Gamepad | Unitree wireless gamepad. See [Gamepad tutorial](gamepad.md). |
| **Shift+3** (`#`) | ZMQ | ZMQ streaming (requires `--zmq-host` etc.). See [ZMQ tutorial](zmq.md). |

```{tip}
A **ROS2** interface is also available via **Shift+4** (`$`) when built with ROS2 support. It requires the planner to be loaded and falls back to Keyboard otherwise. The ROS2 interface is provided as a reference implementation for building custom ROS2-based control pipelines and may not receive the same level of updates as the other interfaces.
```

### What Happens When You Switch

Each switch triggers a **safety reset** on **all** managed interfaces (not just the one you're switching to). This:

- Disables the planner and returns to reference motion at frame 0
- Resets heading and movement states
- Disables ZMQ streaming (if it was active)
- Prevents control discontinuities from stale state in the previous interface

The safety reset ensures you always start from a clean state after switching, regardless of what the previous interface was doing.

```{note}
All four interfaces are created at startup and stay alive across switches. Their internal state (e.g., gamepad mode, ZMQ connection) is preserved — only the planner/motion state is reset for safety. This means you don't need to re-establish the ZMQ connection or re-pair the gamepad when switching back.
```

## Global Controls

These controls work at the manager level, **regardless of which interface is active**. They are intercepted by the manager before being passed to the active interface.

### Emergency Stop

| Key | Action |
|-----|--------|
| **O** / **o** | Immediate emergency stop — works even when gamepad or ZMQ is the active interface |

### Compliance Controls

These adjust hand compliance and grasp parameters globally. They are especially useful during teleoperation (ZMQ / ROS2) where the robot's hands interact with objects.

| Key | Action |
|-----|--------|
| **G** / **H** | Increase / Decrease left hand compliance by 0.1 |
| **B** / **V** | Increase / Decrease right hand compliance by 0.1 |
| **X** / **C** | Increase / Decrease max hand close ratio by 0.1 |

```{note}
Compliance controls are global — they affect the robot's hands no matter which interface is active. This lets you adjust compliance from the keyboard even while the gamepad or ZMQ is controlling the robot's movement.
```

## Interface-Specific Controls

Once an interface is active, all its normal controls work as documented in its own tutorial:

- **Keyboard** (`!`): All keys from the [Keyboard tutorial](keyboard.md) — T/R/P/N for motions, ENTER for planner, WASD for movement, etc.
- **Gamepad** (`@`): All buttons from the [Gamepad tutorial](gamepad.md) — Start, A/B, L1/R1, analog sticks, etc.
- **ZMQ** (`#`): ENTER to toggle streaming, T/P/N/R for reference motions when not streaming. See the [ZMQ tutorial](zmq.md).

The only exceptions are:
- **`O`** is always intercepted by the manager for emergency stop (not passed to the active interface).
- **`G/H/B/V/X/C`** are always intercepted for compliance control.
- **`!/@ /#/$`** are always intercepted for interface switching.

