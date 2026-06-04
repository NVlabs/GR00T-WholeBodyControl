# XR Controller GEAR-SONIC Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Quest 3 / Pico 4 Ultra controller controls for grippers, pause/resume, quit, locomotion, squat, and kneel in the XR-to-GEAR-SONIC export flow.

**Architecture:** Keep controller intent generation in Unitree `xr_teleoperate`, because it owns Vuer controller data. Extend the JSON export payload with planner fields and control toggles, then extend the GEAR-SONIC bridge to forward those fields into existing planner and manager-state messages.

**Tech Stack:** Python 3.10, Vuer/OpenXR controller data, pyzmq JSON export, GEAR-SONIC ZMQ planner messages, pytest.

---

### Task 1: Add Pure Controller Intent Helpers

**Files:**
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/teleop_hand_and_arm.py`

- [ ] **Step 1: Add constants and helper functions near existing debug helpers**

Add:

```python
XR_CONTROL_MODES = [0, 1, 2, 3, 4, 5, 6]
XR_STANCE_MODES = {4, 5, 6}
XR_WALK_MODES = {1, 2, 3}
XR_STICK_DEADZONE = 0.15
XR_TURN_RATE_RAD_S = 1.2
XR_HEIGHT_RATE_M_S = 0.25
XR_MIN_HEIGHT = 0.2
XR_MAX_HEIGHT = 0.8
XR_DEFAULT_HEIGHT = 0.8
XR_DEFAULT_SPEED_BY_MODE = {
    0: 0.0,
    1: 0.4,
    2: 1.0,
    3: 1.8,
    4: 0.0,
    5: 0.0,
    6: 0.0,
}
XR_DEX3_OPEN = [0.0] * 7
XR_DEX3_CLOSED = [0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]


def _stick_with_deadzone(stick, deadzone=XR_STICK_DEADZONE):
    vec = np.asarray(stick, dtype=float).reshape(2)
    if np.linalg.norm(vec) < deadzone:
        return np.zeros(2, dtype=float)
    return np.clip(vec, -1.0, 1.0)


def _button_combo(*values):
    return all(bool(value) for value in values)


def _edge(current, previous):
    return bool(current) and not bool(previous)


def _controller_gripper_targets(tele_data, trigger_threshold=0.35):
    left_closed = bool(tele_data.left_ctrl_trigger) or tele_data.left_ctrl_triggerValue <= (10.0 - trigger_threshold * 10.0)
    right_closed = bool(tele_data.right_ctrl_trigger) or tele_data.right_ctrl_triggerValue <= (10.0 - trigger_threshold * 10.0)
    left = XR_DEX3_CLOSED if left_closed else XR_DEX3_OPEN
    right = XR_DEX3_CLOSED if right_closed else XR_DEX3_OPEN
    return list(left) + list(right)
```

- [ ] **Step 2: Syntax check**

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPYCACHEPREFIX=/tmp/xr_teleoperate_pycache PYTHONDONTWRITEBYTECODE=1 \
  /home/jihun/anaconda3/envs/tv/bin/python -B -m py_compile teleop/teleop_hand_and_arm.py
```

Expected: exits 0.

### Task 2: Add Controller Runtime State

**Files:**
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/teleop_hand_and_arm.py`

- [ ] **Step 1: Initialize controller export state after `last_gear_sonic_debug_time`**

Add:

```python
    xr_publish_paused = False
    xr_selected_mode = 1
    xr_facing_yaw = 0.0
    xr_height = XR_DEFAULT_HEIGHT
    xr_prev_combos = {
        "pause_ax": False,
        "pause_by": False,
        "quit": False,
        "next_mode": False,
        "prev_mode": False,
    }
```

- [ ] **Step 2: Add a small mode-cycling helper**

Add near helper functions:

```python
def _cycle_mode(current_mode, direction):
    modes = XR_CONTROL_MODES
    index = modes.index(int(current_mode)) if int(current_mode) in modes else 0
    return modes[(index + int(direction)) % len(modes)]
```

- [ ] **Step 3: Syntax check**

Run the same `py_compile` command from Task 1.

Expected: exits 0.

### Task 3: Generate Planner Intent From Controller Sticks And Buttons

**Files:**
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/teleop_hand_and_arm.py`

- [ ] **Step 1: Add planner helper**

Add:

```python
def _controller_planner_intent(tele_data, selected_mode, facing_yaw, height, dt):
    left_stick = _stick_with_deadzone(tele_data.left_ctrl_thumbstickValue)
    right_stick = _stick_with_deadzone(tele_data.right_ctrl_thumbstickValue)

    facing_yaw += float(right_stick[0]) * XR_TURN_RATE_RAD_S * max(float(dt), 0.0)
    facing = [float(np.cos(facing_yaw)), float(np.sin(facing_yaw)), 0.0]

    mode = int(selected_mode)
    speed = XR_DEFAULT_SPEED_BY_MODE.get(mode, -1.0)
    movement = [0.0, 0.0, 0.0]
    out_height = -1.0

    if mode in XR_STANCE_MODES:
        height = float(np.clip(height + float(right_stick[1]) * XR_HEIGHT_RATE_M_S * max(float(dt), 0.0), XR_MIN_HEIGHT, XR_MAX_HEIGHT))
        out_height = height
        speed = 0.0
    elif mode in XR_WALK_MODES:
        if np.linalg.norm(left_stick) < XR_STICK_DEADZONE:
            mode = 0
            speed = 0.0
        else:
            movement = [-float(left_stick[1]), -float(left_stick[0]), 0.0]
    else:
        mode = 0
        speed = 0.0

    return {
        "mode": mode,
        "movement": movement,
        "facing": facing,
        "speed": speed,
        "height": out_height,
        "selected_mode": int(selected_mode),
        "facing_yaw": facing_yaw,
        "stored_height": height,
    }
```

- [ ] **Step 2: In the main loop, compute button combos for controller mode**

After `tele_data = tv_wrapper.get_tele_data()`, add:

```python
            xr_control_toggle_data_collection = False
            if args.gear_sonic_export and args.input_mode == "controller":
                combo_pause_ax = _button_combo(tele_data.right_ctrl_aButton, tele_data.left_ctrl_aButton)
                combo_pause_by = _button_combo(tele_data.right_ctrl_bButton, tele_data.left_ctrl_bButton)
                combo_quit = _button_combo(
                    tele_data.right_ctrl_aButton,
                    tele_data.right_ctrl_bButton,
                    tele_data.left_ctrl_aButton,
                    tele_data.left_ctrl_bButton,
                )
                combo_next_mode = _button_combo(tele_data.right_ctrl_aButton, tele_data.right_ctrl_bButton)
                combo_prev_mode = _button_combo(tele_data.left_ctrl_aButton, tele_data.left_ctrl_bButton)

                if _edge(combo_quit, xr_prev_combos["quit"]):
                    STOP = True
                if _edge(combo_pause_ax, xr_prev_combos["pause_ax"]) or _edge(combo_pause_by, xr_prev_combos["pause_by"]):
                    xr_publish_paused = not xr_publish_paused
                    xr_control_toggle_data_collection = True
                    logger_mp.info(f"[XR CONTROL] publish_paused={xr_publish_paused}")
                if _edge(combo_next_mode, xr_prev_combos["next_mode"]):
                    xr_selected_mode = _cycle_mode(xr_selected_mode, +1)
                    if xr_selected_mode in XR_STANCE_MODES:
                        xr_height = XR_DEFAULT_HEIGHT
                    logger_mp.info(f"[XR CONTROL] selected_mode={xr_selected_mode}")
                if _edge(combo_prev_mode, xr_prev_combos["prev_mode"]):
                    xr_selected_mode = _cycle_mode(xr_selected_mode, -1)
                    if xr_selected_mode in XR_STANCE_MODES:
                        xr_height = XR_DEFAULT_HEIGHT
                    logger_mp.info(f"[XR CONTROL] selected_mode={xr_selected_mode}")

                xr_prev_combos.update(
                    {
                        "pause_ax": combo_pause_ax,
                        "pause_by": combo_pause_by,
                        "quit": combo_quit,
                        "next_mode": combo_next_mode,
                        "prev_mode": combo_prev_mode,
                    }
                )
```

- [ ] **Step 3: Syntax check**

Run the same `py_compile` command from Task 1.

Expected: exits 0.

### Task 4: Publish Controller Gripper And Planner Fields

**Files:**
- Modify: `/home/jihun/work/unitree_official/xr_teleoperate/teleop/teleop_hand_and_arm.py`

- [ ] **Step 1: In export mode, build `dual_hand_payload` from triggers in controller mode**

Replace:

```python
                dual_hand_payload = []
                if args.ee == "dex3":
                    with dual_hand_data_lock:
                        dual_hand_payload = list(dual_hand_action_array[:])
```

with:

```python
                dual_hand_payload = []
                if args.ee == "dex3" and args.input_mode == "controller":
                    dual_hand_payload = _controller_gripper_targets(tele_data)
                elif args.ee == "dex3":
                    with dual_hand_data_lock:
                        dual_hand_payload = list(dual_hand_action_array[:])
```

- [ ] **Step 2: Add pause handling before JSON send**

Before `payload = { "timestamp": ... }`, add:

```python
                planner_intent = None
                if args.input_mode == "controller":
                    dt_control = max(time.time() - start_time, 1.0 / args.frequency)
                    planner_intent = _controller_planner_intent(
                        tele_data, xr_selected_mode, xr_facing_yaw, xr_height, dt_control
                    )
                    xr_facing_yaw = planner_intent["facing_yaw"]
                    xr_height = planner_intent["stored_height"]

                if xr_publish_paused:
                    if args.gear_sonic_debug and now_debug - last_gear_sonic_debug_time >= args.gear_sonic_debug_interval:
                        logger_mp.info("[XR CONTROL] publishing paused")
                        last_gear_sonic_debug_time = now_debug
                    time.sleep(1 / args.frequency)
                    continue
```

- [ ] **Step 3: Add planner and control fields to payload**

After creating `payload`, add:

```python
                if planner_intent is not None:
                    payload.update(
                        {
                            "mode": planner_intent["mode"],
                            "movement": planner_intent["movement"],
                            "facing": planner_intent["facing"],
                            "speed": planner_intent["speed"],
                            "height": planner_intent["height"],
                            "selected_mode": planner_intent["selected_mode"],
                            "toggle_data_collection": xr_control_toggle_data_collection,
                            "stop": bool(STOP),
                        }
                    )
```

- [ ] **Step 4: Extend debug log**

Append to the existing `[GEAR-SONIC DEBUG]` log:

```python
                            f"planner={planner_intent} "
                            f"paused={xr_publish_paused} "
```

- [ ] **Step 5: Syntax check**

Run the same `py_compile` command from Task 1.

Expected: exits 0.

### Task 5: Forward Recording Toggles And Stop In The Bridge

**Files:**
- Modify: `gear_sonic/utils/teleop/xr_upperbody_bridge.py`
- Modify: `gear_sonic/tests/test_xr_upperbody_bridge.py`

- [ ] **Step 1: Extend `UpperBodyFrame` fields**

Add fields:

```python
    toggle_data_collection: bool = False
    toggle_data_abort: bool = False
    stop: bool = False
```

- [ ] **Step 2: Extend `frame_from_mapping`**

Add constructor arguments:

```python
        toggle_data_collection=bool(data.get("toggle_data_collection", False)),
        toggle_data_abort=bool(data.get("toggle_data_abort", False)),
        stop=bool(data.get("stop", False)),
```

- [ ] **Step 3: Forward manager-state toggles in `_send_zmq_json_loop`**

Replace:

```python
            pub.send(build_manager_state_message(stream_mode=args.stream_mode))
```

with:

```python
            pub.send(
                build_manager_state_message(
                    stream_mode=args.stream_mode,
                    toggle_data_collection=frame.toggle_data_collection,
                    toggle_data_abort=frame.toggle_data_abort,
                )
            )
            if frame.stop:
                pub.send(build_command_message(start=False, stop=True, planner=True))
```

- [ ] **Step 4: Add test for manager-state toggle forwarding**

Add a test that builds a frame from:

```python
{
    "dual_arm_position": [0.0] * 14,
    "dual_hand_joints": [0.0] * 14,
    "toggle_data_collection": True,
    "stop": True,
}
```

Assert:

```python
assert frame.toggle_data_collection is True
assert frame.stop is True
```

- [ ] **Step 5: Run tests**

Run:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-upperbody-bridge
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  pytest gear_sonic/tests/test_xr_upperbody_bridge.py -q
```

Expected: all tests pass.

### Task 6: Documentation And Final Verification

**Files:**
- Modify: `docs/source/tutorials/xr_upperbody_bridge.md`

- [ ] **Step 1: Add controller-mode command example**

Document:

```bash
python teleop/teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex3 \
  --input-mode controller \
  --sim \
  --img-server-ip 127.0.0.1 \
  --gear-sonic-export \
  --gear-sonic-export-port 5560 \
  --gear-sonic-export-topic xr_teleop \
  --gear-sonic-debug
```

- [ ] **Step 2: Add controller mapping table**

Document the confirmed mapping:

| Input | Action |
|---|---|
| Left trigger | Left Dex3 gripper close/open |
| Right trigger | Right Dex3 gripper close/open |
| Left stick | Walk / strafe |
| Right stick X | Turn / yaw |
| Right stick Y | Squat/kneel height |
| A + X | Pause/resume |
| B + Y | Pause/resume |
| A + B | Next locomotion mode |
| X + Y | Previous locomotion mode |
| A + B + X + Y | Quit/stop |

- [ ] **Step 3: Run final syntax checks**

Run:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
PYTHONPYCACHEPREFIX=/tmp/xr_teleoperate_pycache PYTHONDONTWRITEBYTECODE=1 \
  /home/jihun/anaconda3/envs/tv/bin/python -B -m py_compile \
  teleop/teleop_hand_and_arm.py \
  teleop/televuer/src/televuer/televuer.py \
  teleop/robot_control/robot_hand_unitree.py \
  teleop/robot_control/robot_arm_ik.py \
  teleop/robot_control/hand_retargeting.py
```

Expected: exits 0.

Run:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-upperbody-bridge
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  pytest gear_sonic/tests/test_xr_upperbody_bridge.py -q
```

Expected: all tests pass.

---

## Self-Review

Spec coverage:

- Controller grippers: Task 1 and Task 4.
- Pause/resume and quit: Task 3 and Task 4.
- Walking/turning: Task 3 and Task 4.
- Squat/kneel height: Task 3 and Task 4.
- Bridge forwarding: Task 5.
- Documentation and validation: Task 6.

Placeholder scan: no TBD/TODO placeholders remain.

Type consistency: JSON fields match `UpperBodyFrame` mapping and `build_planner_message` fields.
