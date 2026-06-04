# 2026-05-29 XR Controller Export To GEAR-SONIC

## Context

Goal: use Unitree `xr_teleoperate` as an upper-body/controller source for Quest 3 or Pico 4 Ultra while GEAR-SONIC remains the only robot body command writer. This avoids the official whole-body teleop ankle-sensor calibration path and keeps lower-body balance/posture inside GEAR-SONIC.

## Architecture Decisions

- `xr_teleoperate` runs with `--gear-sonic-export`.
- In export mode, `xr_teleoperate` computes arm IK but skips direct Unitree arm command writes.
- Dex3 direct hand DDS writes are disabled in export mode.
- `xr_teleoperate` publishes JSON on ZMQ topic `xr_teleop` at port `5560`.
- `gear_sonic/scripts/xr_upperbody_bridge.py` subscribes to that JSON and republishes GEAR-SONIC `planner` and `manager_state` ZMQ messages.
- GEAR-SONIC remains responsible for lower-body balance, safe posture, locomotion, and the final body command stream.

## Implemented Features

GEAR-SONIC worktree:

- `gear_sonic/utils/teleop/xr_upperbody_bridge.py`
  - Accepts live `zmq-json` payloads.
  - Maps Unitree G1_29 `dual_arm_position[14]` into SONIC upper-body `[17]`.
  - Splits Unitree Dex3 `dual_hand_joints[14]` into left/right `[7]`.
  - Passes planner fields: `mode`, `movement`, `facing`, `speed`, `height`.
  - Preserves `toggle_data_collection` and `toggle_data_abort` into `manager_state`.
  - Treats payload `stop` as source-lifecycle metadata only; it must not stop GEAR-SONIC.
- `gear_sonic/scripts/xr_upperbody_bridge.py`
- `gear_sonic/tests/test_xr_upperbody_bridge.py`
- `docs/source/tutorials/xr_upperbody_bridge.md`

Unitree `xr_teleoperate` checkout:

- Added `--gear-sonic-export`, `--gear-sonic-export-port`, and `--gear-sonic-export-topic`.
- Added `teleop/logging_mp.py` compatibility shim for missing `logging_mp`.
- Made IK and hand-retargeting asset paths cwd-independent.
- Added Pico/Vuer `HAND_MOVE` diagnostics and `msgpack.ExtType` handling for hand payloads.
- Added controller-mode export features:
  - controller pose drives arm IK;
  - left/right index triggers control left/right Dex3 open/close presets;
  - left stick controls movement;
  - right stick X controls yaw/facing;
  - right stick Y adjusts squat/kneel height;
  - `A+B` cycles locomotion mode forward;
  - `X+Y` cycles locomotion mode backward;
  - `A+X` or `B+Y` toggles pause/resume and data-collection state;
  - `A+B+X+Y` publishes one final idle planner command, exits XR teleop, and does not stop GEAR-SONIC.

## Controller Mapping

| Input | Action |
|---|---|
| Left index trigger | Left Dex3 gripper open/close |
| Right index trigger | Right Dex3 gripper open/close |
| Left stick | Walk / strafe |
| Right stick X | Turn / yaw |
| Right stick Y | Squat/kneel height |
| A + X | Pause/resume export and toggle collection state |
| B + Y | Pause/resume export and toggle collection state |
| A + B | Next locomotion mode |
| X + Y | Previous locomotion mode |
| A + B + X + Y | Final idle command, quit XR export, keep GEAR-SONIC running |

Planner modes:

| Mode | Meaning |
|---:|---|
| 0 | Idle |
| 1 | Slow Walk |
| 2 | Walk |
| 3 | Run |
| 4 | Squat |
| 5 | Kneel two legs |
| 6 | Kneel |

Squat/kneel modes publish `height` clamped to `0.2..0.8`. Walking modes publish `height=-1`.

## Commands

Start the bridge:

```bash
cd /home/jihun/work/GR00T-WholeBodyControl/worktrees/xr-upperbody-bridge
PYTHONPATH=. python gear_sonic/scripts/xr_upperbody_bridge.py \
  --source zmq-json \
  --source-host localhost \
  --source-port 5560 \
  --source-topic xr_teleop \
  --hz 50 \
  --start-control \
  --debug-live
```

Start Unitree XR export in controller mode:

```bash
cd /home/jihun/work/unitree_official/xr_teleoperate
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

Open headset browser:

```text
https://<host-pc-lan-ip>:8012/?ws=wss://<host-pc-lan-ip>:8012
```

If the local page fails, use:

```text
https://vuer.ai?ws=wss://<host-pc-lan-ip>:8012
```

## Debugging Notes

- Pico page load is confirmed by `GET /assets/... 200` and `websocket is connected`.
- If `xr_upperbody_bridge` prints only `waiting for live XR JSON source...`, either `xr_teleoperate` is not publishing or export is paused/skipping invalid XR frames.
- Pico hand tracking with controllers can produce `HAND_MOVE` events where `left/right` are missing/undefined. Controller mode is preferred for this data-collection path.
- The previous unsafe behavior where a source `stop` flag stopped GEAR-SONIC was removed. `A+B+X+Y` should only stop XR teleop after a final idle command.

## Verification

Passed:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  pytest gear_sonic/tests/test_xr_upperbody_bridge.py -q
# 17 passed
```

Passed:

```bash
PYTHONPYCACHEPREFIX=/tmp/xr_teleoperate_pycache PYTHONDONTWRITEBYTECODE=1 \
  /home/jihun/anaconda3/envs/tv/bin/python -B -m py_compile teleop/teleop_hand_and_arm.py
```

Passed:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python -B -m py_compile \
  gear_sonic/utils/teleop/xr_upperbody_bridge.py \
  gear_sonic/scripts/xr_upperbody_bridge.py
```

## Open Caveats

- Latest controller mapping still needs a full end-to-end sim run after the final idle/quit behavior change.
- Unitree checkout patches are local to `/home/jihun/work/unitree_official/xr_teleoperate`; decide later whether to maintain them as a fork branch, patch file, or documented local dependency.
- Button mapping is expected to work on Quest 3 and Pico 4 Ultra because both expose A/B/X/Y, left/right sticks, index triggers, and grip triggers, but final hardware validation is still required.
