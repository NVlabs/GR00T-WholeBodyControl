# Safe stop (teleop / VLA → soft ready stand)

A stop for social-contact sessions: **the robot stops following the operator or
the VLA, softens its arms first, then lowers its hands along a smooth path to
SONIC's idle stand** (a hug opens wide first). It is independent of the
arm-compliance layer (it reuses its profiles) and works with
`--input-type zmq_manager`, the deploy used for PICO teleop and the VLA
(`run_inference_affectivevla.py`).

The Unitree remote and the `O` key remain the whole-robot emergency stop.

## Trigger

| How | Stop | Release |
|---|---|---|
| Keyboard, typed in the **deploy terminal** | `k` | `u` |
| ZMQ `command` topic (e.g. a future voice node) | bool field `safe_stop` = 1 | bool field `safe_release` = 1 |
| Code (any thread) | `safe_stop::Request("why")` | `safe_stop::Release("why")` |

A voice node can send a message containing only the `safe_stop` field; `start` /
`stop` / `planner` are not needed:

```python
import json, struct, zmq
HEADER_SIZE = 1280
def safe_msg(field):  # field = "safe_stop" or "safe_release"
    hdr = json.dumps({"v": 1, "endian": "le", "count": 1,
                      "fields": [{"name": field, "dtype": "u8", "shape": [1]}]}).encode()
    return b"command" + hdr.ljust(HEADER_SIZE, b"\x00") + struct.pack("B", 1)
# pub = zmq.Context().socket(zmq.PUB); pub.connect("tcp://<host>:5556")  # the port deploy's --zmq-port listens on
# pub.send(safe_msg("safe_stop"))
```

Note: deploy SUBSCRIBES to the sender's PUB socket. The PICO / VLA script already
binds that port, so a voice node must either publish through that script or the
deploy must be pointed at it; the simplest is to add a `safe_stop` call inside the
PICO / VLA script (it already has a PUB socket to deploy).

## What happens

1. **Stop following** (input thread, `zmq_manager.hpp`): the operator's / VLA's
   hands, upper-body / hand targets and walking commands are ignored; walking is
   forced to IDLE. Full-body streaming (POSE) is first switched to PLANNER with
   SONIC's own safety reset. While stopped, any start / switch to streamed
   motion is ignored.
2. **Soften first** (control thread, `safe_stop.hpp` + `CreatePolicyCommand`):
   the arm gains ramp (minimum-jerk, `--safe-stop-soften`, 0.5 s) to the
   **`--safe-stop-profile`** (default `SOFT` = Kp × 0.25, Kd × 0.5), any
   compliance profile — built-in or from `--compliance-profiles` — per joint.
   The hands hold where they were meanwhile. If the compliance layer is also
   running, the softer of the two wins.
3. **Lower the hands** (input thread): the VR hand targets follow a smooth path
   to the rest pose, which the policy follows like an operator slowly lowering
   the hands (it keeps its balance, now with soft arms):
   - **Hug** (both hands more than 12 cm in front of the rest pose): the path
     first opens the hands **outward** by `--safe-stop-open-width` (0.25 m),
     then brings them down beside the body — they never sweep inward across the
     person's back.
   - **Otherwise** (handshake, fist bump): straight down.
   - Peak hand speed `--safe-stop-hand-speed` (0.2 m/s); 2–6 s. Position:
     cubic Bézier with minimum-jerk timing; orientation: slerp.
   - Keyboard planner / VLA tokens without VR hands: planner idle directly.
4. **Hold** soft at the rest pose. **Latched.**
5. **Release** (`u` / `safe_release`): stiffness ramps back over 1 s; the robot
   stays in idle with the hands at the rest pose. VR hands and walking resume
   only after the operator **leaves teleop** on the PICO (then re-enters), so the
   arms never jump. The next start / mode command (a button or key press on the
   PICO or VLA keyboard) is accepted normally.

Not covered: the **BrainCo hands**. `run_inference_affectivevla.py` drives them
directly (not through deploy), so the safe stop cannot open or freeze them; add a
check in that script (e.g. listen for the same stop and open the hands).

## Run

```bash
./deploy.sh --input-type zmq_manager sim      # or real; add --arm-compliance etc. as usual
# in this terminal:  k = safe stop,  u = release
# options: --safe-stop-profile SOFT  --safe-stop-soften 0.5  --safe-stop-hand-speed 0.2
#          --safe-stop-open-width 0.25  --safe-stop-log ~/safe_stop_log.csv
```

Log lines to expect:
```
[SafeStop] STOP (key k): teleop/VLA ignored; arms soften, then the hands are lowered ...
[SafeStop] Hands hold 0.5 s (arms soften), then OPEN WIDE and come down (hug) over 6 s (path 0.65 m)
[SafeStop] Arms -> SOFT over 0.5 s
[SafeStop] RELEASED (key u): ...
[SafeStop] Restoring arm stiffness over 1 s
[SafeStop] Operator left teleop: VR / upper-body targets accepted again.
```
If instead you see `No VR hands to steer`, the stop found no VR hands (keyboard
planner, or VLA tokens) and the robot goes to planner idle directly.

`--safe-stop-log` appends one row per control tick (50 Hz) during each stop and
3 s after the release: `stop, t, soft_blend, vr_live, vrL_xyz, vrR_xyz`, then
the 14 arm joint targets from the policy (`q_target_15..28`), measured
positions (`q_`), speeds (`dq_`) and applied Kp (`kp_`). Use it to see whether
jerky motion comes from the targets or from the tracking.

## Files

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/safe_stop.hpp` — state, triggers, hand path, arm softening, log
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp` — `k` / `u`, command fields, stop / re-engage logic
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` — softening + log in `CreatePolicyCommand()`, `--safe-stop-*` flags, profile lookup
- `gear_sonic_deploy/deploy.sh` — flag pass-through
