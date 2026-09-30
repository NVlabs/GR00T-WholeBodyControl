# Safe stop (teleop / VLA → soft ready stand)

A stop for social-contact sessions: **the robot stops following the operator or
the VLA, goes to SONIC's idle stand with the arms lowered by the policy itself,
and then softens its arms slowly.** It is independent of the arm-compliance
layer and works with `--input-type zmq_manager`, the deploy used for PICO
teleop and the VLA (`run_inference_affectivevla.py`).

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

1. **Stop following** (input thread, `zmq_manager.hpp`)
   - VR 3-point teleop (planner mode): the operator's hands, upper-body / hand
     targets and walking commands are ignored; locomotion is forced to IDLE.
     The **VR hand targets follow a smooth minimum-jerk path** from where the
     hands were to the rest pose (peak hand speed `--safe-stop-hand-speed`,
     0.2 m/s, 2–5 s) and then stay there. The policy follows it like an operator
     slowly lowering the hands, so it keeps its balance. (Dropping the VR hands
     instantly, as in the first version, lowered the arms too fast.)
   - Streamed motion (full-body POSE, **VLA tokens**): switched to PLANNER idle
     with SONIC's own safety reset (the same path as a `{planner: true}` command).
   - While stopped, any request to switch to streamed motion or to start is ignored.
2. **Soften slowly** (control thread, `safe_stop.hpp` + `CreatePolicyCommand`)
   - Wait until the lowering path is done and the arms are still (all arm
     joints slower than 0.15 rad/s for 0.3 s; at the latest 2 s after the path).
     The log reports the peak arm joint speed of the lowering — use it to
     compare settings.
   - Then ramp the arm stiffness (minimum-jerk) over `--safe-stop-soften` (2 s)
     to `--safe-stop-kp` × default Kp (0.5); Kd × √0.5 keeps the damping ratio.
     If the compliance layer is also running, the softer of the two wins.
   - **Latched.**
3. **Release** (`u` / `safe_release`)
   - Arm stiffness ramps back to normal over 1 s; the robot stays in idle.
   - VR hands and walking from the planner topic resume only after the operator
     **leaves teleop** on the PICO (a planner message without VR hands), then
     re-enters — so the arms never jump to where the operator's hands happen to be.
   - The next start / mode command (a button or key press on the PICO or VLA
     keyboard) is accepted normally, e.g. the VLA's "start" to resume.

Not covered: the **BrainCo hands**. `run_inference_affectivevla.py` drives them
directly (not through deploy), so the safe stop cannot open or freeze them; add a
check in that script (e.g. listen for the same stop and open the hands).

## Run

```bash
./deploy.sh --input-type zmq_manager sim      # or real; add --arm-compliance etc. as usual
# in this terminal:  k = safe stop,  u = release
# options: --safe-stop-hand-speed 0.2  --safe-stop-kp 0.5  --safe-stop-soften 2.0  --safe-stop-settle-max 4.0
```

Log lines to expect:
```
[SafeStop] STOP (key k): teleop/VLA ignored, going to ready stand; ...
[SafeStop] Lowering the hands to the rest pose over 3.9 s (farthest hand 0.42 m away)   (VR teleop)   or
[SafeStop] Streamed motion -> PLANNER idle (safety reset)   (VLA / full body)
[SafeStop] Arms down and still after 4.3 s (peak arm joint speed 0.6 rad/s) -> softening arms to Kp x0.5 over 2 s
[SafeStop] RELEASED (key u): ...
[SafeStop] Operator left teleop: VR / upper-body targets accepted again.
```

## Files

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/safe_stop.hpp` — state, triggers, arm softening
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp` — `k` / `u`, command fields, stop / re-engage logic
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` — softening in `CreatePolicyCommand()`, `--safe-stop-*` flags
- `gear_sonic_deploy/deploy.sh` — flag pass-through
