# Arm compliance layer (runtime Kp/Kd profiles)

Switches the stiffness and damping of the G1's **arm motors only** (hardware
indices 15–28: shoulders, elbows, wrists) while SONIC is running. It works the
same under keyboard, PICO teleop and VLA inference, because it sits in the C++
deploy binary right where Kp/Kd are written (`CreatePolicyCommand()`).

**It is off by default.** Without `--arm-compliance` the deploy binary behaves
exactly as before.

## Profiles

Scale factors on SONIC's default gains (`policy_parameters.hpp`: Kp = Jω²,
Kd = 2ζJω). Shoulder/elbow and wrist are scaled separately.

| Profile | Kp scale (shoulder-elbow / wrist) | Kd scale (shoulder-elbow / wrist) | Use |
|---|---|---|---|
| **P0** rigid | 1.00 / 1.00 | 1.00 / 1.00 | SONIC default |
| **P1** soft | 0.50 / 0.80 | 0.71 / 0.90 | handshake, fist bump |
| **P2** compliant | 0.25 / 0.60 | 0.50 / 0.77 | hug |
| **ESTOP** | Kp = 0 (`--compliance-estop-kp`) | Kd = 8 (`--compliance-estop-kd`) | arms go limp, legs keep balancing |

These are **starting values to tune in sim** (edit `BuiltinProfiles()` in
`include/arm_compliance.hpp`, or send custom scales, see below). Kd is scaled
by roughly √(Kp scale) to keep each joint's damping ratio about the same.

Behaviour:
- Profile changes ramp linearly over 0.3 s (`--compliance-slew`, or `"slew_s"` per command).
- ESTOP applies on the next control tick by default, or ramps over
  `--compliance-estop-ramp` seconds (e.g. `--compliance-estop-kd 0 --compliance-estop-ramp 0.5`
  for a smooth go-limp). It is **latched**; only `{"release_estop": true, "profile": ...}`
  leaves it, ramping over `--compliance-estop-release` (1 s).
- If commands stop arriving, the last gains are **held** (never snapped back to
  rigid) and a warning is printed.
- Legs and waist are never touched.

Known limitation: `tau_ff = 0` in this stack, so low shoulder/elbow Kp means the
arms sag under gravity. A gravity-compensation feed-forward is the next step.

## Run it (sim)

```bash
# Terminal 1 — simulator
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 — deploy with the layer enabled
cd gear_sonic_deploy
./deploy.sh --input-type zmq_manager --arm-compliance sim     # or plain `./deploy.sh --arm-compliance sim` for keyboard

# Terminal 3 — PICO streamer (teleop), as usual
python gear_sonic/scripts/pico_manager_thread_server.py --manager

# Terminal 4 — switch profiles from the keyboard
python gear_sonic/scripts/arm_compliance_cli.py
#   0/1/2 = P0/P1/P2, e or SPACE = ESTOP, r = release ESTOP -> P0, q = quit
```

Check the gains actually sent to the motors (reads `rt/lowcmd` over DDS):

```bash
python -m gear_sonic.g1_upper_body_telemetry.inspect_upper_body_state --network-interface lo --print-hz 2
```

## Command format (for the VLA / agent)

ZMQ PUB (the sender binds, default port **5565**), topic `compliance`,
single-frame string `"compliance <json>"`. Re-send the current command at
~10 Hz as a heartbeat.

```json
{"profile": "P1"}
{"profile": "P2", "slew_s": 0.5}
{"kp_scale": 0.4, "kd_scale": 0.6}
{"kp_scale": [14 values], "kd_scale": [14 values]}
{"estop": true}
{"release_estop": true, "profile": "P0"}
```

14-value arrays are ordered L shoulder pitch/roll/yaw, L elbow, L wrist
roll/pitch/yaw, then the same for the right arm. Scales must be in [0, 1.5].

Python example:

```python
import json, zmq
sock = zmq.Context().socket(zmq.PUB)
sock.bind("tcp://*:5565")
sock.send_string("compliance " + json.dumps({"profile": "P2"}))
```

## Flags (`g1_deploy_onnx_ref`)

| Flag | Default | |
|---|---|---|
| `--arm-compliance` | off | enable the layer |
| `--compliance-host` | localhost | host of the command publisher |
| `--compliance-port` | 5565 | its port |
| `--compliance-topic` | compliance | ZMQ topic |
| `--compliance-profile` | P0 | profile at start-up |
| `--compliance-slew` | 0.3 | default ramp time (s) |
| `--compliance-estop-kp` | 0.0 | arm Kp during ESTOP |
| `--compliance-estop-kd` | 8.0 | arm Kd during ESTOP |
| `--compliance-estop-ramp` | 0.0 | ramp time into ESTOP (0 = immediate) |
| `--compliance-estop-release` | 1.0 | ramp time out of ESTOP |
| `--compliance-watchdog` | 1.0 | warn + hold after this many s without commands (0 = off) |

`deploy.sh` passes through `--arm-compliance` and all `--compliance-*` flags above
except `--compliance-topic` and `--compliance-watchdog`.

## Files

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/arm_compliance.hpp` — profiles, command parsing, ramp/ESTOP/watchdog logic
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/arm_compliance_subscriber.hpp` — ZMQ receiver thread
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` — CLI flags, wiring, `Apply()` in `CreatePolicyCommand()`
- `gear_sonic_deploy/deploy.sh` — flag pass-through
- `gear_sonic/scripts/arm_compliance_cli.py` — keyboard sender
