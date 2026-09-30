# Arm compliance layer (runtime Kp/Kd profiles)

Switches the stiffness and damping of the G1's **arm motors only** (hardware
indices 15–28: shoulders, elbows, wrists) while SONIC is running. It works the
same under keyboard, PICO teleop and VLA inference, because it sits in the C++
deploy binary right where Kp/Kd are written (`CreatePolicyCommand()`).

**It is off by default.** Without `--arm-compliance` the deploy binary behaves
exactly as before.

## Profiles

A profile scales SONIC's default gains (`policy_parameters.hpp`: Kp = Jω²,
Kd = 2ζJω, ζ = 2) **per side and per joint group**: shoulder (3 joints),
elbow, wrist (3 joints). α = Kp scale, β = Kd scale. β can also be given as a
damping ratio ζ: **β = √α · ζ / 2** (ζ = 2 keeps SONIC's damping ratio; that is
the default when only α is given).

Built-in profiles (Step 0 tests on the real robot, `sonic_v1_1`):

| Profile | Shoulder α/β | Elbow α/β | Wrist α/β | Step 0 note |
|---|---|---|---|---|
| **RIGID** | 1 / 1 | 1 / 1 | 1 / 1 | SONIC default (alias `P0`) |
| **HANDSHAKE** | 1 / 1 | **0.6 / 0.83** | 1 / 1 | soft elbows felt better |
| **HUG** | 0.6 / 0.83 | 0.6 / 0.83 | 0.6 / 0.83 | "maybe", to confirm |
| **FISTBUMP** | 1 / 1 | 1 / 1 | 1 / 1 | only rigid felt natural (pilot, n = 2) |
| **FISTBUMP_SOFTWRIST** | 1 / 1 | 1 / 1 | right only: 0.5 / 0.7 | candidate for the study |
| **SOFT** | 0.25 / 0.5 | 0.25 / 0.5 | 0.25 / 0.5 | uniform soft (ζ = 2) |
| **ESTOP** | controlled stop, see below | | | latched |

### Your own profiles (JSON file, no rebuild)

```json
{"profiles": {
  "MY_HANDSHAKE": {"default": {"kp": 1.0},
                   "elbow":   {"kp": 0.5, "zeta": 2.0},
                   "right_wrist": {"kp": 0.7, "kd": 0.8}}
}}
```

Keys (later ones override earlier ones): `default`, `shoulder` / `elbow` /
`wrist`, `left` / `right`, `left_shoulder` … `right_wrist`. Each entry: `kp`
plus either `kd` or `zeta`. File profiles are added to the built-ins (same name
= override). Load with `--compliance-profiles FILE` in deploy **and**
`--profiles FILE` in the keyboard tool.

`arm_compliance/study_8sets.json` (in `gear_sonic_deploy/`) has the 2×2×2 study
sets: shoulder / elbow / wrist each soft (0.6 / 0.83) or rigid, both arms,
named `S?_E?_W?` with `r` = rigid, `s` = soft (e.g. `Sr_Es_Ws` = soft elbow and
wrist).

### Transitions

- **Minimum-jerk** ramp shape (10r³ − 15r⁴ + 6r⁵): no kink at start or end.
- A joint that gets **stiffer** ramps over `--compliance-stiffen` (1.0 s), one
  that gets **softer** over `--compliance-soften` (0.3 s, fast for safety).
  `--compliance-slew S` sets both; `"slew_s"` in a command overrides both.
- Damping stays on the high side: when stiffening, **Kd finishes first** (in the
  first 40% of the ramp) and Kp follows; when softening, **Kp drops first** and
  Kd follows over the full ramp. The joint is never briefly stiff-but-underdamped.
- If commands stop arriving, the last gains are **held** (never snapped back to
  rigid) and a warning is printed.
- Legs and waist are never touched.

Known limitation: `tau_ff = 0` in this stack, so low shoulder/elbow Kp means the
arms sag under gravity.

### ESTOP (controlled stop)

Lowering the gains alone is not a real stop. SONIC is a whole-body policy: in
VR_3PT teleop it gets the operator's hands as a target (`vr_3point_local_target`
/ `_orn_target`) and keeps trying to reach them — if the arms are held back it
leans the torso and steps, and loses balance. So ESTOP changes **what the policy
is aiming for**, not only the motor gains.

**Handoff (default, `--compliance-estop-mode auto`)** — used when the policy's
reference is the planner (planner walking, VR_3PT teleop):

1. **Handoff** — the policy's hand targets blend smoothly (minimum-jerk,
   `--compliance-handoff`, 1.5 s) from the operator's hands to the planner's idle
   hands (arms down). The **policy itself** brings the arms down and keeps its
   balance. Arm gains soften at the same time (`--compliance-retract-kp`, 0.6 ×
   default, never stiffer than before).
2. **Limp** — hand targets = idle pose; gains ramp to `--compliance-estop-kp` /
   `--compliance-estop-kd` (absolute; default 1.5 / 0.9 ≈ 10 % stiffness, normal
   damping) over `--compliance-estop-ramp` (1 s). **Latched.**
3. **Release** (`r`, `{"release_estop": true, "profile": ...}`) — the reverse:
   hand targets blend back to the operator and gains ramp to the profile over
   `--compliance-estop-release` (1 s). Lower your arms before releasing, or the
   robot's arms will rise to wherever yours are.

**Retract (fallback)** — used when the reference is a motion clip (keyboard `T`)
or full-body POSE streaming, where there is no idle pose to hand over to (and
always with `--compliance-estop-mode retract`): our layer overrides the arm
targets with a minimum-jerk path from the measured pose to the policy's default
arm pose (`--compliance-retract-speed` 45 °/s, 0.8–`--compliance-retract-max`
3 s), then limp and release as above (release blends the targets back).

`--compliance-estop-mode limp` = gains only (no handoff, no override).
Legs and waist always stay under the policy.

## Run it (sim)

All commands from the repo root. Python terminals use the repo's teleop venv
(`bash install_scripts/install_pico.sh` creates it); the deploy terminal uses
`scripts/setup_env.sh`.

```bash
# Terminal 1 — simulator
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 — deploy with the layer enabled
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager --arm-compliance sim     # or `./deploy.sh --arm-compliance sim` for keyboard
# study sets:  add  --compliance-profiles arm_compliance/study_8sets.json
# sonic_v1_1:  add  --cp policy/sonic_v1_1/model --obs-config policy/sonic_v1_1/observation_config.yaml

# Terminal 3 — PICO streamer (teleop), as usual
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager

# Terminal 4 — switch profiles from the keyboard
source .venv_teleop/bin/activate
python gear_sonic/scripts/arm_compliance_cli.py      # add --profiles gear_sonic_deploy/arm_compliance/study_8sets.json for the study sets
#   prints a key menu: 0 RIGID, 1 HANDSHAKE, 2 HUG, ...; e or SPACE = ESTOP, r = release ESTOP -> RIGID, q = quit
```

> **PC with ROS 2 installed?** `setup_env.sh` sources ROS, whose CycloneDDS
> clashes with the Unitree SDK's. Symptoms: `create domain error` in the
> simulator, `free(): invalid pointer` in deploy. Fix: run the Python terminals
> in a shell without ROS (`unset LD_LIBRARY_PATH PYTHONPATH` before activating
> the venv), and in the deploy terminal put the Unitree DDS libs first:
> `export LD_LIBRARY_PATH=$PWD/thirdparty/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH`
> (after `source scripts/setup_env.sh`).

Check the gains actually sent to the motors (reads `rt/lowcmd` over DDS):

```bash
source .venv_teleop/bin/activate
python -m gear_sonic.g1_upper_body_telemetry.inspect_upper_body_state --network-interface lo --print-hz 2
```

### Using `sonic_v1_1` (the model used in the Step 0 robot tests)

This fork's deploy code supports all `sonic_v1_1` observation terms; only the
model files are missing. Fetch them with NVIDIA's current downloader (from the
repo root, teleop venv active):

```bash
git fetch nvlabs main   # remote: https://github.com/NVlabs/GR00T-WholeBodyControl.git
git show nvlabs/main:download_from_hf.py > /tmp/download_from_hf_nvlabs.py
python /tmp/download_from_hf_nvlabs.py --sonic-v1-1
```

## User study (`arm_compliance_study.py`)

Runs one participant through all sets of a profile file (e.g. the 2×2×2
`study_8sets.json`): per set the robot switches stiffness, the participant does
each gesture and rates it right after. Use it **instead of** the keyboard tool
(both publish on port 5565). Deploy must load the same file.

```bash
# deploy (T2): add  --compliance-profiles arm_compliance/study_8sets.json
python gear_sonic/scripts/arm_compliance_study.py \
    --profiles gear_sonic_deploy/arm_compliance/study_8sets.json \
    --control-mode teleop --blind
# resume an interrupted session:   --resume P03
```

- **Intake:** consent check, then age, gender, height, weight (optional),
  dominant hand, robot experience, prior contact with a humanoid. The name goes
  only to `names_key.csv`; data files use the ID (P01, P02, ...).
- **Practice:** always first, always the built-in **RIGID** profile, one Enter per
  gesture, no ratings (familiarisation). `--rate-practice` asks the ratings too.
  The analysed rigid condition is `Sr_Er_Wr` inside the 8 counterbalanced sets.
  `--no-practice` exists only for debugging.
- **Order:** sets in a balanced Latin square (Williams design) by participant
  number; gesture order rotates across sets. Fixed at the first run, reused on
  resume.
- **Per gesture:** one Enter when the gesture is done (start time = when the
  gesture is announced, end = Enter; for syncing with robot telemetry), then 3 ratings on 1–7: **perceived safety**, **comfort** and
  **naturalness** (edit `QUESTIONS` at the top of the script to add items, e.g.
  perceived softness as a manipulation check). A CSV started with the old 2-question
  version is upgraded automatically on `--resume` (old rows keep an empty
  naturalness cell). Then `y`/`n`: was the trial valid
  (`n` e.g. robot stumbled; a reason is asked).
- **Commands at any prompt:** `!e` ESTOP, `!r` release, `!b` break (robot →
  RIGID), `!n` note, `!s` skip trial, `!q` save and quit.
- **Output** (default `~/arm_compliance_study_data/`, outside the repo):
  `P03/P03_ratings.csv` (one row per trial, with the shoulder/elbow/wrist
  factor levels for analysis), `P03/P03_session.json` (demographics, order,
  profile values, questions), `names_key.csv`.
- `--blind` shows set letters (A–H) instead of profile names, in case the
  participant can see the screen. `--reps N` repeats each gesture.

## Command format (for the VLA / agent)

ZMQ PUB (the sender binds, default port **5565**), topic `compliance`,
single-frame string `"compliance <json>"`. Re-send the current command at
~10 Hz as a heartbeat.

```json
{"profile": "HANDSHAKE"}
{"profile": "HUG", "slew_s": 0.5}
{"kp_scale": 0.4, "kd_scale": 0.6}
{"kp_scale": [14 values], "kd_scale": [14 values]}
{"estop": true}
{"release_estop": true, "profile": "RIGID"}
```

14-value arrays are ordered L shoulder pitch/roll/yaw, L elbow, L wrist
roll/pitch/yaw, then the same for the right arm. Kp scales must be in [0, 1.5],
Kd scales in [0, 3].

Python example:

```python
import json, zmq
sock = zmq.Context().socket(zmq.PUB)
sock.bind("tcp://*:5565")
sock.send_string("compliance " + json.dumps({"profile": "HUG"}))
```

## Flags (`g1_deploy_onnx_ref`)

| Flag | Default | |
|---|---|---|
| `--arm-compliance` | off | enable the layer |
| `--compliance-host` | localhost | host of the command publisher |
| `--compliance-port` | 5565 | its port |
| `--compliance-topic` | compliance | ZMQ topic |
| `--compliance-profiles` | – | JSON file with extra profiles |
| `--compliance-profile` | RIGID | profile at start-up |
| `--compliance-soften` | 0.3 | ramp time when a joint gets softer (s) |
| `--compliance-stiffen` | 1.0 | ramp time when a joint gets stiffer (s) |
| `--compliance-slew` | – | sets both ramp times |
| `--compliance-estop-mode` | auto | `auto` (handoff when the reference is the planner, else retract), `retract`, or `limp` (gains only) |
| `--compliance-handoff` | 1.5 | hand-target blend time in handoff ESTOP (s) |
| `--compliance-retract-speed` | 45 | peak joint speed of the retract (°/s) |
| `--compliance-retract-kp` | 0.6 | arm stiffness while retracting (× default Kp; Kd × √) |
| `--compliance-retract-max` | 3.0 | longest retract (s) |
| `--compliance-estop-kp` | 1.5 | arm Kp once retracted (absolute; default arm Kp ≈ 14.3) |
| `--compliance-estop-kd` | 0.9 | arm Kd once retracted (absolute; default arm Kd ≈ 0.9) |
| `--compliance-estop-ramp` | 1.0 | ramp to the ESTOP gains (s) |
| `--compliance-estop-release` | 1.0 | blend back to the policy after release (s) |
| `--compliance-watchdog` | 1.0 | warn + hold after this many s without commands (0 = off) |

`deploy.sh` passes through `--arm-compliance` and all `--compliance-*` flags above
except `--compliance-topic` and `--compliance-watchdog`.

## Files

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/arm_compliance.hpp` — profiles, command parsing, gain ramps, controlled-stop ESTOP, watchdog
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/arm_compliance_subscriber.hpp` — ZMQ receiver thread
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` — CLI flags, wiring, `Apply()` in `CreatePolicyCommand()`
- `gear_sonic_deploy/deploy.sh` — flag pass-through
- `gear_sonic/scripts/arm_compliance_cli.py` — keyboard sender
- `gear_sonic/scripts/arm_compliance_study.py` — user-study session runner
- `gear_sonic_deploy/arm_compliance/study_8sets.json` — 2×2×2 study profiles
