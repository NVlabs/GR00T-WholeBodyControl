# PICO 4 Ultra + Unitree G1 EDU

This guide has two mutually exclusive robot paths:

- The released path uses the repository's default **XRoboToolkit** input and
  released 29-DoF SONIC policy.
- A rev-1.0 G1 reporting `mode_machine == 4` must use the true 23-DoF
  retraining and promotion path below. It must never run the released 29-DoF
  decoder or obtain 23 outputs by masking six outputs at runtime.

Neither path uses Rocco, the custom Quest bridge, ALVR, or the CloudXR/Thor
path.

## Required hardware

- PICO 4 Ultra headset with User OS newer than 5.12
- Both PICO controllers
- Two actual PICO Motion Tracker sensors, each secured in an ankle strap
- Unitree G1 EDU with either the released 29-DoF body configuration or the
  rev-1.0 23-DoF body configuration. Confirm which embodiment is physically
  present before selecting a policy.
- Low-latency Wi-Fi between PICO and workstation
- Dedicated Ethernet between workstation and G1
- Gantry, clear 3 m operating area, and a second operator at the deployment
  terminal

A strap by itself is only a mount; it does not provide ankle tracking. The
released SONIC model is not compatible with the 23-DoF G1 body. Adaptation and
retraining are required for that embodiment.

## Rev-1.0 true 23-DoF path (`mode_machine == 4`)

This path is intentionally fail-closed. Repository initialization checkpoints
prove shapes and warm-start lineage only; they are not trained deployment
artifacts and the ONNX exporter rejects them.

The embodiment contract is exact:

- Hardware motor slots kept: `0-12`, `15-19`, `22-26`
- Hardware motor slots absent/locked: `13`, `14`, `20`, `21`, `27`, `28`
- Decoder output: native PhysX/IsaacLab 23-joint order, exactly 23 values
- Released canonical observation slots absent on this body: `5`, `8`,
  `25-28`
- Missing position observations: fixed default joint positions
- Missing velocity and previous-action observations: zero in every history
  frame

The PICO must remain in **Full body** / BodyTracking mode. Both ankle bands
feed PICO's fused 24-role body solution; left and right ankle roles are indices
7 and 8. BodyTracking and raw MotionTracking are exclusive modes, so combining
retained records from the two modes is forbidden. The hardened stream must
prove calibration, `BT_VALID`, two distinct connected bands, all 24 body
roles, advancing timestamps/sample sequence, and the versioned derivative
layout.

Once that stream is live, capture immutable XR24 replay evidence without
opening DDS or a robot channel:

```bash
python -m gear_sonic.scripts.capture_g1_23dof_pico_raw \
  --output /path/to/pico-xr24-capture.json \
  --pico-client-apk-sha256 <hardened-apk-sha256>
```

The default 60 advancing frames cover both reference profiles. This is raw
evidence only: it cannot approve the adapter or produce policy inputs until the
capture is replayed through the pinned NVIDIA SOMA backend and the exact
XR24 coordinate/role adapter passes review.

Before training, the full preflight must pass without `--skip-runtime` or
`--skip-motion-data`:

```bash
python -m gear_sonic.scripts.preflight_g1_23dof_training \
  --checkpoint sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  --low-latency \
  --json
```

Train the genuine low-latency 23-output policy only from the pinned
initialization checkpoint and complete BONES-SEED motion tree:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_g1_23dof_rev_1_0_low_latency_warm_start \
  +checkpoint=sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  num_envs=4096 headless=True
```

The repository trainer still uses IsaacLab. MuJoCo can independently test and
promote a genuinely trained checkpoint, but it cannot relabel either
`*_init.pt` checkpoint as trained or replace the missing optimizer updates and
training-data lineage.

### Generic non-causal MuJoCo artifact path

This subsection documents the older generic true-23 artifact workflow. Do not
mix these files with the causal V12 package described below: the current V12
gate requires its schema-2 causal promotion and rejects this legacy promotion
schema.

IsaacLab is not required for this validation path. Install the exact approved
MuJoCo version and the artifact tools:

```bash
python -m pip install -e "gear_sonic[sim,artifact_export]"
```

First export immutable, explicitly non-deployable candidate ONNX files from a
genuinely trained weights-only checkpoint:

```bash
python -m gear_sonic.scripts.export_g1_23dof_mujoco_candidate \
  <trained>.promotion.pt \
  <prefix>.encoder.candidate.onnx \
  <prefix>.decoder.candidate.onnx \
  <prefix>.candidate.metadata.json
```

Then run the exact 23-actuator model at 500 Hz physics / 50 Hz policy rate.
The approved campaign executes 198 five-second episodes across nominal, 50%,
and 100% deterministic disturbance scenarios and writes raw JSONL traces:

```bash
python -m gear_sonic.scripts.run_g1_23dof_mujoco_sim2sim \
  --checkpoint <trained>.promotion.pt \
  --encoder-onnx <prefix>.encoder.candidate.onnx \
  --decoder-onnx <prefix>.decoder.candidate.onnx \
  --metadata <prefix>.candidate.metadata.json \
  --output <prefix>.mujoco-sim2sim.json
```

Only a passing report can create a promotion sidecar. Promotion replays every
record through the exact ONNX pair and MuJoCo model; it does not trust report
pass booleans or hashes alone:

```bash
python -m gear_sonic.scripts.promote_g1_23dof_mujoco_candidate \
  --sidecar <prefix>.mujoco-promotion.json \
  --checkpoint <trained>.promotion.pt \
  --encoder-onnx <prefix>.encoder.candidate.onnx \
  --decoder-onnx <prefix>.decoder.candidate.onnx \
  --metadata <prefix>.candidate.metadata.json \
  --report <prefix>.mujoco-sim2sim.json
```

Recheck the package later without writing anything by adding
`--verify-only`. The promotion sidecar authorizes those immutable ONNX bytes
for read-only deployment shadow testing; it deliberately keeps
`active_motor_control_authorized=false`.

With the robot connected, the native gate can then dry-run the same pair and
observe five advancing CRC-valid `mode_machine == 4` LowState samples:

```bash
gear_sonic_deploy/target/release/g1_true23_shadow_gate \
  --mode shadow \
  --network <robot-facing-interface> \
  --encoder <prefix>.encoder.candidate.onnx \
  --decoder <prefix>.decoder.candidate.onnx \
  --metadata <prefix>.candidate.metadata.json \
  --promotion <prefix>.mujoco-promotion.json \
  --once
```

That binary has no LowCmd publisher, command writer, or motion-mode transition.
It cannot actuate the robot.

The local initialization diagnostic is
`sonic_release/diagnostics/g1_23dof_rev_1_0_init_full_v2_20260730.json`.
It covered 198 episodes and 49,500 control records with no falls or non-finite
values, but correctly failed promotion because 697 records violated the soft
joint-position envelope. It is test evidence, not a trained policy.

### Current causal V12 package and promoted live shadow

This workstation's selected causal package is optimizer update 50 from the
true-23 V12 recovery run. Independent verification replayed 264 episodes and
66,000 MuJoCo control records across three seeds and four scenarios. It found
zero falls, terminations, non-finite policy values, joint-limit violations,
velocity-limit violations, or effort-limit violations. Worst disturbance
recovery was 1.92 seconds. These results authorize immutable deployment bytes
for read-only shadow only; they do **not** authorize motor control.

The exact files inside Ubuntu 22.04 are:

```text
/root/g1_true23_runs/causal_safe_target_v12_smoke100/checkpoints/causal_model_50.pt
/root/g1_true23_eval/v12/causal_model_50_diagnostic.encoder.onnx
/root/g1_true23_eval/v12/causal_model_50_diagnostic.decoder.onnx
/root/g1_true23_eval/v12/causal_model_50_diagnostic.diagnostic.json
/root/g1_true23_eval/v12/causal_model_50_mujoco_full_diagnostic.json
/root/g1_true23_eval/v12/causal_model_50_causal_mujoco_deployment_bytes_promotion.json
```

Re-verify every source, checkpoint, ONNX, metadata, MuJoCo record, trace,
configuration, MJCF, and archived producer byte without writing:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl
/root/.venvs/g1_true23_mjlab/bin/python \
  -m gear_sonic.scripts.promote_g1_true23_causal_mujoco \
  --output /root/g1_true23_eval/v12/causal_model_50_causal_mujoco_deployment_bytes_promotion.json \
  --checkpoint /root/g1_true23_runs/causal_safe_target_v12_smoke100/checkpoints/causal_model_50.pt \
  --encoder-onnx /root/g1_true23_eval/v12/causal_model_50_diagnostic.encoder.onnx \
  --decoder-onnx /root/g1_true23_eval/v12/causal_model_50_diagnostic.decoder.onnx \
  --metadata /root/g1_true23_eval/v12/causal_model_50_diagnostic.diagnostic.json \
  --full-report /root/g1_true23_eval/v12/causal_model_50_mujoco_full_diagnostic.json \
  --verify-only
```

Validate the promoted ONNX pair without opening PICO, DDS, or robot APIs:

```bash
./install_scripts/run_g1_true23_v12_promoted_shadow.sh --check-only
```

For the real read-only integration, first require all of these conditions:

- PICO reports awake; hardened XRoboToolkit client is foregrounded.
- PC Service is `192.168.1.182` and reports `WORKING`.
- Full-body mode is active, calibrated, and advancing with both ankle trackers.
- Both controllers retain position-and-rotation tracking.
- Robot Ethernet receives advancing CRC-valid LowState with
  `mode_machine == 4`.
- No Rocco process, replay process, custom publisher, simulator, or LowCmd
  publisher is running.

Then run the one-shot shadow. This script cannot create a LowCmd publisher:

```powershell
wsl.exe -d Ubuntu-22.04 -- /mnt/z/codex/GR00T-WholeBodyControl/install_scripts/run_g1_true23_v12_promoted_shadow.sh --network eth0 --frames 100 --timeout-seconds 120
```

Success requires both generated evidence files to end in PASS, with 100
consecutive accepted causal action frames, fresh mode-4 LowState, exact 20 ms
control timestamps, zero CRC rejects, and no limit, slew, freshness, or
inference-deadline violation. A timeout, sleeping headset, missing tracker,
stale LowState, or failed record leaves active control unauthorized.

Only after this fresh promoted shadow passes may an exact, hash-bound
gantry-active sidecar be created using a separately supplied authorization ID
and literal `I_CONFIRM_G1_TRUE23_STAGE1_GANTRY` phrase. Stage one remains
gantry-only; free-standing control is explicitly forbidden.

Within five minutes of the successful shadow, set its exact path and create
the session-bound stage-one sidecar. The operator must type the phrase
deliberately; do not store it in a shell profile or wrapper:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl
export LIVE_SHADOW='/root/g1_true23_runs/live/v12_promoted_shadow_YYYYMMDD_HHMMSS.jsonl'
export AUTHORIZATION_ID='gantry-YYYYMMDD-operator'
export ACTIVE_SIDECAR=/root/g1_true23_eval/v12/causal_model_50_causal_gantry_active_promotion_${AUTHORIZATION_ID}.json

/root/.venvs/g1_true23_mjlab/bin/python \
  -m gear_sonic.scripts.authorize_g1_true23_causal_gantry \
  --output "$ACTIVE_SIDECAR" \
  --promotion /root/g1_true23_eval/v12/causal_model_50_causal_mujoco_deployment_bytes_promotion.json \
  --checkpoint /root/g1_true23_runs/causal_safe_target_v12_smoke100/checkpoints/causal_model_50.pt \
  --encoder-onnx /root/g1_true23_eval/v12/causal_model_50_diagnostic.encoder.onnx \
  --decoder-onnx /root/g1_true23_eval/v12/causal_model_50_diagnostic.decoder.onnx \
  --metadata /root/g1_true23_eval/v12/causal_model_50_diagnostic.diagnostic.json \
  --full-report /root/g1_true23_eval/v12/causal_model_50_mujoco_full_diagnostic.json \
  --live-shadow-evidence "$LIVE_SHADOW" \
  --authorization-id "$AUTHORIZATION_ID" \
  --gantry-authorize I_CONFIRM_G1_TRUE23_STAGE1_GANTRY
```

The authorizer reparses every JSONL record, re-verifies all 66,000 MuJoCo
records and material hashes, rejects diagnostic/failed/trailing evidence, and
rehashes every input before writing. It cannot create a sidecar from a shadow
older than 300 seconds.

Before opening any robot interface, dry-validate the resulting sidecar and
all bound bytes:

```bash
gear_sonic_deploy/target/release/g1_true23_active_gantry \
  --network eth0 \
  --pico-endpoint tcp://127.0.0.1:5557 \
  --authorization-id "$AUTHORIZATION_ID" \
  --encoder /root/g1_true23_eval/v12/causal_model_50_diagnostic.encoder.onnx \
  --decoder /root/g1_true23_eval/v12/causal_model_50_diagnostic.decoder.onnx \
  --metadata /root/g1_true23_eval/v12/causal_model_50_diagnostic.diagnostic.json \
  --promotion /root/g1_true23_eval/v12/causal_model_50_causal_mujoco_deployment_bytes_promotion.json \
  --active-promotion "$ACTIVE_SIDECAR" \
  --live-shadow-evidence "$LIVE_SHADOW" \
  --validate-only
```

`--validate-only` requires the network and PICO URI strings so they can match
the bound live-shadow evidence, but it does not open either interface. It
rejects execution and authorization arguments, opens no DDS channel, and
creates no LowCmd publisher. Do not proceed unless it prints the exact causal
gantry-promotion PASS.

Use the paired publisher evidence from the same promoted-shadow run. The
launcher rejects mismatched timestamps, failed/stale packets, changed
publisher, capture-worker, XRT, APK, controller, or model bytes, and any known
competing LowCmd controller. First validate the complete launch contract:

```bash
export PUBLISHER_EVIDENCE='/root/g1_true23_runs/live/v12_pico_causal_publisher_YYYYMMDD_HHMMSS.jsonl'

./install_scripts/run_g1_true23_v12_stage1_gantry.sh \
  --active-promotion "$ACTIVE_SIDECAR" \
  --live-shadow-evidence "$LIVE_SHADOW" \
  --publisher-evidence "$PUBLISHER_EVIDENCE" \
  --authorization-id "$AUTHORIZATION_ID" \
  --network eth0 \
  --pico-endpoint tcp://127.0.0.1:5557 \
  --validate-only
```

For the first bounded gantry session, use a correctly rated overhead gantry
with correct slack, a clear fall zone, a second observer, and a tested physical
e-stop held ready. Run from an interactive TTY. The launcher asks for the
authorization phrase without echoing or storing it. After READY, the controller
publishes damping-only LowCmd frames before arming (zero feedforward and
``kp=0``). Hold L2, then press A on a rising edge to arm. Releasing L2, pressing
B/R2, or using the e-stop is a terminal stop. A successful evidence run must
complete the full requested 20 seconds; an intentional early stop is safe but
does not PASS. If A was pressed before READY, release it fully, keep L2 held,
then press A again after READY:

```bash
./install_scripts/run_g1_true23_v12_stage1_gantry.sh \
  --active-promotion "$ACTIVE_SIDECAR" \
  --live-shadow-evidence "$LIVE_SHADOW" \
  --publisher-evidence "$PUBLISHER_EVIDENCE" \
  --authorization-id "$AUTHORIZATION_ID" \
  --network eth0 \
  --pico-endpoint tcp://127.0.0.1:5557 \
  --duration-seconds 20 \
  --execute-stage-one
```

Status 0 requires at least 100 successfully written armed policy commands,
then 250 fail-safe damping commands, plus a strictly validated immutable active
evidence file. No A press, early fault, publisher-write failure, missing
damping, or output with no policy actuation returns nonzero and never reports
PASS. This stage remains gantry-only.

### IsaacLab validation alternative

Training evidence must record at least 50 optimizer updates, changed policy
weights, the exact approved source checkpoint, and the exact BONES archive and
processed-tree manifests. First execute the trained 267-input encoder and
994-input decoder on CPU and prove exactly 23 finite float32 outputs, without
launching IsaacLab:

```bash
python -m gear_sonic.scripts.run_g1_23dof_sim_validation \
  --checkpoint <trained>.promotion.pt \
  --output <trained>.simulation.json \
  --dry-run
```

Then run hash-bound nominal and disturbance validation. The normal command
repeats the same output dry-run before it launches IsaacLab:

```bash
python -m gear_sonic.scripts.run_g1_23dof_sim_validation \
  --checkpoint <trained>.promotion.pt \
  --output <trained>.simulation.json
```

Only a trained promotion checkpoint plus its matching raw simulation traces
can be exported:

```bash
python -m gear_sonic.scripts.export_g1_23dof_onnx \
  <trained>.promotion.pt \
  <trained>.simulation.json \
  <output-prefix>
```

The paired encoder/decoder metadata binds the 267-input encoder, 994-input
decoder, 23-output decoder, checkpoint, training evidence, simulation report,
runtime-source manifest, and dataset manifest. Deployment must next pass the
read-only native shadow gate on five advancing CRC-valid
`mode_machine == 4` samples. Gantry testing comes only after fresh live-shadow
evidence; free-standing control comes only after successful gantry review.

Do not use the released 29-DoF deployment commands later in this guide for a
23-DoF robot. Until retraining, disturbance validation, paired ONNX export,
live shadow, and gantry evidence all exist, the correct result is **NO-GO**.

## Rocco teleoperation comparison

This comparison comes from a read-only audit of earlier Rocco/Quest
teleoperation logs and Codex task history. This profile neither imports nor
modifies Rocco.

The 2026-08-03 audit rechecked these claims against Rocco source without
writing that tree. Evidence locations, relative to the Rocco root, were
`unigichat/server/app.py:1181`, `unigichat/server/app.py:1268`,
`unigichat/server/app.py:1409`, `unigichat/server/static/dashboard.js:1082`,
`unigichat/server/robot/supervisor.py:337`,
`unigichat/server/robot/supervisor.py:978`,
`unigichat/server/robot/unitree_gantry.py:211`, and
`unigichat/server/robot/unitree_gantry.py:268`.

| Earlier failure | Why it was unsafe or incomplete | This profile |
|---|---|---|
| Head and controllers arrived, but no proven 24-joint body stream | Three points can drive arms, but do not prove whole-body SONIC input | Requires an atomic, advancing 24-joint body frame plus authoritative tracked/valid state for both controllers |
| A general XR timestamp could advance while body data stayed unchanged | Unrelated controller/head traffic could make a stale body pose look fresh | Uses the body-specific PICO joint clock and paired-controller source clock; packet arrival alone cannot refresh a cached body pose |
| Head height and wrist/autocalibration offsets disagreed with the robot reference frame | A large frame offset can command an abrupt, physically impossible target | Calibration commits transactionally only after finite, normalized, bounded, and continuous poses pass validation |
| Stale, replayed, or non-finite samples could continue driving the last command | Tracking or publisher loss did not reliably remove command authority | Python watchdog and independent native 500 ms lease both expire; stop, exceptions, and tracking loss send repeated STOP commands |
| Ping/TCP checks passed while Unitree DDS repeatedly failed; host, robot, and Wi-Fi IPs were mixed | Reachability did not prove LowState or correct DDS interface binding | Live preflight requires a local robot-facing interface and real advancing LowState samples; the robot IP is never used as the host interface |
| Record/replay, custom publishers, relays, and the deployment process could overlap | More than one command owner makes stop and mode state ambiguous | Runbook requires one command owner and explicit shutdown of all old publishers before deployment |
| Browser pause/resume posted to local TCP 17603, then the Orin proxy fire-and-forgot UDP to hard-coded `192.168.1.182:17602`; neither repository contains the receiving listener | UDP `sendto()` success was reported as command success without acknowledgement, ownership, replay protection, or freshness | Start/stop travels through the validated ZMQ manager protocol and native lease state on ports 5556/5557 |
| Dashboard “ready” ignored robot connection, PICO frames, teleop process, `mode_machine`, trained policy, ONNX, and inference; stored bridge status did not expire | A green UI could describe stale preparation state rather than a working control loop | Readiness binds fresh PICO, LowState, trained checkpoint, simulation traces, paired ONNX, integrated inference, and exact embodiment evidence |
| Rocco `g1_23` selected only ten arm motors (`15-19`, `22-26`) | The label did not mean a 23-output whole-body policy and provided no legs, waist, balance policy, or ankle-tracker retargeting | True23 means exactly 23 native policy outputs and the complete mode-4 hardware mapping |
| Robot detection defaulted to 29-DoF without LowState and inferred 23-DoF from a few missing/non-finite slots; it recorded but did not gate `mode_machine` | Finite placeholder slots could misclassify a rev-1.0 robot and select an incompatible policy | True23 requires five advancing CRC-valid LowState samples with `mode_machine == 4` and latches closed if the mode changes |
| “Prepare” stopped old arm publishers and released MotionSwitcher ownership to the Unitree app | Ownership release did not launch GR00T, claim a receiver session, prove a policy, or establish teleoperation | Deployment and manager form a fresh receiver/publisher session, then prove tracking and policy readiness independently |
| The working official PICO route proved controller-only arm teleoperation | Advancing packet timestamps did not prove either controller was still position-and-rotation tracked, and arm motion did not validate PICO Motion Trackers | Control requires `BT_VALID`, calibration, successful body APIs, at least two distinct connected trackers, 24 calculated body roles, and authoritative position-and-rotation tracking for each controller |

## One-time workstation setup

Use an x86_64 workstation with native Ubuntu 22.04 or an Ubuntu 22.04 WSL2
filesystem. The PC-service package in this procedure is AMD64 and is not an
onboard-Orin installer. If using WSL2, configure mirrored networking before
setup:

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
```

Save this as `%UserProfile%\.wslconfig` on Windows, then run
`wsl --shutdown` from PowerShell and reopen Ubuntu before continuing.

Set `GR00T_WBC_ROOT` when the clone is not under your Linux home directory.
`GR00T_WBC_CACHE_ROOT` defaults to a user-writable native cache. On WSL, point
it at a large mounted drive when available, for example:

```bash
export GR00T_WBC_CACHE_ROOT=/mnt/y/WSL-Caches/groot-wbc
```

Persist custom root values in your shell profile or re-export them in every
new terminal.

Before pulling LFS artifacts or running the PICO installer, install the
deployment prerequisites and exact TensorRT release from the
[deployment installation guide](installation_deploy.md). Desktop x86_64
requires TensorRT 10.13; G1 onboard Orin requires TensorRT 10.7 with JetPack 6.
Do not substitute another TensorRT version.

Then install deployment dependencies, fetch required LFS artifacts, install
the supplied XRoboToolkit PC service package, and install the regular-PICO
profile:

```bash
export GR00T_WBC_ROOT="${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
export GR00T_WBC_CACHE_ROOT="${GR00T_WBC_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/groot-wbc}"
cd "$GR00T_WBC_ROOT"
export UV_CACHE_DIR="$GR00T_WBC_CACHE_ROOT/uv"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$UV_CACHE_DIR"
bash gear_sonic_deploy/scripts/install_deps.sh
git lfs pull --include="decoupled_wbc/control/teleop/device/pico/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb,external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/lib/libPXREARobotSDK.so,gear_sonic/data/robot_model/model_data/g1/meshes/**,gear_sonic_deploy/g1/**,gear_sonic_deploy/reference/example/**,gear_sonic_deploy/thirdparty/unitree_sdk2/lib/x86_64/**,gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64/**"
sudo apt-get install xdg-utils desktop-file-utils
sudo apt install ./decoupled_wbc/control/teleop/device/pico/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
SKIP_ISAAC_TELEOP=1 bash install_scripts/install_pico.sh
```

On WSL, a mounted-drive cache prevents large Python/CUDA wheels from filling
the WSL system drive. Native Linux uses the user cache by default; neither
default requires creating a root-owned directory under `/mnt`.
The installer pins the desktop Python environment to the official PyTorch
2.9.1 CUDA 12.8 wheel; this remains compatible with CUDA 12.x drivers and
avoids accidentally selecting a CUDA 13 wheel on a CUDA 12 driver.

### Hardened PICO client required for real control

The stock XRoboToolkit PICO v1.1.1 APK is useful for raw visualization and
protocol diagnostics, but it is not accepted by the manager even in
simulation. It does not transmit `GetBodyTrackingState`, continuous
tracker connection/calibration state, or the body-data API return code. It can
also retain an old Body object after tracking becomes invalid. An advancing
HEAD joint timestamp prevents a cached Body frame from appearing fresh, but
does not independently prove that both ankle trackers remain healthy. Its
packet clock can likewise advance while a disconnected controller retains an
identity or cached pose.

Real G1 control therefore fails closed unless the side-by-side hardened client
(`com.xrobotoolkit.client.hardened`) is running. Build it from the unmodified
official v1.1.1 source and the repository patch, then install it over ADB:

```powershell
Set-Location -LiteralPath C:\path\to\GR00T-WholeBodyControl
.\install_scripts\build_xrobotoolkit_hardened.ps1 -InstallToConnectedPico
```

The script leaves the official `com.xrobotoolkit.client` package installed.
Only one XRoboToolkit client may send at a time. Force-stop the stock package,
open **XRoboToolkit Hardened**, and use the same workstation IP and UI options.
Preflight verifies the hardened protocol and refuses the stock APK for
real-robot readiness.

Source the checked environment profile before build, preflight, or deployment:

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
source install_scripts/pico_g1_env.sh
```

Download the lower-lookahead policy:

```bash
export GR00T_WBC_ROOT="${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
export GR00T_WBC_CACHE_ROOT="${GR00T_WBC_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/groot-wbc}"
cd "$GR00T_WBC_ROOT"
source .venv_teleop/bin/activate
uv pip install huggingface_hub
GEAR_SONIC_REVISION=7c90a56cfe04788c4f041daeef5b1e12930675ad
MODEL_CACHE="$GR00T_WBC_CACHE_ROOT/models/$GEAR_SONIC_REVISION"
mkdir -p "$MODEL_CACHE"
HF_HOME="$GR00T_WBC_CACHE_ROOT/huggingface" \
  python download_from_hf.py \
    --revision "$GEAR_SONIC_REVISION" \
    --low-latency \
    --output-dir "$MODEL_CACHE"

mkdir -p gear_sonic_deploy/policy gear_sonic_deploy/planner/target_vel
ln -sfnT "$MODEL_CACHE/policy/low_latency" \
  gear_sonic_deploy/policy/low_latency
ln -sfnT "$MODEL_CACHE/planner/target_vel/V2" \
  gear_sonic_deploy/planner/target_vel/V2
```

The pinned revision matches the model hashes enforced by preflight. These links
also place generated TensorRT engines under the selected cache root. To use the
repository's default model locations instead, omit `--output-dir` and the
links; keep the pinned `--revision`.

## Network setup

1. Connect the PICO and workstation Wi-Fi to the same LAN.
2. In the XRoboToolkit PICO app, set the workstation Wi-Fi IP.
3. In **XRoboToolkit Hardened**, enable **Head + Controller**, **Send**, and
   **Full body**. The PICO client uses one exclusive mode selector:
   **Full body** emits the 24-joint Body stream and hardened tracker-health
   sideband, while **Object** emits raw Motion Tracker records. It cannot emit
   both pose streams at once; raw Object/Motion records are not a Full-body
   gate. The hardened controller sideband must also report each controller
   device valid, tracked, and carrying both position and rotation tracking.
4. Connect the G1 Ethernet link and assign the workstation a local
   `192.168.123.x/24` address, normally `192.168.123.99/24` or the address
   already used by the robot setup.
5. Bind Unitree DDS to the workstation's local interface, such as `eth0`.
   Never pass the robot's IP as the DDS interface.

XRoboToolkit uses TCP 63901 and UDP 29888. The teleop/deployment link uses TCP
5556 and feedback TCP 5557. Restrict any firewall rules to these ports and the
local trusted networks. Unitree DDS also needs UDP 7400-7401 and its negotiated
unicast range; with default CycloneDDS participant selection this can be
32768-65535. Scope that range to `192.168.123.0/24`, not all networks.

On WSL, run this once from an **elevated Windows PowerShell**:

```powershell
$RepoRoot = Read-Host 'Windows path to GR00T-WholeBodyControl'
Set-Location -LiteralPath $RepoRoot
.\install_scripts\configure_wsl_firewall.ps1 -PicoSubnet 192.168.1.0/24
```

Replace `192.168.1.0/24` with the actual trusted LAN CIDR shared by the PICO
and workstation. Supply its canonical network address, not a host address; for
example, use `192.168.1.0/24`, not `192.168.1.25/24`. Both PICO and Unitree
subnets must remain wholly inside RFC1918 private space. The script deliberately
has no PICO subnet default, adds only scoped Hyper-V rules, and does not alter
existing unrelated firewall rules.

## ZMQ control-session boundary

`--input-type zmq_manager` uses a fail-closed protocol-v2 ownership handshake:

1. Each native deployment process generates a new nonzero 128-bit receiver
   epoch and publishes it on the feedback `control_session` topic.
2. The PICO manager waits for that epoch, creates its own random publisher
   session, sends a claim, and waits for native acknowledgement. It publishes
   no actuation, planner, or pose frames before the claim succeeds.
3. Every command, planner frame, and pose frame carries both session tokens
   plus its monotonic replay sequence. The native process accepts only the
   permanently claimed publisher and never clears that claim on stop, mode
   change, disconnect, or tracking recovery.

After acknowledgement, the manager continuously verifies the advertised epoch
and bound owner. A native restart, ownership mismatch, or 500 ms feedback
heartbeat loss makes the manager fail closed; if active, it sends STOP before
exiting.

A STOP or native 500 ms deadman is terminal: the deployment stops and exits.
After a manager crash, operator STOP, tracking-loss STOP, or deadman, stop any
surviving manager process, restart the native deployment first, then start a
new manager. The new deployment publishes a new receiver epoch. A second
manager attaching to an already claimed deployment is rejected by design.

Standalone `--input-type zmq` retains its legacy wire format for compatibility.
It does **not** provide the `zmq_manager` protocol-v2 ownership boundary and is
not a substitute for this runbook.

Receiver epochs and publisher sessions prevent cross-process replay and
accidental multi-publisher overlap; they are not credentials. ZMQ traffic
remains plaintext and unauthenticated. Use only a trusted private LAN, keep the
firewall scoped to the documented hosts/subnets, and use a trusted VPN when
crossing any other network.

## Preflight

The preflight is read-only toward the robot. It validates exact dependencies,
model hashes, the operator's embodiment assertion, local interface selection,
live PICO body/controller frame progression, and the Unitree LowState stream.
It may start the local XRoboToolkit PC service and leaves that service running:

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
source .venv_teleop/bin/activate
source install_scripts/pico_g1_env.sh
python gear_sonic/scripts/pico_g1_preflight.py \
  --robot-dof 29 \
  --robot-interface eth0 \
  --probe-lowstate \
  --live-pico
```

To test PICO full-body tracking before connecting the G1, run the explicit
PICO-only diagnostic:

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
source .venv_teleop/bin/activate
source install_scripts/pico_g1_env.sh
python gear_sonic/scripts/pico_g1_preflight.py \
  --robot-dof 29 \
  --pico-only
```

This starts the local XRoboToolkit service and checks an advancing 24-joint
Body frame, both controllers, and the hardened sidebands in the same packet:
`BT_VALID`, tracking active, calibrated, successful connection/state/data API
calls, at least two connected and distinct trackers, all 24 calculated finite
body roles, and valid position-and-rotation tracking on both controllers. It
does not initialize a G1 interface, subscribe to LowState, publish commands,
or actuate the robot. The HEAD/body-source clock still detects a frozen cached
Body frame independently. The deliberate `PICO-only scope` warning means a
passing result is not G1 readiness.

For dependency and asset diagnostics when hardware is deliberately
disconnected, opt into offline mode explicitly:

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
source .venv_teleop/bin/activate
source install_scripts/pico_g1_env.sh
python gear_sonic/scripts/pico_g1_preflight.py \
  --robot-dof 29 \
  --offline
```

An offline result is not a real-robot readiness result and must never be used
to bypass the live PICO and LowState gates.

Do not continue while any check reports `FAIL`. In particular, zero LowState
samples means DDS is not ready even if ping or SSH works.
`--robot-dof 29` is an operator assertion, not robot telemetry: independently
confirm the 29-DoF body on the G1 label/order sheet and robot configuration.

## PICO tracker calibration

1. Pair both controllers and both Motion Trackers.
2. Secure trackers at the ankles; do not allow strap movement.
3. Complete PICO body-tracker calibration.
4. Wear the headset and face forward.
5. Stand in the documented neutral calibration pose.
6. Release all four face buttons before starting. The manager deliberately
   requires a release/re-press after startup or tracking recovery.

## Simulation gate

Use three terminals.

Terminal 1:

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py --no-enable-onscreen
```

Terminal 2:

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}/gear_sonic_deploy"
source ../install_scripts/pico_g1_env.sh
source scripts/setup_env.sh
./deploy.sh \
  --cp policy/low_latency/model \
  --obs-config policy/low_latency/observation_config.yaml \
  --input-type zmq_manager \
  sim
```

Terminal 3:

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --input-timeout 0.5 \
  --vis_vr3pt \
  --vis_smpl
```

The manager waits for the native `control_session` announcement and claims it
before streaming. A missing challenge, rejected claim, or already-bound
publisher is a hard failure; do not bypass it.
Keep native deployment and manager on the same host and leave
`--zmq_feedback_host localhost`. Source freshness uses their shared monotonic
clock; non-loopback feedback hosts are rejected.

Prove calibration, start/stop, POSE mode, planner mode, locomotion deadzone, and
tracking-loss stop in simulation. Removing the headset or stopping the PICO app
while active must stop the manager within 0.5 seconds. STOP and deadman tests
terminate that native deployment; restart Terminal 2, then Terminal 3, before
the next destructive test.

## Guarded real-robot run

This section is **only** for a robot positively identified as the released
29-DoF embodiment. Never run this `deploy.sh` command on the rev-1.0
`mode_machine == 4` true-23 body. That robot must use the causal V12 promoted
shadow and separately authorized gantry-stage path above.

Before every real run:

- Stop Rocco record/replay, custom arm publishers, old relays, simulators, and
  every other process that can publish robot commands.
- Confirm one owner only: the SONIC deployment process.
- Confirm G1 is physically stable, gantry slack is correct, and the clear zone
  is empty.
- Keep a second operator at the C++ terminal with the `O` emergency stop.
- Re-run preflight and require fresh LowState.

Start deployment first:

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}/gear_sonic_deploy"
source ../install_scripts/pico_g1_env.sh
source scripts/setup_env.sh
./deploy.sh \
  --cp policy/low_latency/model \
  --obs-config policy/low_latency/observation_config.yaml \
  --input-type zmq_manager \
  eth0
```

Wait for `Init done`. Then start the PICO manager:

For a real interface, `deploy.sh` requires typing the full word `yes`; Enter
alone cancels. Its no-argument default is simulation.

```bash
cd "${GR00T_WBC_ROOT:-$HOME/GR00T-WholeBodyControl}"
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --input-timeout 0.5
```

The manager must report a successful control-session claim before any start
gesture is accepted. If it reports that another publisher owns the deployment,
stop all publisher processes and restart the native deployment; do not reuse
that receiver epoch.

While holding the neutral calibration pose, press **A+B+X+Y** to engage. Press
**A+X** for POSE mode. Press **A+B+X+Y** again to stop. The second operator can
press **O** in the deployment terminal at any time. Either stop is terminal.
For another session, restart the native deployment first and then the manager.

The manager continuously requires advancing Body and both-controller frames,
plus fresh hardened health proving both controllers remain position-and-
rotation tracked, all 24 body roles are calculated, `BT_VALID`, calibration, successful API
results, and at least two distinct connected trackers. It sends a repeated stop
burst on operator stop, stale/non-finite/implausibly discontinuous tracking,
unhealthy tracker telemetry, Ctrl+C, or an unexpected manager exception.

A manager crash without a STOP is still bounded by the independent native
deadman. Once it fires, treat the deployment as terminated and restart both
processes in deployment-then-manager order.
