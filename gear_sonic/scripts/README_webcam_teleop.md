# Webcam Teleoperation: GEM → SONIC → Unitree G1

Real-time whole-body teleoperation of the Unitree G1 from a single RGB webcam.
A human motion estimation model (GEM/GENMO) converts the camera stream into SMPL
poses, a Python bridge converts them into the SONIC streamed-motion protocol, and
the C++ SONIC deploy stack drives the robot (MuJoCo simulation or the real G1).

```
Webcam ──► GEM (GENMO, ONNX/GPU) ──► ZMQ :5558 (pickle) ──► Bridge (CPU, 50 Hz)
                                                             │
                                        ZMQ :5556 (SONIC protocol v3, topic "pose")
                                                             ▼
                              C++ g1_deploy_onnx_ref ──► G1 (MuJoCo sim or real robot)
```

Components (all in this repo):

| Component | Script | Python env |
|---|---|---|
| Pose estimation | `gear_sonic/scripts/gem_webcam_zmq_publisher.py` | `external_dependencies/GENMO/.venv` |
| Bridge | `gear_sonic/scripts/webcam_smpl_streamer.py` | `.venv_teleop` |
| Simulator | `gear_sonic/scripts/run_sim_loop.py` | `.venv_sim` |
| Controller | `gear_sonic_deploy` (`just run g1_deploy_onnx_ref …`) | — (C++/TensorRT) |

---

## 1. One-time setup

### 1.1 Python environments

```bash
# MuJoCo simulator env (.venv_sim)
bash install_scripts/install_mujoco_sim.sh

# Teleop/bridge env (.venv_teleop)
bash install_scripts/install_pico.sh

# GEM/GENMO env (external_dependencies/GENMO/.venv) — clones GENMO, installs
# torch cu124 + onnxruntime-gpu with uv (Python 3.10)
bash install_scripts/install_gem_webcam.sh
```

### 1.2 SMPL-X body model (manual, required for `--render`)

Register at <https://smpl-x.is.tue.mpg.de/>, download `SMPLX_NEUTRAL.npz` and place it at:

```
external_dependencies/GENMO/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```

### 1.3 C++ deploy stack

Build `gear_sonic_deploy` following its own README (CMake + TensorRT). The first
launch converts the ONNX policy/encoder/planner to TensorRT engines (takes a few
minutes, cached afterwards).

### 1.4 GEM ONNX models

Downloaded automatically from Hugging Face (`nvidia/GEM-X`, ~8.7 GB) on the first
run of the publisher. GEM assets resolve relative to the CWD — always run from the
repo root (an `inputs` symlink at the root points into the GENMO checkout).

### 1.5 Multicast for DDS (after every reboot)

The CycloneDDS transport between the controller and the simulator uses multicast
on the loopback interface:

```bash
sudo ip link set lo multicast on
sudo ip route add 224.0.0.0/4 dev lo
```

---

## 2. Running — Web UI (one button)

```bash
cd ~/Documentos/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/teleop_webui/server.py          # http://localhost:8080
# add --host 0.0.0.0 to open the UI from a tablet/phone on the same network
```

Open <http://localhost:8080> and press **START**. The orchestrator runs the
pre-flight checks (venvs, webcam, ports, multicast route), launches the four
components in the correct order with readiness detection, and presses `]` on
the C++ controller for you. Wait for all four component LEDs to turn green
(the C++ controller takes ~30 s loading TensorRT; the GEM warmup needs your
**full body visible** in the camera preview). The MuJoCo window is passive —
you never type or click in it; everything is controlled from the page:

| UI button | Effect |
|---|---|
| START | Pre-flight + launch everything + start the policy (`]`) |
| Stop | Graceful shutdown of the whole stack |
| E-STOP | Immediately halts control (sends `o` to the controller) |
| Start policy ( ] ) | Re-send `]` (e.g. after a fall + re-init) |
| Toggle stream (Enter) | If the robot stands but does not imitate you |
| Arm (s) / Pause (p) | Resume / pause imitation (robot idles standing) |
| Next pilot (n) | Pause + reset the session timer for the next person |
| Cleanup | Kill leftover processes from a previous run (frees ports) |

The **Upper body mode** checkbox (default on) restricts imitation to arms and
torso — legs and root orientation stay neutral. Recommended for public demos.

### Phone camera (remote)

Select **Phone camera (remote)** in the camera dropdown before START. The
orchestrator additionally launches MediaMTX (expected at
`~/.local/opt/mediamtx/mediamtx`, config `gear_sonic/config/mediamtx_teleop.yml`)
and GEM reads `rtsp://127.0.0.1:8554/cam` instead of the webcam. While starting,
the page shows the publish URL — on the phone (same network or tailnet), open
`https://<PC-IP>:8889/cam/publish`, accept the self-signed certificate warning
and tap **Publish** (rear camera recommended). GEM waits for the stream, then
proceeds normally. Alternative publishers into the same path:

- SRT app (e.g. Larix Broadcaster): `srt://<PC-IP>:8890?streamid=publish:cam`
- ffmpeg push: `ffmpeg -re -i src -c:v libx264 -tune zerolatency -f rtsp rtsp://<PC-IP>:8554/cam`

If the phone drops, GEM auto-reconnects and the bridge safety layer holds the
robot in a safe pose meanwhile. Note: the GEM frame reader uses a raw ffmpeg
subprocess (~36 ms ingest→decode measured on localhost); OpenCV's VideoCapture
adds ~0.5 s of undroppable buffering on live RTSP and must not be used here.

The manual procedure below remains the reference and fallback.

---

## 3. Running manually (4 terminals)

> Alternatively, `bash gear_sonic/scripts/launch_webcam_teleop.sh sim` starts all
> four panes in one tmux session. The manual procedure below is the reference.

Start the terminals **in this order** and wait for each ready marker.

### Terminal 1 — MuJoCo simulator

```bash
cd ~/Documentos/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

✅ Ready when the MuJoCo window opens with the G1. This window is passive — you
never need to click or type in it.

### Terminal 2 — C++ SONIC controller

```bash
cd ~/Documentos/GR00T-WholeBodyControl/gear_sonic_deploy
just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq_manager --output-type all \
  --zmq-host localhost --disable-crc-check
```

✅ Ready when it prints `Init Done` (20–30 s loading TensorRT engines).
**Keep this terminal focused later — all control keys are typed here.**

### Terminal 3 — GEM + webcam

```bash
cd ~/Documentos/GR00T-WholeBodyControl
source external_dependencies/GENMO/.venv/bin/activate
python gear_sonic/scripts/gem_webcam_zmq_publisher.py --no_imgfeat --render --render_mode opencv
```

✅ Ready when it prints `[ZMQ] Publishing SMPL frames on tcp://*:5558` and the
OpenCV window opens.

**Stand up in front of the camera with your FULL body visible (head to feet)**
and hold still during the warmup (`Warmup …/120`, ~4 s). The skeleton overlay
should track you in the OpenCV window. If only your torso is visible, the
estimated pose is garbage and the robot will fall.

Useful flags: `--camera_id N` to pick another webcam; `--no_imgfeat` skips the
HMR2 image features (faster, recommended on RTX 30xx).

### Terminal 4 — Bridge

```bash
cd ~/Documentos/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/webcam_smpl_streamer.py --auto_start
```

✅ Ready when it prints `Buffer full — streaming pose + START command sent`
followed by periodic lines like:

```
[Bridge] mode=TRACKING out_fps=49.8 gem_frames=1234 step=5678 rejected=0
```

`out_fps` must be ~49.8 and `gem_frames` must keep increasing. If `gem_frames`
stays at 0, GEM is not detecting you — go back in front of the camera.

Without `--auto_start`, press `s` in this terminal to begin streaming.
Other bridge keys: `p` = pause (safety idle; press `s` to resume), `n` = next
pilot (pause + reset session timer), `o`/`q` = stop and exit.

Useful flags: `--upper_body` (arms/torso only — legs and root orientation are
forced neutral; recommended for public demos; enables a 90 s session timer,
configurable with `--session_timeout`), `--record_dir DIR` / `--no_record`.

A safety state machine sits between GEM and the robot: per-tick clamps on
joint/pose/quaternion deltas, sanity + jump rejection of GEM samples, and a
watchdog. If tracking is lost (pilot leaves the frame, occlusion, GEM crash)
the bridge holds the last pose, blends to a neutral stance, and switches the
robot to planner idle; when tracking returns it resumes with a smooth ramp.
All of this is automatic — no key presses needed.

### Final step — start the policy (critical!)

Click on **Terminal 2 (C++)** and press:

| Key | Effect |
|---|---|
| **`]`** | **Starts the control policy.** Without this the robot is limp/frozen and falls. Press once. |
| `Enter` | Toggles between the pre-loaded reference motions and the ZMQ stream. If the robot stands but does not imitate you after `]`, press `Enter` once. |
| `o` / `O` | **EMERGENCY STOP** — halts control immediately. |
| `i` | Re-initialize (after a fall: reset the sim, then `i` + `]`). |
| `f` | Print motor temperatures (real robot). |

**Summary:** T1 → T2 (wait `Init Done`) → T3 (wait warmup, full body in frame) →
T4 (wait `mode=TRACKING`) → focus T2 → press `]` → (if not imitating) press `Enter`.
The robot in MuJoCo should now mirror your movements. Start with slow arm motions.

---

## 3. Real robot

Use `bash gear_sonic/scripts/launch_webcam_teleop.sh real` or replace the network
interface (`lo`) and drop `--disable-crc-check` in the Terminal 2 command according
to your robot network setup. **Always validate in simulation first.** Keep a hand
on the emergency stop (`O` in Terminal 2) at all times.

---

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Robot collapses immediately | Policy not started — press `]` in Terminal 2. |
| Robot stands but ignores you | Stream toggle off — press `Enter` in Terminal 2; check `gem_frames` increasing in Terminal 4. |
| `Address already in use` (port 5557) on C++ start | A previous `g1_deploy_onnx_ref` is still dying. `pkill -9 -f g1_deploy_onnx_ref`, wait 1 s, relaunch. |
| Bridge shows `gem_frames=0` | GEM does not see you (out of frame / bad light / warmup not finished). |
| Sim does not react at all | Multicast route missing (section 1.5) — redo after every reboot. |
| `ModelProto does not have a graph` | Truncated ONNX download — delete the file and re-download. |
| Erratic robot motion when you leave the frame | Known limitation: GEM extrapolates garbage when the body is not fully visible. Stay fully in frame; stop with `o` if needed. |
| Check that everything is alive | `pgrep -fa "run_sim_loop\|g1_deploy_onnx_ref\|webcam_smpl_streamer\|gem_webcam"` → must list 4 processes. |

---

## 5. Protocol notes (for developers)

- Bridge → C++ wire format: `[topic][1280-byte JSON header ljust '\0'][LE binary]`,
  protocol v3, topic `pose`. Required fields per message (N=5 frame chunks):
  `smpl_pose (N,21,3) f32`, `smpl_joints (N,24,3) f32`, `body_quat_w (N,4) f32`,
  `joint_pos (N,29) f64`, `joint_vel (N,29) f64`, `frame_index (N,) i64`.
- Packing helpers: `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`.
- C++ parser: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp`.
- Canonical Python reference: `gear_sonic/scripts/pico_manager_thread_server.py`
  (class `SmplStream`).
- GEM → bridge channel (port 5558) is plain pickled dicts (`send_pyobj`) with
  `body_pose (63,)`, `global_orient (3,)` (y-up world), `transl`, timestamps.
- The bridge upsamples the GEM rate (~30 Hz) to 50 Hz by linear interpolation with
  one source-interval delay, then ships 5-frame chunks.
