# Live Camera Teleoperation with GEM-X

Teleoperate the Unitree G1 from a **single RGB camera** — no motion-capture suit, no VR headset, no body trackers. [GEM-X](https://github.com/NVlabs/GEM-X) recovers your full-body 3D motion from the webcam image, this example converts that motion to SMPL and streams it over SONIC's ZMQ pose protocol (`--input-type zmq`), and the SONIC policy tracks it while keeping the robot balanced.

<figure style="margin: 1em 0;">
<video width="100%" controls autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/live_camera_teleop/teleop_real_robot.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Live webcam teleoperation on a real G1 — the operator (left) is tracked by a single camera, with no suit, gloves, or headset. The robot is on a safety gantry, as recommended for a first hardware session.</figcaption>
</figure>

```{admonition} Prerequisites
:class: note
Complete the [Quick Start](../getting_started/quickstart.md) so the sim2sim loop runs. This tutorial drives the same interface described in [Streaming Motion Tracking](zmq.md) — read that page first if you have not used `--input-type zmq` before, since the keyboard controls and stream protocol are shared.
```

```{admonition} GEM-X is an external dependency
:class: important
GEM-X is **not** bundled with this repository and is not installed by any of the SONIC install scripts. You install it separately and point this example at it with `--gemx-root` (or `$GEMX_ROOT`). GEM-X is Apache-2.0 licensed.
```

---

## How It Works

```text
  Webcam ──RGB──►  GEM-X rolling window  ──SOMA──►  soma_to_smpl.py
  (1 camera)       YOLOX → ViTPose (2D)   77 joints  SOMA → SMPL, 24 joints
                   → diffusion denoiser              root-local, Z-up
                   → SOMA decoder                            │
                                                             │  ZMQ Protocol v3
                                                             │  topic "pose"
                                                             │  tcp://<gemx-host>:5556
                                                             ▼
                                              SONIC C++ deployment
                                              smpl encoder → WBC policy
                                                             │
                                                             ▼
                                              Unitree G1 (sim or real)
```

| Stage | What happens | Output |
|---|---|---|
| Capture | OpenCV reads one frame from the camera | RGB image |
| Detect + 2D pose | YOLOX detection, then ViTPose keypoints (ONNX) | 77 2D keypoints |
| Lift to 3D | GEM-X diffusion denoiser runs over a rolling window of buffered frames | SOMA latent |
| Decode | GEM-X decoder produces body parameters, and the gravity-aligned root orientation is fused in | SOMA body params (77 joints) |
| Convert | `soma_to_smpl.py` maps SOMA → SMPL, rotates Y-up → Z-up, removes the root rotation | `smpl_joints` (24×3), `body_quat` |
| Stream | ZMQ `PUB` socket sends a Protocol v3 message per frame | wire message on topic `pose` |
| Track | SONIC's `smpl` encoder (mode 2) turns SMPL into latent commands the policy tracks | robot joint targets |

### Why stream SMPL instead of joint angles

SONIC ships a learned `smpl` encoder that was trained on human SMPL motion, so the human-to-robot mapping already lives **inside the policy**. Streaming SMPL hands that mapping to the encoder and keeps the online loop free of any per-frame retargeting or inverse-kinematics solve.

The alternative — converting SOMA to G1 joint angles offline and streaming Protocol v1 — requires a retargeter in the loop, which is the expensive part. The conversion in `soma_to_smpl.py` mirrors `gear_sonic/scripts/pico_manager_thread_server.py:process_smpl_joints` exactly, so the streamed data is in-distribution for the same encoder the PICO path uses.

### Files

Everything lives in `gear_sonic/examples/live_camera_teleop/`:

| File | Role |
|---|---|
| `webcam_stream.py` | Live loop: capture → GEM-X → convert → publish. Also the camera test tool. |
| `soma_to_smpl.py` | SOMA → SMPL conversion and the Protocol v3 ZMQ publisher. Imported, not run directly. |
| `soma_pt_to_sonic_v3.py` | Replays a saved GEM-X result (`hpe_results.pt`) — camera-free verification. |
| `README.md` | Short reference version of this tutorial. |

---

## Prerequisites

1. **SONIC deployment built and runnable** — see [Installation (Deployment)](../getting_started/installation_deploy.md) and [Quick Start](../getting_started/quickstart.md), which also covers [downloading the released checkpoint](../getting_started/download_models.md) whose `smpl` encoder this path uses.
2. **GEM-X installed** — cloned, its virtual environment built, and its checkpoint available. It is a separate repository that no SONIC install script sets up for you; Step 1 below covers it.
3. **A camera** — any UVC USB webcam or laptop camera. A video file works as a stand-in for everything except the live feel.

---

## Step 1 — Install GEM-X and Set Up

Follow the [GEM-X installation guide](https://github.com/NVlabs/GEM-X/blob/main/docs/INSTALL.md). The short version:

```bash
git clone --recursive https://github.com/NVlabs/GEM-X.git
cd GEM-X

pip install uv && uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -e third_party/soma && (cd third_party/soma && git lfs pull)
bash scripts/install_env.sh
```

Then two things this example needs on top of a stock GEM-X install:

```bash
# 1. SOMA body-model assets must be reachable at inputs/soma_assets
mkdir -p inputs
ln -sfn "$PWD/third_party/soma/assets" inputs/soma_assets

# 2. Extra Python deps for the ZMQ bridge
uv pip install pyzmq scipy
```

```{admonition} The soma_assets link is not optional
:class: warning
`webcam_stream.py` constructs `SomaLayer(data_root="<gemx-root>/inputs/soma_assets")`. Without that path the run fails when the converter starts up, *after* the camera and the denoiser have already initialized — which looks like a late, confusing crash. Create the link before the first live run.

Use an **absolute** target as shown. A relative target would be resolved against `inputs/`, not the repo root, and silently produce a dangling link.
```

GEM-X downloads its checkpoint from Hugging Face on first use, so the first run is slower than later ones.

Finally, point the example at both repositories. The scripts live in the SONIC repo but import `gem` from GEM-X, so they run in **GEM-X's virtual environment** and locate SONIC by path:

```bash
cd /path/to/GEM-X
source .venv/bin/activate

export GEMX_ROOT=$PWD
export SONIC_ROOT=/path/to/GR00T-WholeBodyControl
```

```{admonition} Why SONIC_ROOT is required
:class: note
The conversion in `soma_to_smpl.py` reuses SONIC's own rotation helpers (`gear_sonic.isaac_utils.rotations`) so the SMPL convention matches the deployment bit for bit. Python puts the *script's* directory on `sys.path`, not the repo root, so `gear_sonic` is not importable from the GEM-X environment unless you set `$SONIC_ROOT` (or pass `--sonic-root`). Those helpers only need `torch` and `numpy` — you do **not** need to install `gear_sonic` into the GEM-X environment.
```

---

## Step 2 — Verify the Input Camera

Do this before involving the robot. `webcam_stream.py` doubles as the camera test tool: `--kp-only` runs detection and 2D keypoints but skips the 3D denoiser, so it starts fast and needs no checkpoint.

### Find the device

```bash
ls /dev/video*
v4l2-ctl --list-devices                       # sudo apt install v4l-utils
v4l2-ctl -d /dev/video0 --list-formats-ext    # modes the camera actually supports
```

The index in `/dev/videoN` is what you pass to `--source` (`/dev/video0` → `--source 0`). Many USB cameras expose several nodes for one physical device; the lowest-numbered one is usually the capture node.

### Preview the tracking

```bash
# Live overlay window, press q to quit
python "$SONIC_ROOT/gear_sonic/examples/live_camera_teleop/webcam_stream.py" \
    --source 0 --kp-only --show
```

You should see green keypoints tracking your body and a steady frame rate on the status line. Fix any missing or jumping keypoints here — the controller can only track what the estimator sees.

### Request a capture mode

A camera's default mode is often low-resolution or low-fps, which caps the teleop rate. Request one of the modes reported by `--list-formats-ext`:

```bash
... --source 0 --resolution 1280x720 --cap-fps 30
```

Cameras silently ignore unsupported settings, so the script logs the mode it actually negotiated. Check that line rather than assuming the request applied.

(framing)=
### How to Position the Camera and the Operator

Monocular estimation only knows what it can see, and the lower body is what the controller balances on:

- Keep the **full body in frame, including the feet.** Cropped legs make the lower body unreliable.
- Stand roughly **2–3 m back**, camera near torso height, lens roughly level.
- **Even and front facing** light. Avoid strong backlight.
- **One person in frame** — the largest detection is the one tracked.
- **Face the same direction as the robot.** Orientation is streamed in the robot's own frame, so operator and robot should be aligned.

---

## Step 3 — Dry Run Without a Camera

`soma_pt_to_sonic_v3.py` replays a saved GEM-X result, which isolates the conversion and the wire format from anything camera-related. Produce an `hpe_results.pt` by running any GEM-X demo on a video, then:

```bash
# Convert frame 0 and print the result — no ZMQ, no robot
python "$SONIC_ROOT/gear_sonic/examples/live_camera_teleop/soma_pt_to_sonic_v3.py" \
    --pt /path/to/hpe_results.pt --dry-run
```

Once that prints a sane 24-joint skeleton, replay it as a stream against the simulator (start Terminals 1 and 2 from the next section first):

```bash
python "$SONIC_ROOT/gear_sonic/examples/live_camera_teleop/soma_pt_to_sonic_v3.py" \
    --pt /path/to/hpe_results.pt --fps 30 --loop
```

This is the fastest way to confirm the SONIC side is wired up correctly, and it is reproducible — the same input produces the same motion every time.

---

## Step 4 — Teleop in Simulation

Run **three terminals**.

### Terminal 1 — MuJoCo simulator

From the **repo root**:

```bash
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

### Terminal 2 — C++ deployment

From `gear_sonic_deploy/`:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq --zmq-host localhost sim
# Wait until you see "Init done"
```

```{admonition} Do not add --zmq-port or --zmq-topic here
:class: warning
`deploy.sh` forwards only `--input-type` and `--zmq-host`. The port and topic stay at the deployment defaults, `5556` and `pose` — exactly what this example publishes on, so no extra flags are needed.

Unrecognized flags are not rejected; `deploy.sh` treats any unknown argument as the positional interface argument, so passing `--zmq-port 5556` can quietly override your `sim` / `real` selection depending on argument order.
```

(remote-streamer)=
```{admonition} Running GEM-X on another machine
:class: note
The example **binds** its ZMQ `PUB` socket (`tcp://*:5556`) and the deployment **connects** to it. So `--zmq-host` is the IP of the machine running `webcam_stream.py`, not the robot. Keep it on `localhost` when both run on the same workstation; pass the workstation's IP when the deployment runs onboard the robot.
```

### Terminal 3 — GEM-X Webcam Bridge

From the **GEM-X repo**, with its environment active and both roots exported (Step 1):

```bash
python "$SONIC_ROOT/gear_sonic/examples/live_camera_teleop/webcam_stream.py" \
    --source 0 --stream-sonic --window 30 --smooth 0.8
```

Expect output along these lines:

```text
[webcam] backends: vitpose=..., denoiser=...
[bridge] Using SONIC's pack_pose_message (exact wire format).
[webcam] streaming SMPL v3 to SONIC on tcp://*:5556
[webcam] frame 42 |  12.3 fps | soma=yes
```

Two lines are worth reading carefully:

- **`Using SONIC's pack_pose_message`** confirms `$SONIC_ROOT` resolved. If you instead see `gear_sonic not importable ... using built-in fallback packer`, the wire format is still byte-identical, but the conversion needs those same helpers — so fix `$SONIC_ROOT` rather than relying on the fallback.
- **`soma=yes`** means 3D output is flowing. The first couple of frames report `warmup/kp-only` while the rolling window fills.

### Your first session

With all three terminals running, drive the deployment from Terminal 2:

1. Press **`]`** to start the control system.
2. In the MuJoCo window, press **`9`** to drop the robot to the ground.
3. Stand in front of the camera in a **relaxed, upright pose** with your whole body in frame. Check Terminal 3 shows `soma=yes`.
4. Back in Terminal 2, press **`ENTER`** to enable ZMQ streaming. The terminal prints `ZMQ STREAMING MODE: ENABLED` and the robot begins tracking you.
5. Start with **slow arm motions only.** Confirm the robot mirrors you in the expected direction before moving your legs or torso.
6. Press **`ENTER`** again to return to reference-motion mode, and **`O`** to stop and exit.

```{admonition} Align yourself before enabling the stream
:class: danger
The robot snaps to whatever pose is streaming the moment you press **`ENTER`**. A large mismatch between the robot's current pose and yours produces sudden, aggressive motion — the same hazard described for POSE mode in the [PICO VR tutorial](vr_wholebody_teleop.md). Stand relaxed and upright, and verify tracking is stable, before enabling the stream.
```

---

## Step 5 — Teleop on the Real Robot

```{admonition} Safety Warning
:class: danger
Only proceed once you can run a smooth session in simulation and are comfortable with the emergency stop. **Terminate `run_sim_loop.py` first** — a simulator and a real robot running at once will conflict.

Bring the robot up on a gantry for the first real session, and keep a safety operator on **`O`** and the hardware E-stop. Review the [Whole-body Teleoperation Guide](../user_guide/teleoperation.md) before you start.
```

```{admonition} Expect the Robot to Walk Forward
:class: warning
Reaching your arms out in front of you, or making large, fast arm motions, shifts the streamed reference enough that the policy steps forward to keep its balance. On hardware that means the robot leaves the spot it started on, without any locomotion being commanded — so keep the space in front of it clear, do not stand directly in its path, and start with small arm motions close to your body. Work on this is in progress.
```

The real-robot workflow uses **two terminals** (no MuJoCo).

### Terminal 1 — C++ deployment

From `gear_sonic_deploy/`:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh

# 'real' auto-detects the robot network interface (192.168.123.x).
# If GEM-X runs on this workstation, localhost is correct:
./deploy.sh --input-type zmq --zmq-host localhost real

# If the deployment runs onboard and GEM-X is on a workstation,
# point it at the workstation:
#   ./deploy.sh --input-type zmq --zmq-host <workstation-IP> real

# Wait until you see "Init done"
```

### Terminal 2 — GEM-X Webcam Bridge

Identical to the simulation case:

```bash
cd /path/to/GEM-X && source .venv/bin/activate
export GEMX_ROOT=$PWD SONIC_ROOT=/path/to/GR00T-WholeBodyControl

python "$SONIC_ROOT/gear_sonic/examples/live_camera_teleop/webcam_stream.py" \
    --source 0 --stream-sonic --window 30 --smooth 0.8
```

Then follow the same sequence: **`]`** to start, **`ENTER`** to enable streaming once tracking is stable, **`O`** to stop.

---

## Controls Reference

All keys are pressed in the **C++ deployment terminal** and are shared with [Streaming Motion Tracking](zmq.md).

| Key | Action |
|---|---|
| **`]`** | Start the control system |
| **`ENTER`** | Toggle ZMQ streaming mode (camera tracking) on / off |
| **`Q`** / **`E`** | Adjust heading left / right (±0.1 rad per press) |
| **`I`** | Reinitialize base quaternion and reset heading to zero |
| **`T`** / **`N`** / **`P`** / **`R`** | Reference-motion playback (streaming mode off) |
| **`O`** | **Emergency stop** — halt control and exit |

---

## Command-Line Reference

### `webcam_stream.py`

| Option | Default | Description |
|---|---|---|
| `--source` | `0` | Camera index (`ls /dev/video*`) or a video file path |
| `--stream-sonic` | off | Publish the Protocol v3 SMPL stream. Without it, nothing is sent to SONIC. |
| `--window` | `120` | Rolling-window length in frames. Dominates the frame rate — see [Tuning](#tuning). |
| `--smooth` | `0.75` | Temporal smoothing weight on the streamed reference (`0` = off) |
| `--port` | `5556` | ZMQ `PUB` port to bind. Matches the deployment default; change both or neither. |
| `--resolution` | camera default | Requested capture resolution, e.g. `1280x720` |
| `--cap-fps` | camera default | Requested capture frame rate |
| `--kp-only` | off | 2D keypoints only — skips the denoiser. Camera checks. |
| `--show` / `--save` | off | Live preview window / write an overlay mp4 |
| `--max-frames` | `0` | Stop after N frames (`0` = run until interrupted) |
| `--gemx-root` | `$GEMX_ROOT` | GEM-X repo root |
| `--sonic-root` | `$SONIC_ROOT` | SONIC repo root (see Step 1) |

### `soma_pt_to_sonic_v3.py`

| Option | Default | Description |
|---|---|---|
| `--pt` | *required* | Path to a saved GEM-X `hpe_results.pt` |
| `--fps` | `30` | Replay rate |
| `--loop` | off | Replay continuously |
| `--dry-run` | off | Convert frame 0, print it, and exit — no ZMQ |
| `--smooth` | `0.0` | Temporal smoothing (off by default, unlike the live path) |
| `--port` / `--max-frames` / `--gemx-root` / `--sonic-root` | — | As above |

---

(tuning)=
## Tuning: Frame Rate vs. Stability

The denoiser runs over the whole buffered window each frame, so cost grows with `--window` until the buffer is full. That makes `--window` the main throughput knob.

| Symptom | Try |
|---|---|
| Frame rate too low | Lower `--window` (e.g. `30` → `20`); request a lighter capture mode with `--resolution` / `--cap-fps` |
| Robot jitters or takes small stepping motions | Raise `--smooth` toward `0.85`; improve lighting and [framing](#framing) |
| Robot feels laggy or "behind" you | Lower `--smooth`, lower `--window`, and move more deliberately |
| Legs or feet look wrong | Almost always framing — get the feet fully in frame and step further back |
| Robot drifts in heading | Press **`I`** to reinitialize the base quaternion, then nudge with **`Q`** / **`E`** |

Start at `--window 30 --smooth 0.8` and adjust one knob at a time.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not open video source: 0` | Wrong index (`ls /dev/video*`), or the user is not in the `video` group (`sudo usermod -aG video $USER`, then re-login) |
| Opens, but frames fail or hang | Another process holds the device (browser tab, earlier run). Check with `sudo fuser /dev/video0` |
| `ModuleNotFoundError: No module named 'gear_sonic'` | `$SONIC_ROOT` is unset or wrong — see Step 1 |
| `ModuleNotFoundError: No module named 'gem'` | Running in the wrong environment. Activate GEM-X's `.venv`, not a SONIC one. |
| Crash when the converter starts, referencing SOMA assets | The `inputs/soma_assets` link is missing or dangling — see Step 1 |
| `KeyError` about `identity_coeffs` / `scale_params` | The SOMA decode is incomplete; those parameters have no safe zero default, so the bridge refuses to stream a collapsed skeleton rather than sending garbage to the controller |
| Robot never moves after **`ENTER`** | No messages arriving. Check Terminal 3 prints `soma=yes`, that `--zmq-host` points at the machine running GEM-X, and that port `5556` is reachable |
| Very low frame rate | Camera negotiated a slow mode, or `--window` is too large — see [Tuning](#tuning) |
| `--show` fails on a headless box | No display; use `--save` instead |
| No keypoints detected | Body not fully in frame, too dark, or too far away |

---

## Limitations

- **No commanded root translation.** The stream carries root-local pose plus heading, not an explicit root translation, so walking the robot around by walking yourself is not supported. Combine with the [kinematic planner](keyboard.md) for locomotion and use the camera for upper-body motion.
- **Wrists and fingers are not tracked.** `smpl_pose` and the 6 wrist joints are streamed as zeros. GEM-X estimates hands, so wiring them through is a natural extension.
- **The robot walks forward to keep its balance.** Some arm motions — reaching out in front of you, or large, fast swings — shift the streamed reference far enough that the policy takes steps to stay upright. The robot then drifts from where it started even though no locomotion was commanded. Work on this is in progress.
- **Monocular lower body is the weakest signal.** Feet and depth are inherently less certain from one camera than from trackers, and that is exactly what the controller balances on. Framing matters more here than any parameter.
- **Single subject, static camera.** The largest detection is tracked, and the loop assumes a fixed camera.

---

## Next Steps

- [Streaming Motion Tracking](zmq.md) — the underlying ZMQ interface and full protocol reference
- [PICO VR Whole-body Teleop](vr_wholebody_teleop.md) — tracker-based teleoperation, for comparison
- [Whole-body Teleoperation Guide](../user_guide/teleoperation.md) — movement patterns and operating practices
- [Data Collection for VLA](data_collection.md) — recording teleoperation sessions as training data
