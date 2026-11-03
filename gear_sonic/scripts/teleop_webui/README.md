# Teleop Web UI — one-button realtime teleoperation of the G1

Browser control panel that launches and supervises the whole realtime
teleoperation stack: camera → GEM (pose estimation) → bridge → C++ SONIC
controller → G1 (MuJoCo sim or real robot). Everything is driven from a single
web page — no terminal juggling, no typing into the MuJoCo window.

```
Camera (webcam or phone) ─► GEM (GENMO, ONNX/GPU) ─► ZMQ :5558 ─► Bridge (50 Hz)
                                                                     │
                                                     ZMQ :5556 (SONIC protocol v3)
                                                                     ▼
                                      C++ g1_deploy_onnx_ref ─► G1 (sim or real)
```

For the manual 4-terminal procedure and protocol details see
[../README_webcam_teleop.md](../README_webcam_teleop.md).

---

## 1. One-time setup

### 1.1 Python environments and C++ stack

```bash
bash install_scripts/install_mujoco_sim.sh     # .venv_sim
bash install_scripts/install_pico.sh           # .venv_teleop (bridge + this UI)
bash install_scripts/install_gem_webcam.sh     # external_dependencies/GENMO/.venv
```

Then build `gear_sonic_deploy` (CMake + TensorRT, see its README) and place
`SMPLX_NEUTRAL.npz` at
`external_dependencies/GENMO/inputs/checkpoints/body_models/smplx/` (manual
download from <https://smpl-x.is.tue.mpg.de/>). GEM ONNX models (~8.7 GB)
download automatically on first run.

### 1.2 MediaMTX (only for the phone camera)

```bash
mkdir -p ~/.local/opt/mediamtx && cd ~/.local/opt/mediamtx
curl -L https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_amd64.tar.gz | tar xz
```

The orchestrator starts/stops it automatically with the repo config
[gear_sonic/config/mediamtx_teleop.yml](../../config/mediamtx_teleop.yml) and
generates the self-signed TLS certificate on first use. To publish from outside
the LAN, install [Tailscale](https://tailscale.com) on the PC and the phone and
log both into the same tailnet.

### 1.3 Multicast route (after every reboot)

```bash
sudo ip link set lo multicast on
sudo ip route add 224.0.0.0/4 dev lo
```

Without this the controller and the simulator cannot talk (DDS) — the preflight
check will catch it.

---

## 2. Running

```bash
cd <repo-root>
.venv_teleop/bin/python gear_sonic/scripts/teleop_webui/server.py
# add --host 0.0.0.0 to open the UI from another device on the network
```

Open <http://localhost:8080>, pick the **camera source** and press **START**.
The orchestrator runs preflight checks (venvs, camera/ports, multicast route),
launches every component in order with readiness detection and starts the
policy for you. Wait for all component LEDs to turn green — the C++ controller
takes ~30 s loading TensorRT engines on the first run.

### 2.1 Local webcam

Leave the source on **Local webcam**. When GEM is up, stand with your **full
body visible** (head to feet) and hold still through the warmup (~4 s). The
robot in MuJoCo then mirrors your movements.

### 2.2 Phone camera (remote) — validated over LAN and Tailscale

1. Select **Phone camera (remote)** and press **START**. The page shows two
   publish URLs: LAN (`https://<lan-ip>:8889/cam/publish`) and Tailscale
   (`https://<ts-ip>:8889/cam/publish` — works from anywhere).
2. On the phone browser, open one of the URLs and accept the self-signed
   certificate warning.
3. **Set Video codec to H264** (hardware-encoded on phones; VP9 is
   software-encoded and cannot be decoded in realtime on the PC), bitrate
   ~2000 kbps, audio off, then tap **Publish**.
4. Keep the phone screen on and your full body in frame; wait for the warmup.

GEM waits for the stream ("waiting for publisher …") and auto-reconnects if the
phone drops; the bridge safety layer holds the robot in a safe pose meanwhile.

### 2.3 UI buttons

| Button | Effect |
|---|---|
| START | Preflight + launch everything + start the policy (`]`) |
| Stop | Graceful shutdown of the whole stack |
| **E-STOP** | Immediately halts control (`o` to the controller) |
| Start policy ( ] ) | Re-send `]` (e.g. after a fall + re-init) |
| Toggle stream (Enter) | If the robot stands but does not imitate you |
| Arm (s) / Pause (p) | Resume / pause imitation (robot idles standing) |
| Next pilot (n) | Pause + reset the session timer for the next person |
| Cleanup | Kill leftovers from a previous run (frees ports) |

**Upper body mode** (default on) restricts imitation to arms/torso — recommended
for public demos.

### 2.4 Real robot

Same UI with mode **real** — but **always validate in simulation first**, keep a
hand near E-STOP, and start with slow arm motions.

---

## 3. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Preflight fails on "multicast (lo)" | Redo section 1.3 (needed after every reboot). |
| Robot fell in sim | Press **Stop**, then **START** again. |
| Robot stands but ignores you | Press **Toggle stream**; check `gem_frames` increasing in the bridge panel. |
| GEM stuck at "waiting for publisher" | Nothing published yet — open the publish URL on the phone and tap Publish. |
| MediaMTX logs `write queue is full` | Decoder can't keep up: make sure the phone publishes **H264**, not VP9. |
| Video laggy / PC overloaded | Publish H264 at ≤2000 kbps. The stack already caps GEM CPU threads and uses a low-latency ffmpeg reader (`-vsync 0`). |
| Ports busy after a crash | Press **Cleanup** in the UI. |
| MediaMTX `address already in use` on START | A previous instance survived — **Cleanup**, or `pkill -f mediamtx`. |

### Latency notes (why it is built this way)

- `cv2.VideoCapture` adds ~535 ms of **undrainable** buffering on live RTSP and
  ignores `OPENCV_FFMPEG_CAPTURE_OPTIONS`. The GEM reader therefore uses a raw
  ffmpeg subprocess (~36 ms ingest→decode measured on localhost) that keeps only
  the newest frame. Never use `cv2.VideoCapture` for live URLs.
- `-vsync 0` on the reader is mandatory: without it ffmpeg's CFR rawvideo output
  duplicates WebRTC frames ~10x (GEM sees 300+ fps from a 30 fps stream) and
  saturates the CPU.
- Benchmark tool: [gear_sonic/scripts/bench_stream_latency.py](../bench_stream_latency.py).
