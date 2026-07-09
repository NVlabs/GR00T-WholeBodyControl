# Live Camera Teleoperation with GEM-X

Drive the Unitree G1 from a **live webcam** by feeding
[GEM-X](https://github.com/NVlabs/GEM-X) 3D human pose estimation into GEAR-SONIC.
No motion-capture suit or VR trackers; just a camera.

GEM-X estimates full-body human motion (the SOMA body model); this example bridges
that motion into SONIC's existing ZMQ streaming interface, and SONIC's policy
tracks it while keeping balance.

> GEM-X is an **optional external dependency** and is **not** bundled here. Install
> it separately and point these scripts at it with `--gemx-root` (or `$GEMX_ROOT`).

## How it works

| Script(s) | SONIC protocol | SONIC encoder |
|---|---|---|
| `webcam_stream.py` + `soma_to_smpl.py` | v3 (SMPL) | `smpl` (mode 2) |
| `soma_pt_to_sonic_v3.py` (offline verify) | v3 (SMPL) | `smpl` (mode 2) |

GEM-X runs in a rolling-window loop producing SOMA per frame; `soma_to_smpl.py`
converts SOMA (77 joints) -> SMPL (24 joints, root-local, gravity-aligned) and
streams Protocol v3 so SONIC's learned `smpl` encoder does the human->robot
mapping (no offline retargeter in the loop).

## Prerequisites

1. **SONIC deployment** built and runnable (see the repo Quick Start), with the
   released policy that has the `smpl` encoder (`policy/release/`).
2. **GEM-X** cloned and installed: https://github.com/NVlabs/GEM-X
   (its checkpoints/assets download from HuggingFace on first run).
3. Python deps in the environment you run these scripts in: `pyzmq`, `scipy`,
   `numpy`, `opencv-python`, and a CUDA-matched `onnxruntime-gpu`.

## Usage

Run the SONIC deploy in ZMQ mode first (sim shown; use `real` for hardware):

```bash
# terminal 1 (repo root): MuJoCo sim
python gear_sonic/scripts/run_sim_loop.py
# terminal 2 (gear_sonic_deploy/): deploy, ZMQ input
bash deploy.sh --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose --zmq-conflate sim
```

### Live webcam (Protocol v3 / smpl encoder)
```bash
export GEMX_ROOT=/path/to/GEM-X
# camera sanity check (no robot):
python gear_sonic/examples/live_camera_teleop/webcam_stream.py --source 0 \
    --kp-only --save webcam_test.mp4 --max-frames 60
# live teleop:
python gear_sonic/examples/live_camera_teleop/webcam_stream.py --source 0 \
    --stream-sonic --window 30 --smooth 0.8
```

### Offline SMPL verify (no camera)
```bash
python gear_sonic/examples/live_camera_teleop/soma_pt_to_sonic_v3.py \
    --gemx-root /path/to/GEM-X --pt /path/to/hpe_results.pt --fps 30 --loop
```

On the deploy side: press `]` to start, drop the robot (`9` in MuJoCo), then
`ENTER` to enable ZMQ streaming. Use `O` for emergency stop. On real hardware,
keep a safety operator on the E-stop.

## Key options
- `--window`: rolling-window length; smaller = higher fps.
- `--no-imgfeat`: skip GEM-X's SAM-3D-Body image features for speed (lower
  quality legs/depth).
- `--smooth`: temporal smoothing of the streamed reference (0 = off,
  0.6-0.85 steadier).
- `--source`: camera index (`ls /dev/video*`) or a video file (stand-in).

## Notes / limitations
- The live path streams root-local SMPL pose + heading; it does not command an
  explicit root translation, so controlled forward navigation is limited (best
  combined with the kinematic planner for the lower body).
- `smpl_pose` and wrist joints are streamed as zeros (the `smpl` encoder consumes
  `smpl_joints` + anchor + wrists); wrist/finger fidelity is future work.
- Monocular lower-body/foot estimation is the weakest signal; image features
  (`--no-imgfeat` off) improve grounding at the cost of frame rate.
