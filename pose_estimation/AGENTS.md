# Unitree G1 FoundationPose handoff

## Documentation

`README.md` is the user-facing guide for setup, execution, TF publication, and
validation. Keep it synchronized with command-line arguments, endpoints, camera
settings, and the runtime pipeline. This file contains additional operational
handoff details for agents working on the code.

This directory was copied from `/home/athena/Camera/pose_estimation/g1` and
renamed from `g1/` to `pose_estimation/`. Do not recreate or refer to the old
`g1/` path.

## Unitree access

- Robot: `unitree@192.168.123.164`
- Robot OS: Ubuntu 20.04 on Jetson/aarch64
- Remote camera server: `/home/unitree/image_server/image_server.py`
- SSH authentication is interactive; do not save the robot password in this
  repository.

The robot-side image server must be running:

```bash
ssh unitree@192.168.123.164
# Select ROS Foxy (1) if the login asks.
cd /home/unitree/image_server
/usr/bin/python3 image_server.py
```

The remote server was minimally changed to align depth to color:

```python
self.align = rs.align(rs.stream.color)
frames = self.align.process(frames)
```

Do not assume alignment merely from matching image dimensions. Verify these
lines in the active remote source and restart the remote server after any source
change.

## Head camera and stream contract

- Head camera: Intel RealSense D435I
- Serial: `344522070363`
- Resolution: `640x480`
- Frame rate: `30 FPS`
- ZMQ endpoint: `tcp://192.168.123.164:5555`
- The current stream is raw, not JPEG-compressed.
- Each ZMQ multipart message contains:
  1. 20-byte metadata: `struct.pack("!IIIII", color_h, color_w, color_c, depth_h, depth_w)`
  2. BGR `uint8` color bytes, normally `480x640x3`
  3. aligned raw `uint16` depth bytes, normally `480x640`

Head-camera color intrinsics at `640x480`:

```text
fx = 605.231384
fy = 604.627441
cx = 324.900238
cy = 246.474838
depth_scale = 0.001 metres per raw unit
```

These intrinsics are specific to this D435I, serial number, and resolution.
Re-query them if the camera or resolution changes. The depth scale is currently
hardcoded in `detector.py`; re-query it if the depth camera changes.

Before FoundationPose, the host converts:

```python
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
depth = depth_raw.astype(np.float32) * 0.001
```

FoundationPose receives RGB plus aligned `float32` depth in metres. It must not
receive the raw `uint16` array directly.

## Local code

- `detector.py`: live G1 ZMQ receiver, GroundingDINO detection, FastSAM mask,
  FoundationPose registration, and continuous tracking.
- `estimater.py`: FoundationPose implementation used by `detector.py`.
- `publish_object_tf.py`: receives 4x4 poses from `detector.py` over ZMQ and
  publishes `d435_link -> detected_object` to ROS 2 TF by default.
- `camera_scripts/receive_video.py`: lightweight raw RGB-D stream viewer.
- `g1_head/`: saved RGB-D samples, masks, poses, and visualizations.

`g1_head/` and Python bytecode caches are generated local artifacts and are
ignored by Git.

`detector.py` currently tracks one object. The first frame runs detection,
segmentation, and `register(iteration=10)`. Later frames use
`track_one(iteration=1)`; the previous value was `2`. Press `r` to discard the
tracked pose and register again.

The live pose pipeline is:

```text
D435I stream -> detector.py -> ZMQ :5561 -> publish_object_tf.py -> ROS 2 TF
```

The complete robot TF tree can be published separately with
`planner_wbc/scripts/collect_robot_transform.py`.

Before using the object TF for grasping, validate known positions in front,
left/right, and above/below the camera. FoundationPose uses optical camera
coordinates; keep the TODO in `publish_object_tf.py` until the mapping to
`d435_link` has been quantitatively confirmed.

The copied code still depends on the complete FoundationPose repository at
`/home/athena/Camera/pose_estimation` for models, weights, FastSAM,
GroundingDINO, `Utils`, and other modules. `POSE_ESTIMATION_ROOT` near the top of
`detector.py` is the single dependency-root setting to update if that repository
moves.

Known tested model:

```text
/home/athena/Camera/pose_estimation/models/trash_truck/textured.obj
```

## Run

Follow `README.md` for the complete workflow. For detector-only operation,
first start the robot image server. Then, on the host:

```bash
cd /home/athena/GR00T-WholeBodyControl
conda activate foundation_pose
python pose_estimation/detector.py
```

Controls: `r` re-registers; `q` quits.

The copied detector was tested from this GR00T repository. Observed performance
was about 3 seconds for initial registration and about 70-80 ms per tracking
frame (roughly 13-14 FPS). This is estimator throughput, while the incoming
camera stream is about 30 FPS.
