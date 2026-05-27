# VR_3PT Real-Robot Data Collection Progress

Date: 2026-05-27

## Current Status

- Real-robot deploy runs on PC2 from `gear_sonic_deploy/target/release/g1_deploy_onnx_ref`.
- PICO controller-only `VR_3PT` teleoperation runs from the workstation with `openxr_unitree` pose conventions.
- Real robot does not have Dex3/Inspire hands; deploy must use `--disable-dex3-hands`.
- During calibration diagnostics, use ramp-only mode by disabling the deploy VR_3PT safety filter and manager-side entry/watchdog gates. Re-enable safeguards before non-diagnostic real runs.
- The VLA data exporter can now run without a physical image server using `--use-dummy-camera`. This writes black `observation.images.ego_view` frames and is intended only for format and signal-path validation.

## Validated Dataset Smoke Test

Dataset:

```text
outputs_vla/g1_vr3pt_dummy_camera_smoke_001
```

Validation results:

- `codebase_version`: `v2.1`
- `total_episodes`: `1`
- `total_frames`: `898`
- `fps`: `50`
- data file: `data/chunk-000/episode_000000.parquet`
- video file: `videos/chunk-000/observation.images.ego_view/episode_000000.mp4`
- `action.motion_token` present with shape `[64]`
- `teleop.vr_3pt_position` present with shape `[9]`
- `teleop.vr_3pt_orientation` present with shape `[18]`
- `teleop.stream_mode` recorded as `5` for VR_3PT

Note: this smoke-test episode was discarded via `Left Grip + B`, so `discarded_episode_indices` contains `[0]`. The dataset format and signal path are valid, but this episode should not be used as training data.

## Working Smoke-Test Exporter Command

Run on workstation while PC2 deploy and PICO manager are running:

```bash
cd ~/work/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "teleoperate the robot to reach forward" \
  --dataset-name g1_vr3pt_dummy_camera_smoke_001 \
  --root-output-dir outputs_vla \
  --use-dummy-camera \
  --sonic-zmq-host localhost \
  --sonic-zmq-port 5556 \
  --state-zmq-host 192.168.0.223 \
  --state-zmq-port 5557
```

Recording controls:

- `Left Grip + A`: start / stop-save episode.
- `Left Grip + B`: discard current episode.

Hold `Left Grip` first, then tap `A` or `B`; the manager publishes rising-edge toggle flags on the `manager_state` ZMQ topic.

## Camera Status

- PC2 sees the Intel RealSense D435i at the USB layer and through `pyrealsense2`.
- The local composed camera RealSense path was stabilized with color-only streaming, explicit serial selection, and configurable RealSense FPS.
- The D435i connection is still physically/runtime unstable on PC2: observed `set_xu` timeout/protocol errors and `Frame didn't arrive within 5000` after startup.
- Consider switching image transport to Unitree `teleimager` and adding an exporter adapter if it proves more stable for RealSense on PC2.

## Next Steps

1. Record one non-discarded dummy-camera episode and confirm `discarded_episode_indices` is empty.
2. Decide between continuing with the repo's composed camera RealSense path or integrating `teleimager`.
3. After camera streaming is stable, record a short real image episode and validate the LeRobot v2.1 dataset again.
4. Re-enable real-robot safety safeguards before collecting non-diagnostic teleop episodes.
