# PICO XR24 to True23 SOMA replay

This path converts hardened PICO Ultra BodyTracking into complete G1 reference
terms without opening a Unitree, DDS, or ZMQ channel:

`XR24 BodyTracking -> bracketed 50 Hz poses -> pinned NVIDIA SOMA -> MJ29 -> canonical IL29 q/dq -> locked-True23 G1 FK -> 267 encoder terms`

It does not zero-fill lower-body output or repeat one pose as future data.  The
normal True23 profile needs 47 complete 50 Hz frames.  Its ten command samples
are therefore measured delayed references spanning 0.9 seconds, with one extra
velocity-proof frame.  Minimum semantic delay is 0.94 seconds.

## 1. Capture actual hardened XR24 data

The XRoboToolkit extension is built for Python 3.10.  Replace the APK hash with
the SHA-256 of the installed hardened PICO client:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl

PYTHONPATH="$PWD:/root/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64" \
python3 gear_sonic/scripts/capture_g1_23dof_pico_raw.py \
  --output /root/g1_true23_live/pico_xr24_capture.json \
  --pico-client-apk-sha256 ACTUAL_64_HEX_SHA256 \
  --service-binary /opt/apps/roboticsservice/RoboticsServiceProcess \
  --frames 70
```

Capture accepts only advancing, calibrated, healthy 24-role BodyTracking with
two unique connected ankle bands.  Raw MotionTracking pairing is rejected.

## 2. Replay through pinned SOMA

The robot quaternion must be measured at the semantic playback frame, in Unitree
`w,x,y,z` order.  Identity below is valid only for an offline neutral fixture:

```bash
PY="$HOME/.venvs/g1_true23_soma/bin/python"

PYTHONPATH="$PWD" "$PY" gear_sonic/scripts/replay_g1_23dof_xr24_soma.py \
  --capture /root/g1_true23_live/pico_xr24_capture.json \
  --soma-source-root /root/.cache/g1_true23_soma/source \
  --profile true23_step5_0p1s \
  --robot-anchor-quat-wxyz W X Y Z \
  --output-dir /root/g1_true23_live/pico_xr24_soma_replay
```

`encoder_reference_terms.json` then has exact
`g1_true23_two_source_encoder_terms` schema v2 and dimensions
`240 + 9 + 12 + 6 = 267`.

The same two ABI stages can run from one command.  This captures one real
window and replays it; it is not a continuous publisher:

```bash
PYTHONPATH="$PWD" "$PY" gear_sonic/scripts/replay_g1_23dof_xr24_soma.py \
  --capture-live \
  --pico-client-apk-sha256 ACTUAL_64_HEX_SHA256 \
  --capture-python /usr/bin/python3 \
  --xrt-binding-dir /root/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
  --service-binary /opt/apps/roboticsservice/RoboticsServiceProcess \
  --frames 70 \
  --robot-anchor-quat-wxyz W X Y Z \
  --output-dir /root/g1_true23_live/pico_xr24_soma_live_window
```

## Current hard gate

Pinned SOMA is an offline batch pipeline.  Warm local proof solved 64 frames in
about 1.64 seconds (about 39 solver frames/s); full warm CLI took 18.8 seconds.
Continuous 50 Hz output and rolling-window boundary continuity are not proven.
Synthetic replay validates mechanics only.  Deployment approval remains false
until a real hardened PICO capture proves neutral axes, left/right direction,
feet/lower-body motion, timestamps, and output limits.  Do not feed these files
to motors before that live replay review passes.
