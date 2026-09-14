# Isaac Teleop Setup (CloudXR / DeviceIO, in-process)

This page documents the Isaac Teleop / CloudXR bring-up for **G1 with a Thor backpack** that drives `GR00T-WholeBodyControl` directly from the headset. Using `pico_manager_thread_server.py --input-source isaac-teleop`, the CloudXR runtime is hosted **in-process** via the `isaacteleop[cloudxr]` Python package.

```{admonition} Scope
:class: important
Real-robot deployment is supported only on **G1 + Thor backpack**. Sim2Sim (MuJoCo) can run on both Thor and x86_64 workstations.
```

## Prerequisites

1. **Completed the [Quick Start](../getting_started/quickstart.md)** — you can run the Sim2Sim loop (includes [installing the deployment](../getting_started/installation_deploy.md) and [downloading model checkpoints](../getting_started/download_models.md)).
2. **Completed the [VR Teleop Setup](../getting_started/vr_teleop_setup.md)** — `.venv_teleop` is ready and `install_pico.sh` has been run (on Thor for real-robot deployment; on your workstation for Sim2Sim).

This page is a condensed, repo-specific version of the upstream [Isaac Teleop](https://nvidia.github.io/IsaacTeleop/) docs:

- [Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html)
- [`isaacteleop[cloudxr]` Python API](https://nvidia.github.io/IsaacTeleop/main/)

## Step 1: Prepare the Thor Host

Install the prerequisites on Thor (skip any you've already done for the rest of the deploy):

```bash
sudo apt install -y build-essential curl git-lfs
git lfs install
```

```{note}
The remainder of this step (max power mode, thermal check) uses Thor/Jetson-specific tools. If you are running Sim2Sim (MuJoCo) on an x86_64 workstation instead of real G1 hardware, skip ahead to Step 2.
```

For Thor performance, enable max power mode before teleoperation:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Optional thermal / over-current check:

```bash
cat /sys/class/hwmon/hwmon*/oc*_event_cnt
```

## Step 2: Confirm `isaacteleop[cloudxr]` Installed

```{note}
This step is a checkpoint, not a new action. `install_pico.sh`, run during [VR Teleop Setup](../getting_started/vr_teleop_setup.md) Step 3, already installs `isaacteleop[cloudxr]`. If that step is complete, skip to Step 3 below; otherwise, complete it before continuing.
```

For reference, `install_pico.sh` installs the package from the public NVIDIA index:

```bash
# Already wired into install_pico.sh; shown here for reference
uv pip install 'isaacteleop[cloudxr]~=1.4.0' --prerelease=allow \
    --extra-index-url https://pypi.nvidia.com
```

It also seeds `~/cloudxr.env` with `NV_DEVICE_PROFILE=Quest3` (override by editing the file). `CloudXRLauncher` reads this on startup.

## Step 3: Start the C++ Deployment

From `gear_sonic_deploy/`:

```bash
export TensorRT_ROOT=$HOME/TensorRT   # only if not already in ~/.bashrc
./docker/run-ros2-dev.sh

# inside the container (setup_env.sh is sourced automatically):
just build                            # first run only

# run one of these:
./deploy.sh --input-type zmq_manager real   # real robot
./deploy.sh --input-type zmq_manager sim    # Sim2Sim (MuJoCo)
# Wait until you see "Init done"
```

```{note}
For Sim2Sim, first start the MuJoCo simulator on the host as shown in the [Quick Start](../getting_started/quickstart.md) Sim2Sim section, or the deployment will have nothing to control.
```

## Step 4: Launch the Teleop Streamer

From the **repo root**:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --input-source isaac-teleop

# If running offboard with a display, add visualization:
#   --vis_vr3pt --vis_smpl
```

On startup the streamer brings up the in-process CloudXR runtime and logs `Isaac Teleop session initialized.`, then repeats `waiting for Isaac Teleop body data (connect the headset to CloudXR)...` until you connect the client in Step 5.

## Step 5: Connect the XR Client

Connecting the client starts the body-data stream the streamer is waiting for.

- Open the [Isaac Teleop Web Client](https://nvidia.github.io/IsaacTeleop/client/) in the headset browser
- Enter the IP address of the host running the streamer
- Accept the self-signed certificate at `https://<host-ip>:48322`
- On Thor, change **Video Codec** from the default **AV1** to **H.264** or **H.265 (HEVC)**.
- Return to the client page and click **Connect**

Once connected, stand in the [calibration pose](vr_wholebody_teleop.md#calibration-pose) and press **A+B+X+Y** on the PICO controllers to start the policy; the first press also runs the startup calibration and enters PLANNER (locomotion) mode. Then press **A+X** to switch to POSE mode for whole-body teleop, where your motion maps directly to the robot. See [Complete PICO Controls](vr_wholebody_teleop.md#pico-controls) for the other modes and the emergency stop.

For quick validation, the same client URL can also be opened in a desktop browser.

If you prefer to run the WebXR client from source instead of the hosted client, follow the CloudXR/WebXR build instructions linked from the [Isaac Teleop Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html).

## Step 6: Start Camera Visualization

Stream cameras to the headset via upstream IsaacTeleop's `camera_viz.sh`. If you don't have the IsaacTeleop repo yet, clone it first:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/IsaacTeleop.git
```

Then create the environment for the camera visualization streamer:

```bash
cd IsaacTeleop
examples/camera_viz/camera_viz.sh setup
source examples/camera_viz/.venv/bin/activate
cd examples/camera_viz
```
### Optional: Camera Preview in a Window
If you have video preview available such as on Thor or desktop simulation, it might be best to test your camera first with the "window" mode:
```bash
./camera_viz.sh run configs/YOUR_CAMERA.yaml --mode window
```

Choosing the correct yaml configuration camera for your setup is critical. The v4l2.yaml configuration is default but other configs are available:
```bash
# run just one of these that best matches your camera
./camera_viz.sh run configs/v4l2.yaml --mode window
./camera_viz.sh run configs/oakd.yaml --mode window
./camera_viz.sh run configs/zed.yaml --mode window
./camera_viz.sh run configs/realsense.yaml --mode window
```
If you do not see correct output for your camera, you may need to modify the closest yaml file to your setup or create your own in the configs folder. Consider the following:
- Having more than one camera connected will modify what channel the video should be served from.
- First, try the command "lsusb" to see if your camera is connected.
- Next, use "ls -1 /dev/video* to list all video device modes.
- If you are using a v4l2 setup (Video4Linux2), these cammands will help to debug what is possible:
  - To see the association of channels to cameras: "v4l2-ctl --list-devices"
  - For each node, see what it captures and in what pixel formats: "v4l2-ctl --device=/dev/video0 --list-formats"
  - To see full details on formats and available resolutions:  "v4l2-ctl --device=/dev/video0 --list-formats-ext"
Once you have this information, make sure the settings in your yaml config match.

### XR Camera Streaming Command

Once you are certain you have valid yaml configured, shut down any preview windows and run instead with "--mode xr", which will stream frames to Isaac Teleop:

```bash
./camera_viz.sh run configs/[YOUR_CAMERA].yaml --mode xr
```
This should also be consistent with the [instructions found at IsaacTeleop](https://nvidia.github.io/IsaacTeleop/main/references/camera_streaming.html).

## Troubleshooting

### `RuntimeError: Failed to get OpenXR system: -35`

In this setup, that error usually means the XR client is not connected yet. Re-check:

- The headset / web client is fully connected to `https://<host-ip>:48322`
- The CloudXR runtime subprocess is still alive (look for the `Isaac Teleop session initialized.` log)
- `~/cloudxr.env` exists and has the right `NV_DEVICE_PROFILE` for your headset

### Client connects but the video encoder fails to initialize

On Thor, this usually means the client's **Video Codec** is still set to its default, **AV1**. Switch the client to **H.264** or **H.265 (HEVC)** and reconnect (see Step 5).

### `gear_sonic_deploy` build error

If `just build` fails, rebuild from scratch inside the container (Step 3):

```bash
rm -rf build
just build
```

### `isaacteleop` import error

Re-run `install_pico.sh` to reinstall `isaacteleop[cloudxr]~=1.4.0` into `.venv_teleop`. If `pypi.nvidia.com` is unreachable, check your network and the `--extra-index-url` flag.

### Body data not arriving

The streamer logs `[IsaacTeleopReader] No DeviceIO data for 5.0s, flagging disconnect` if the headset stops feeding body data. Confirm:

1. The headset is still connected to CloudXR (Step 5).
2. The Pico body trackers are paired and calibrated (see [VR Teleop Setup → Motion Tracker Setup](../getting_started/vr_teleop_setup.md)).
3. The first time the schema runs, watch for `[IsaacTeleopReader] Unrecognised body_data schema: type=...` — if you see it, the upstream `FullBodyTracker.get_body_pose().data` shape changed and `_body_data_to_24x7()` in `gear_sonic/utils/teleop/input_readers.py` needs an extra branch for the new layout.
