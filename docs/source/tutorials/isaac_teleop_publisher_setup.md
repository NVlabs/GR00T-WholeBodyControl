# Isaac Teleop Setup (CloudXR / DeviceIO, in-process)

This page documents the Isaac Teleop / CloudXR bring-up for **G1 with a Thor backpack** that drives `GR00T-WholeBodyControl` directly from the headset — no separate publisher container, no host `~/.cloudxr` sharing. The CloudXR runtime is hosted **in-process** by `pico_manager_thread_server.py --input-source isaac-teleop` via the `isaacteleop[cloudxr]` Python package.

```{admonition} Scope
:class: important
This workflow is currently documented and supported only for **G1 + Thor backpack**.

As of 07/08/2026, CloudXR 6.2.0 does not support mobile Blackwell GPUs, including GeForce RTX 50-series Laptop GPUs. Support for desktop Blackwell GPUs does not extend to the corresponding laptop variants. This restriction does not apply to the supported Jetson Thor path documented here.
```

## Prerequisites

1. **Completed the [Quick Start](../getting_started/quickstart.md)** — you can run the sim2sim loop (includes [installing the deployment](../getting_started/installation_deploy.md) and [downloading model checkpoints](../getting_started/download_models.md)).
2. **Completed the [VR Teleop Setup](../getting_started/vr_teleop_setup.md)** — `.venv_teleop` is ready and `install_pico.sh` has been run on Thor.

This page is a condensed, repo-specific version of the upstream [Isaac Teleop](https://nvidia.github.io/IsaacTeleop/) docs:

- [Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html)
- [`isaacteleop[cloudxr]` Python API](https://nvidia.github.io/IsaacTeleop/main/)

## Step 1: Prepare the Thor Host

Install the prerequisites on Thor (skip any you've already done for the rest of the deploy):

```bash
sudo apt install -y build-essential curl git-lfs

ARCH=$(uname -m)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh"
bash "Miniforge3-Linux-${ARCH}.sh" -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init
source ~/.bashrc

git lfs install
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

## Step 2: Install `isaacteleop[cloudxr]`

`install_pico.sh` (run from the [VR Teleop Setup](../getting_started/vr_teleop_setup.md) Step 3) installs the `isaacteleop[cloudxr]` package into `.venv_teleop` from the public NVIDIA index:

```bash
# Already wired into install_pico.sh; shown here for reference
uv pip install 'isaacteleop[cloudxr]~=1.3.0' --prerelease=allow \
    --extra-index-url https://pypi.nvidia.com
```

It also seeds `~/cloudxr.env` with `NV_DEVICE_PROFILE=Quest3` (override by editing the file). `CloudXRLauncher` reads this on startup.

## Step 3: Connect the XR Client

The streamer launches the CloudXR runtime as a subprocess of the Python loop the first time you start it (Step 4). Until you connect the headset, `IsaacTeleopClient.start_streaming()` will retry quietly with `"no XR session yet"`.

- Open the [Isaac Teleop Web Client](https://nvidia.github.io/IsaacTeleop/client/) in the headset browser
- Enter the IP address of the Thor host
- Accept the self-signed certificate at `https://<thor-ip>:48322`
- Return to the client page and click **Connect**

For quick validation, the same client URL can also be opened in a desktop browser.

If you prefer to run the WebXR client from source instead of the hosted client, follow the CloudXR/WebXR build instructions linked from the [Isaac Teleop Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html).

```{important}
The streamer will fail to acquire OpenXR until the XR client is connected. Either connect the headset first, or start the streamer and watch for the `Isaac Teleop session initialized.` log line once you do.
```

## Step 4: Start the C++ Deployment

From `gear_sonic_deploy/`:

```bash
export TensorRT_ROOT=$HOME/TensorRT   # only if not already in ~/.bashrc
./docker/run-ros2-dev.sh

# inside the container (setup_env.sh is sourced automatically):
just build                            # first run only
./deploy.sh --input-type zmq_manager real
# Wait until you see "Init done"
```

## Step 5: Launch the Teleop Streamer

From the **repo root**:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --input-source isaac-teleop

# If running offboard with a display, add visualization:
#   --vis_vr3pt --vis_smpl
```

Watch the streamer logs for `Isaac Teleop session initialized.` — that means CloudXR + DeviceIO are both up. The streamer then prints `Manager controls: A+X=toggle mode, A+B+X+Y=start/stop policy` once the headset is connected and body data starts flowing.

For controls and calibration, see [VR Whole-Body Teleop](vr_wholebody_teleop.md#pico-controls).

## Step 6: Start Camera Streaming

Stream cameras to the headset via upstream IsaacTeleop's `camera_streamer.sh`. If you don't have the IsaacTeleop repo yet, clone it first:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/IsaacTeleop.git
```

Then build and launch the camera streamer:

```bash
cd IsaacTeleop
cd examples/camera_streamer

./camera_streamer.sh build
./camera_streamer.sh list-cameras
```

Run with local cameras streaming to the XR headset:

```bash
source scripts/setup_cloudxr_env.sh
./camera_streamer.sh run --source local --mode xr

# Optionally specify a camera configuration:
#   --config config/single_camera.yaml
#   --config config/multi_camera.yaml
```

The `--input-source isaac-teleop` flag also works with [Data Collection](data_collection.md).

## Troubleshooting

### Streaming connection times out on a Blackwell laptop GPU

The CloudXR client may time out after about 30 seconds with error `0xC0F22219`, while `cxr_streamsdk` reports that the GPU device ID is not white-listed. If the server uses a GeForce RTX 50-series Laptop GPU, this is a CloudXR 6.2.0 compatibility limitation: mobile Blackwell GPUs are not supported. Changing network settings or selecting a different codec will not enable the encoder. Use the supported G1 + Thor host described in this guide.

Triage CloudXR compatibility reports first to the [NVIDIA/IsaacTeleop issue tracker](https://github.com/NVIDIA/IsaacTeleop/issues), because `isaacteleop[cloudxr]` packages and launches the runtime. Include the `isaacteleop` and CloudXR versions, full GPU model and PCI device ID, and the CloudXR logs. GPU allow-list and encoder enablement belong to the CloudXR runtime and cannot be fixed in `GR00T-WholeBodyControl`.

### `RuntimeError: Failed to get OpenXR system: -35`

In this setup, that error usually means the XR client is not connected yet. Re-check:

- The headset / web client is fully connected to `https://<thor-ip>:48322`
- The CloudXR runtime subprocess is still alive (look for the `Isaac Teleop session initialized.` log)
- `~/cloudxr.env` exists and has the right `NV_DEVICE_PROFILE` for your headset

### `gear_sonic_deploy` build error

If `just build` fails, rebuild from scratch inside the container (Step 4):

```bash
rm -rf build
just build
```

### `isaacteleop` import error

Re-run `install_pico.sh` to reinstall `isaacteleop[cloudxr]~=1.3.0` into `.venv_teleop`. If `pypi.nvidia.com` is unreachable, check your network and the `--extra-index-url` flag.

### Body data not arriving

The streamer logs `[IsaacTeleopReader] No DeviceIO data for 5.0s, flagging disconnect` if the headset stops feeding body data. Confirm:

1. The headset is still connected to CloudXR (Step 3).
2. The Pico body trackers are paired and calibrated (see [VR Teleop Setup → Motion Tracker Setup](../getting_started/vr_teleop_setup.md)).
3. The first time the schema runs, watch for `[IsaacTeleopReader] Unrecognised body_data schema: type=...` — if you see it, the upstream `FullBodyTrackerPico.get_body_pose().data` shape changed and `_body_data_to_24x7()` in `gear_sonic/utils/teleop/input_readers.py` needs an extra branch for the new layout.
