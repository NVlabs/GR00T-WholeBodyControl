# Simple Custom README — BrainCo Teleoperation

## tmux launchers

Two separate scripts are used:

```text
gear_sonic/scripts/launch_brainco_robot_tmux.sh  # run on the robot
gear_sonic/scripts/launch_brainco_host_tmux.sh   # run on the host computer
```

Install `tmux` on both machines:

```bash
sudo apt install tmux
```

Start the robot stack:

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh
tmux attach -t brainco_robot
```

Start the host stack:

```bash
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection
ROBOT_HOST=192.168.50.132 \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh
tmux attach -t brainco_host
```

Default profile for both launchers:

- Robot `192.168.50.132` sends RGB from the `ego_view` RealSense at `1280x720`
  and the `head` USB camera at `/dev/video6` and `1600x896`; both cameras run
  at `15 FPS`.
- RealSense depth streams are disabled on the robot and are not sent over the network.
- The host records both RGB cameras, the local `/dev/video4` as
  `external-view-camera`, and G1 telemetry.
- The viewer shows `ego_view`, `head`, and `external-view-camera`.
- Raw depth and depth preview video are not saved.

If the external camera uses a different path, set it with
`EXTERNAL_VIEW_CAMERA_DEVICE`. An empty value disables the external camera.
Check the path on the host with `v4l2-ctl --list-devices`.

`--replace` stops the session with the same name and creates it again:


# Robot
```bash

cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh --replace
tmux attach -t brainco_robot
```

# Host

```bash
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection/
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace
tmux attach -t brainco_host
```

To switch tmux windows, press `Ctrl+b` and then the window number, or press
`Ctrl+b` and then `n` / `p`. To detach without stopping the session, press
`Ctrl+b` and then `d`.

```bash
tmux attach -t brainco_robot
tmux attach -t brainco_host
tmux kill-session -t brainco_robot
tmux kill-session -t brainco_host
```

---

# Robot

## 1. Setup

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection

bash install_scripts/install_camera_server.sh
bash install_scripts/install_pico.sh
```

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/activate
pip install pyrealsense2
```

```bash
rs-enumerate-devices
v4l2-ctl --list-devices
```

## 2. Robot tmux launcher

```bash
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh
```

Launcher variables:

| Variable | Default | Values / purpose |
|---|---:|---|
| `TMUX_SESSION` | `brainco_robot` | Name of the tmux session. |
| `CAMERA_MODE` | `realsense-usb` | `two-realsense`, `realsense-usb`, `head-realsense`, or `none`. |
| `EGO_VIEW_DEVICE_ID` | `243422071979` | Serial number of the ego RealSense camera. |
| `HEAD_DEVICE_ID` | `135122071874` | Serial number of the head RealSense camera. |
| `USB_HEAD_DEVICE_ID` | `/dev/video6` | Index or path of the USB head camera. |
| `CAMERA_FPS` | `15` | Ego camera frame rate and base frame rate of the camera server. |
| `REALSENSE_WIDTH`, `REALSENSE_HEIGHT` | `1280`, `720` | RGB resolution of the ego RealSense camera. |
| `HEAD_CAMERA_WIDTH`, `HEAD_CAMERA_HEIGHT` | `1600`, `896` | Separate RGB resolution for the USB or RealSense head camera. |
| `HEAD_CAMERA_FPS` | `15` | Separate frame rate for the USB or RealSense head camera. |
| `HEAD_CAMERA_QUALITY` | `80` | JPEG quality of the head stream, from `1` to `100`. |
| `HEAD_CAMERA_FOURCC` | `MJPG` | Capture format for the USB head camera. Ignored for RealSense. |
| `REALSENSE_DEPTH` | `0` | `0` does not start the depth sensor or send depth; `1` enables it. |
| `HEAD_REALSENSE_DEPTH_WIDTH`, `HEAD_REALSENSE_DEPTH_HEIGHT` | `640`, `480` | Depth resolution for `head-realsense`. |
| `PICO_INTERFACE` | `wlxfc23cd997021` | Network interface for BrainCo DDS. |
| `PICO_PORT` | `5556` | Pico manager port. |
| `TELEMETRY_PORT` | `5560` | G1 telemetry publisher port. |
| `TELEMETRY_HZ` | `50` | Target telemetry frequency. |
| `BRAINCO_CONTAINER` | `g1-brainco-hand-server` | Docker container name. |
| `BRAINCO_MAX_ATTEMPTS` | `10` | Number of BrainCo state checks before reporting an error. |
| `BRAINCO_DDS_INTERFACE` | `PICO_INTERFACE` | DDS interface for the BrainCo state check. |
| `TELEOP_VENV`, `CAMERA_VENV`, `DEPLOY_DIR` | Unitree paths | Paths to environments and deployment files. |

Examples:

```bash
# ego RealSense + USB head camera
CAMERA_MODE=realsense-usb \
EGO_VIEW_DEVICE_ID=243422071979 \
USB_HEAD_DEVICE_ID=/dev/video6 \
HEAD_CAMERA_WIDTH=1600 \
HEAD_CAMERA_HEIGHT=896 \
HEAD_CAMERA_FPS=15 \
HEAD_CAMERA_QUALITY=90 \
HEAD_CAMERA_FOURCC=MJPG \
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh --replace

# The camera server runs separately or on another machine.
CAMERA_MODE=none \
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh --replace

# Head RealSense only: RGB 1280x960, no depth.
CAMERA_MODE=head-realsense \
HEAD_DEVICE_ID=135122071874 \
HEAD_CAMERA_WIDTH=1280 \
HEAD_CAMERA_HEIGHT=960 \
HEAD_CAMERA_FPS=15 \
HEAD_CAMERA_QUALITY=90 \
bash gear_sonic/scripts/launch_brainco_robot_tmux.sh --replace
```

Robot launcher tmux windows: `brainco`, `camera` (except for `CAMERA_MODE=none`),
`pico`, `telemetry`, and `deploy`.

## 3. Camera server

### Two RealSense cameras

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection

python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense \
  --ego-view-device-id 243422071979 \
  --head-camera realsense \
  --head-device-id 135122071874 \
  --realsense-width 1280 \
  --realsense-height 720 \
  --head-camera-width 1280 \
  --head-camera-height 720 \
  --head-camera-fps 15 \
  --head-camera-quality 80 \
  --no-realsense-depth \
  --fps 15 \
  --port 5555
```

Main arguments: `--ego-view-camera`, `--ego-view-device-id`,
`--head-camera`, `--head-device-id`, `--realsense-width`,
`--realsense-height`, `--head-camera-width`, `--head-camera-height`,
`--head-camera-fps`, `--head-camera-quality`, `--head-camera-fourcc`,
`--realsense-depth` / `--no-realsense-depth`, `--realsense-depth-width`,
`--realsense-depth-height`, `--fps`, `--port`.

The default command uses `--no-realsense-depth`: the depth sensor is not
started, and depth frames are not encoded or sent to the host.

### RealSense + USB head camera

```bash
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense \
  --ego-view-device-id 243422071979 \
  --head-camera usb \
  --head-device-id /dev/video6 \
  --realsense-width 1280 \
  --realsense-height 720 \
  --head-camera-width 1280 \
  --head-camera-height 960 \
  --head-camera-fps 15 \
  --head-camera-quality 90 \
  --head-camera-fourcc MJPG \
  --no-realsense-depth \
  --fps 15 \
  --port 5555
```

### Head RealSense only

```bash
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera None \
  --head-camera realsense \
  --head-device-id 135122071874 \
  --head-camera-width 1280 \
  --head-camera-height 960 \
  --head-camera-fps 15 \
  --head-camera-quality 90 \
  --no-realsense-depth \
  --fps 15 \
  --port 5555
```

`None` is case-sensitive here. To enable depth, use
`--realsense-depth` and set supported dimensions with
`--realsense-depth-width` / `--realsense-depth-height`.

When `--head-camera usb` is used, depth settings apply only to other RealSense
cameras: a USB head camera never creates or sends `head_depth`. Before choosing
a resolution and FPS, check supported combinations with
`v4l2-ctl --device /dev/video6 --list-formats-ext`.

## 4. BrainCo service

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/start_brainco_and_check.sh
```

The script restarts `g1-brainco-hand-server` if the container is already
running; otherwise, it starts it. It then checks the state of both hands and
restarts the container again if there is an error. The maximum is 10 checks.

Arguments: `--container`, `--attempts`, `--interface`, `--restart-wait`,
`--python`.

Examples:

```bash
# Another DDS interface
bash gear_sonic/scripts/start_brainco_and_check.sh \
  --interface wlxfc23cd997021

# Another container and five attempts
BRAINCO_CONTAINER=my-brainco-service \
bash gear_sonic/scripts/start_brainco_and_check.sh \
  --container my-brainco-service \
  --attempts 5

docker stop g1-brainco-hand-server
```

## 5. Pico manager

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection

python -m gear_sonic.scripts.pico_manager_brainco_dexterous \
  --manager \
  --port 5556 \
  --target_fps 50 \
  --brainco_network_interface wlxfc23cd997021
```

Main arguments: `--manager`, `--port`, `--target_fps`,
`--input-source {xrt,isaac-teleop}`, `--brainco_dds_domain`,
`--brainco_network_interface`, `--brainco_trigger_threshold`,
`--brainco_trigger_range`, `--brainco_excluded_fingers`,
`--disable_brainco_hand`, `--zmq_feedback_host`, `--zmq_feedback_port`.

Example trigger threshold:

```bash
--brainco_trigger_threshold 0.5
```

## 6. G1 upper-body telemetry publisher

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection

python -m gear_sonic.g1_upper_body_telemetry.robot_publisher \
  --publish-hz 50 \
  --port 5560
```

Arguments: `--publish-hz`, `--bind-host`, `--port`, `--dds-domain-id`,
`--network-interface`, `--state-topic`, `--command-topic`, `--stale-seconds`.

## 7. Gear Sonic deploy

```bash
export TensorRT_ROOT="$HOME/TensorRT"
cd /home/unitree/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real
```

Main arguments: `--input-type zmq_manager` and mode `real` or `sim`.

## 8. Video streaming server

### RealSense

```bash
cd /home/unitree/teleop-ws/XRoboToolkit-Orin-Video-Sender
REALSENSE_FORMAT=YUYV \
REALSENSE_WIDTH=640 \
REALSENSE_HEIGHT=480 \
REALSENSE_FPS=30 \
./server_realsense.sh 192.168.50.132
```

Arguments are set through variables: `REALSENSE_DEVICE`, `REALSENSE_FORMAT`,
`REALSENSE_WIDTH`, `REALSENSE_HEIGHT`, and `REALSENSE_FPS`. The final argument
is the IP address of the host computer.

---

# Host computer

## 1. Setup

```bash
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection
bash install_scripts/install_data_collection.sh --replace
```

## 2. Host tmux launcher

```bash
ROBOT_HOST=192.168.50.132 \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace
```

Launcher variables:

| Variable | Default | Values / purpose |
|---|---:|---|
| `TMUX_SESSION` | `brainco_host` | Name of the tmux session. |
| `ROBOT_HOST` | `192.168.50.132` | Main robot address. |
| `CAMERA_HOST`, `SONIC_HOST`, `STATE_HOST`, `TELEMETRY_HOST` | `ROBOT_HOST` | Addresses of separate ZMQ sources. |
| `CAMERA_PORT`, `SONIC_PORT`, `STATE_PORT`, `TELEMETRY_PORT` | `5555`, `5556`, `5557`, `5560` | Source ports. |
| `PICO_INTERFACE` | `wlp128s20f3` | BrainCo DDS interface on the host. |
| `ENABLE_CAMERA_VIEWER` | `1` | `1` creates the viewer window; `0` does not. |
| `SHOW_MOTOR_STATES` | `0` | `0` hides the motor-state/tau panel; `1` enables it and its subscribers. |
| `CAMERA_STREAMS` | `head ego_view` | Robot RGB streams for the viewer; external view is added automatically. |
| `HEAD_CAMERA_WIDTH`, `HEAD_CAMERA_HEIGHT` | `1600`, `896` | Expected robot head stream size for the exporter; must match the robot launcher. |
| `EXPORTER_EXTRA_ARGS` | empty | Extra exporter arguments. |
| `EXTERNAL_VIEW_CAMERA_DEVICE` | `/dev/video4` | Path to the local host camera; an empty string disables it. |
| `EXTERNAL_VIEW_CAMERA_WIDTH`, `EXTERNAL_VIEW_CAMERA_HEIGHT`, `EXTERNAL_VIEW_CAMERA_FPS` | `1280`, `960`, `15` | Requested V4L2 profile for the local camera. |
| `EXTERNAL_VIEW_CAMERA_FOURCC` | `MJPG` | V4L2 FourCC of the external camera. |
| `EXTERNAL_VIEW_CAMERA_HOST`, `EXTERNAL_VIEW_CAMERA_PORT` | `localhost`, `5582` | ZMQ stream of the standalone external-view camera for the exporter and viewer. |
| `DATA_COLLECTION_VENV` | `.venv_data_collection` | Path to the exporter environment. |

Examples:

```bash
# ego_view only, without the viewer
ROBOT_HOST=192.168.50.132 \
ENABLE_CAMERA_VIEWER=0 \
EXPORTER_EXTRA_ARGS="--ignore-head --record-raw-depth" \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace

# Sources are on different machines
CAMERA_HOST=192.168.50.132 \
SONIC_HOST=192.168.50.132 \
STATE_HOST=192.168.50.132 \
TELEMETRY_HOST=192.168.50.132 \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace

# Local external-view camera on the host at 1280x960
EXTERNAL_VIEW_CAMERA_DEVICE=/dev/video4 \
EXTERNAL_VIEW_CAMERA_WIDTH=1280 \
EXTERNAL_VIEW_CAMERA_HEIGHT=960 \
EXTERNAL_VIEW_CAMERA_FPS=15 \
EXTERNAL_VIEW_CAMERA_FOURCC=MJPG \
SHOW_MOTOR_STATES=0 \
bash gear_sonic/scripts/launch_brainco_host_tmux.sh --replace
```

Host launcher tmux windows: `exporter`, `external_camera` (when
`EXTERNAL_VIEW_CAMERA_DEVICE` is set), and `viewer` (when `ENABLE_CAMERA_VIEWER=1`).

### Standalone external-view camera

```bash
source .venv_data_collection/bin/activate
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection

python gear_sonic/scripts/run_external_view_camera.py \
  --device /dev/video4 \
  --width 1280 \
  --height 960 \
  --fps 15 \
  --fourcc MJPG \
  --port 5582
```

This process is the only one that opens `/dev/video4`. The exporter and viewer
subscribe to its ZMQ stream independently.

## 3. BrainCo data exporter

```bash
source .venv_data_collection/bin/activate
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection

python -m gear_sonic.scripts.brainco_data_exporter \
  --camera-host 192.168.50.132 \
  --camera-port 5555 \
  --sonic-zmq-host 192.168.50.132 \
  --sonic-zmq-port 5556 \
  --state-zmq-host 192.168.50.132 \
  --state-zmq-port 5557 \
  --g1-telemetry-zmq-host 192.168.50.132 \
  --g1-telemetry-zmq-port 5560 \
  --g1-telemetry-expected-hz 50 \
  --record-raw-telemetry \
  --depth-camera-width 1280 \
  --depth-camera-height 720 \
  --webcam-width 1600 \
  --webcam-height 896 \
  --brainco-network-interface wlp128s20f3 \
  --external-view-camera-host localhost \
  --external-view-camera-port 5582 \
  --external-view-camera-width 1280 \
  --external-view-camera-height 960
```

Before connecting to sources, the exporter asks for the task, sex, age, height,
and weight. After checking the values, you can confirm them, change one field,
or enter all values again. The dataset name is generated automatically.
Allowed tasks are `none`, `handshake`, `fist_bump`, and `hug`; sex is `f` or
`m`; age is 1–120 years; height is 120–230 cm; and weight is 40–120 kg.
For age, height, and weight, you can enter `n`. This value is saved as `none`
in the metadata and dataset name.
Example name:

```text
handshake-m-age28-height182-weight78-20260910-153045
```

Main arguments:

| Group | Arguments |
|---|---|
| Dataset | `--root-output-dir`, `--data-collection-frequency`; the task and name are set interactively. |
| ZMQ | `--camera-host`, `--camera-port`, `--sonic-zmq-host`, `--sonic-zmq-port`, `--state-zmq-host`, `--state-zmq-port`. |
| BrainCo DDS | `--brainco-dds-domain-id`, `--brainco-network-interface`, `--brainco-message-timeout`. |
| G1 telemetry | `--g1-telemetry-zmq-host`, `--g1-telemetry-zmq-port`, `--g1-telemetry-expected-hz`, `--g1-telemetry-max-age`, `--g1-telemetry-message-timeout`. |
| Cameras | `--ignore-head`, `--ignore-ego-view`, `--record-wrist-cameras`, and RGB/depth stream sizes. |
| External camera stream | `--external-view-camera-host`, `--external-view-camera-port`, `--external-view-camera-width`, `--external-view-camera-height`, `--external-view-camera-timeout`. |
| Depth | `--record-raw-depth`, `--record-depth-video`, `--depth-video-max-meters`. |
| Raw telemetry | `--record-raw-telemetry`, `--no-record-raw-telemetry`. |
| Other | `--text-to-speech`, `--no-text-to-speech`, `--episode-status-zmq-port`. |

Do not use `--ignore-head` and `--ignore-ego-view` at the same time.

Examples:

```bash
# ego_view only, with raw depth only for this camera
--ignore-head --record-raw-depth

# Head camera only
--ignore-ego-view

# Non-default: explicitly save available depth streams
--record-raw-depth --record-depth-video

# Standalone camera stream. The dataset modality is called external-view-camera.
--external-view-camera-host localhost \
--external-view-camera-port 5582 \
--external-view-camera-width 1280 \
--external-view-camera-height 960
```

## 4. G1 telemetry diagnostic receiver

```bash
source .venv_data_collection/bin/activate
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection

python -m gear_sonic.g1_upper_body_telemetry.host_receiver \
  --host 192.168.50.132 \
  --port 5560 \
  --timeout 10
```

Arguments: `--host`, `--port`, `--timeout`.

This receiver is not required for the exporter: the exporter subscribes to the
same ZMQ topic itself and can run at the same time as the diagnostic receiver.

The raw `rt/lowstate` can be checked directly with the Unitree Python SDK on
the robot:

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate

python -m gear_sonic.g1_upper_body_telemetry.inspect_upper_body_state \
  --print-hz 2
```

When running on another computer that can access DDS, add the interface:

```bash
--network-interface wlp128s20f3
```

The script prints state `q`, `dq`, `ddq`, `tau_est` and command `q`, `dq`,
`tau`, `kp`, `kd` for 17 upper-body motors. `--once` prints one
LowState/LowCmd pair and exits.

## 5. Camera viewer

```bash
source .venv_data_collection/bin/activate
cd /home/nikita/Skoltech/MWS/Teleop-Data-Collection

python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 192.168.50.132 \
  --camera-port 5555 \
  --external-view-camera-host localhost \
  --external-view-camera-port 5582 \
  --camera-streams head ego_view \
  --no-depth \
  --grid-columns 2 \
  --no-motor-states
```

Main arguments: `--camera-host`, `--camera-port`, `--camera-streams`,
`--external-view-camera-host`, `--external-view-camera-port`, `--grid-columns`,
`--depth-display-min-meters`, `--depth-display-max-meters`,
`--show-tau-plot` / `--no-show-tau-plot`, and the `--no-motor-states` alias.

The viewer and exporter receive `external-view-camera` directly from the
standalone publisher. Only `run_external_view_camera.py` opens `/dev/video*`,
so restarting the exporter does not close the camera.

For one ego camera:

```bash
--camera-streams ego_view --no-depth
```

## 6. Dataset paths

```text
outputs/<dataset-name>/
raw-telemetry/chunk-NNN/episode_NNNNNN.npz
meta/interaction_metadata.jsonl
```

`meta/interaction_metadata.jsonl` contains one line for each episode: the
participant profile, `interaction_class`, and zero-valued `approach_start`,
`contact_active_start`, `release_start`, and `idle_start` fields for later
annotation.

For the upper body, Parquet and `raw-telemetry` save `q_cmd`, `kp`, `kd`,
`q_est`, `dq_est`, `tau_est`, `q_residual`, and internal timestamp/sequence
IDs. The change applies only to new episodes.

In the default profile, the `depth/` and `depth_head/` folders are not created.

## 7. Episode and timestamp controls

- `left grip + A` — start or stop episode recording;
- `left grip + B` — save the current episode as discarded;
- press the right grip — add a timestamp;
- holding the right grip creates one timestamp; the press edge is used;
- the left grip alone does not create a timestamp and remains part of the `A/B` commands;
- press and release events for the left and right triggers are saved separately;
- the trigger press threshold is set in Pico manager with
  `--brainco_trigger_threshold`; the default is `0.5`.

---

## Copying the project to Unitree over SSH

```bash
rsync -avh --partial --info=progress2 \
  --filter=':- .gitignore' \
  --exclude='.git/' \
  /home/nikita/Skoltech/MWS/Teleop-Data-Collection/ \
  unitree@192.168.50.132:/home/unitree/teleop-ws/Teleop-Data-Collection/
```

---

## Troubleshooting

### The robot lags during Pico teleoperation

Camera capture and JPEG encoding can compete for CPU with Pico and GearSonic
inference. On the robot, find the actual process PIDs:

```bash
pgrep -af 'deploy|gear_sonic|composed_camera|pico_manager'
```

After starting all tmux windows, the easiest option is to run the helper. It
finds the PIDs, asks for `sudo` once, and applies the required values.

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/set_runtime_priorities.sh
```

The helper requires exactly one GearSonic deploy process, Pico manager, camera
server, and telemetry publisher. If two tmux sessions run at the same time, it
does not change anything and prints the duplicate processes it found.

You can apply the same values manually if needed. Increase the CPU priority of
the actual `g1_deploy_onnx_ref` binary and Pico manager, and lower it for the
camera server. Replace the placeholders with PIDs from the command above:

```bash
sudo renice -n -5 -p <DEPLOY_PID>
sudo renice -n -5 -p <PICO_MANAGER_PID>
sudo renice -n 10 -p <CAMERA_SERVER_PID>
sudo renice -n 5 -p <TELEMETRY_PID>
```

For example, if `g1_deploy_onnx_ref` has PID `17612`, Pico manager has `16999`,
the camera server has `17012`, and the telemetry publisher has `17010`:

```bash
sudo renice -n -5 -p 17612
sudo renice -n -5 -p 16999
sudo renice -n 10 -p 17012
sudo renice -n 5 -p 17010
```

Check the applied priorities:

```bash
ps -o pid,ni,cls,cmd -p <DEPLOY_PID>,<PICO_MANAGER_PID>,<CAMERA_SERVER_PID>,<TELEMETRY_PID>
```

`nice` changes only CPU scheduling: TensorRT/GPU kernels do not receive a
special priority. After a restart, PIDs and priorities change, so run the
commands again after creating a new tmux session. Do not use real-time
scheduling with `chrt` without separate testing: it can block DDS/Pico
processes and increase control latency.

### BrainCo DDS check fails

If `start_brainco_and_check.sh` fails the DDS check, stop it with `Ctrl+C` (or
wait until it finishes), then stop the container manually once:

```bash
docker stop g1-brainco-hand-server
```

Then run the check script again. It starts the container from a clean stopped
state:

```bash
cd /home/unitree/teleop-ws/Teleop-Data-Collection
bash gear_sonic/scripts/start_brainco_and_check.sh \
  --interface wlxfc23cd997021
```

Replace `wlxfc23cd997021` with the actual Pico/DDS interface if it is different.
