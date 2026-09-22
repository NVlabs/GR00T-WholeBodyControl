# Data recording
# On robot

# Run brainco hand server | Iarchuk 


```bash
docker start g1-brainco-hand-server
```

```
if doesn't work left hand, try stop    
```


```bash
docker stop g1-brainco-hand-server
docker start g1-brainco-hand-server
```


# Gear Sonic



```bash
cd /home/unitree/GR00T-WholeBodyControl

cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real
```

## (or it)
```bash
cd /home/unitree/GR00T-WholeBodyControl/gear_sonic_deploy

./deploy.sh \
  --cp policy/release/model \
  --obs-config policy/release/observation_config.yaml \
  --input-type zmq_manager \
  real
```
if errors appears, export that and try again
```
export TensorRT_ROOT="$HOME/TensorRT"
```


# Pico streamer

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection
python gear_sonic/scripts/pico_manager_thread_server_brainco_dexterous_adaptive_control.py \
  --manager \
  --port 5556 \
  --target_fps 50 \
  --brainco_network_interface wlxfc23cd997021
```

# Set camera server



```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense \
  --ego-view-device-id 243422071979 \
  --realsense-width 1280 \
  --realsense-height 720 \
  --realsense-depth-width 640 \
  --realsense-depth-height 480 \
  --fps 15 \
  --port 5555

```


# Launch video streaming server to transfer frame into the vr headset

```
cd /home/unitree/teleop-ws/XRoboToolkit-Orin-Video-Sender
```
## Realsense

```
cd /home/unitree/teleop-ws/XRoboToolkit-Orin-Video-Sender
REALSENSE_FORMAT=YUYV \
REALSENSE_WIDTH=640 \
REALSENSE_HEIGHT=480 \
REALSENSE_FPS=30 \
./server_realsense.sh 192.168.50.132
```

## WebCam

```
cd /home/unitree/teleop-ws/XRoboToolkit-Orin-Video-Sender
REALSENSE_DEVICE=/dev/video6 \
REALSENSE_FORMAT=MJPG \
REALSENSE_WIDTH=1280 \
REALSENSE_HEIGHT=720 \
REALSENSE_FPS=30 \
./server_realsense.sh 192.168.50.132
```




# On local pc

## Disable the firewall on your local pc !


```bash
sudo ufw disable
```

## Run camera viewer
```bash
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 192.168.50.132 \
  --camera-port 5555 \
  --camera-streams ego_view \
  --no-depth \
  --grid-columns 1 \
  --no-show-tau-plot
```


## Run exporter for dataset collection
Set the network interface for wlan, using this command 

```bash
ip -br addr
```
via editing config inside the code below


```bash
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter_brainco_extented.py \
  --task-prompt "dataset_name" \
  --dataset-name brainco-session-001 \
  --camera-host 192.168.50.132 \
  --camera-port 5555 \
  --sonic-zmq-host 192.168.50.132 \
  --sonic-zmq-port 5556 \
  --state-zmq-host 192.168.50.132 \
  --state-zmq-port 5557 \
  --brainco-network-interface <WLAN_INTERFACE> \
  --no-record-wrist-cameras
```