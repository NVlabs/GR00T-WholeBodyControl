robot ip - 192.168.50.132

# On robot

# camera server
source .venv_camera/bin/activate
python -m gear_sonic.camera.composed_camera \
    --ego-view-camera usb \
    --ego-view-device-id 4 \
    --port 5555



# On server


# Gear sonic execution 

cd gear_sonic_deploy
source scripts/setup_env.sh



  bash deploy.sh --input-type zmq \
  --zmq-host 192.168.50.109 \
  --zmq-port 5556 \
  --zmq-topic pose \
  real

# 'real' auto-detects the robot network interface (192.168.123.x).
# If auto-detection fails, pass the G1's IP directly:
#   ./deploy.sh --input-type zmq_manager <G1-IP>



source .venv_data_collection/bin/activate

# Data exporter
python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the cup" \
    --camera-host 192.168.50.132 --camera-port 5555

# Camera viewer (to verify the feed)
python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host 192.168.50.132 --camera-port 5555


source .venv_teleop/bin/activate

# With full visualization (recommended for first run):
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --vis_vr3pt --vis_smpl

python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager --port 5558 --vis_vr3pt --vis_smpl




# Pico streaming

source .venv_teleop/bin/activate

# With full visualization (recommended for first run):
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --vis_vr3pt --vis_smpl

# Without visualization (for headless / onboard practice):
python gear_sonic/scripts/pico_manager_thread_server.py --manager


# VR_3PT Calibration Hint

The VR_3PT mode depends on accurate calibration. Two calibration events occur:
One-time CALIB_FULL (head + wrists)

Before pressing A + B + X + Y for the first time, you must stand in the robot’s all-zero reference pose. The system captures your PICO body-tracking frame as the zero-reference for all subsequent motion mapping.

The reference pose:

    Stand upright, feet together, looking straight forward.

    Upper arms hang straight down, close to your torso.

    Forearms bent 90° forward (L-shape at each elbow), palms facing inward.




# Your First Teleop Session

    Assume the calibration pose — stand upright, feet together, upper arms at your sides, forearms bent 90° forward (L-shape at each elbow), palms inward. See Calibration Pose for details.

    Press A + B + X + Y simultaneously to engage the control policy and run the initial full calibration (CALIB_FULL).

    Align your arms with the robot’s current pose, then press A + X to enter full-body SMPL teleop (POSE mode). Move your arms and legs — the robot follows.

    Press A + X again to fall back to PLANNER (idle) mode.

    Press A + B + X + Y again to stop the robot.





# DANGER — Mode-Switching Safety

Before switching into POSE or VR_3PT, always align your body with the robot’s current pose first.

Recovery from bad VR_3PT calibration:

    Freeze the upper body — switch back via Left Stick Click.

    Re-align your arms with the robot’s current (possibly distorted) pose.

    Switch to POSE mode (A+X) to reset.

Below is the recovery procedure — if you accidentally enter a badly calibrated VR_3PT state, freeze the upper body (Left Stick Click back), then switch to POSE mode (A+X) to reset safely.





python gear_sonic/scripts/run_camera_viewer.py     --camera-host 192.168.50.132 --camera-port 5555


python gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "pick up the cup" \
    --camera-host 192.168.50.132 --camera-port 5555 \
    --state-zmq-host 192.168.50.132 --state-zmq-port 5557 \
    --sonic-zmq-host 192.168.50.132 --sonic-zmq-port 5556