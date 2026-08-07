# XRoboToolkit-PC-Service-Pybind

This project provides a python interface to extract XR state using XRoboToolkit-PC-Service sdk.

## Requirements

- [`pybind11`](https://github.com/pybind/pybind11)
- [`XRoboRoolkit PC Service`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service#)

## Building the Project
### Ubuntu 22.04

```
conda remove --name xr --all
conda create -n xr python=3.10
conda activate xr

mkdir -p tmp
cd tmp
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK 
bash build.sh
cd ../../../..

mkdir -p lib
mkdir -p include
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
# rm -rf tmp

# Build the project
conda install -c conda-forge pybind11

pip uninstall -y xrobotoolkit_sdk
python setup.py install
```
### Linux Ubuntu 22.04 arm64 version (Nvidia orin supported)
```
bash setup_orin.sh
```
### Windows

**Ensure pybind11 is installed before running the following command.**

```
setup_windows.bat
```

## Using the Python Bindings

**1. Get Controller and Headset Poses**

```python
import xrobotoolkit_sdk as xrt

xrt.init()

left_pose = xrt.get_left_controller_pose()
right_pose = xrt.get_right_controller_pose()
headset_pose = xrt.get_headset_pose()

print(f"Left Controller Pose: {left_pose}")
print(f"Right Controller Pose: {right_pose}")
print(f"Headset Pose: {headset_pose}")

xrt.close()
```

**2. Get Controller Inputs (Triggers, Grips, Buttons, Axes)**

```python
import xrobotoolkit_sdk as xrt

xrt.init()

# Triggers and Grips
left_trigger = xrt.get_left_trigger()
right_grip = xrt.get_right_grip()
print(f"Left Trigger: {left_trigger}, Right Grip: {right_grip}")

# Buttons
a_button_pressed = xrt.get_A_button()
x_button_pressed = xrt.get_X_button()
print(f"A Button Pressed: {a_button_pressed}, X Button Pressed: {x_button_pressed}")

# Axes
left_axis = xrt.get_left_axis()
right_axis_click = xrt.get_right_axis_click()
print(f"Left Axis: {left_axis}, Right Axis Clicked: {right_axis_click}")

# Timestamp
timestamp = xrt.get_time_stamp_ns()
print(f"Current Timestamp (ns): {timestamp}")

xrt.close()
```

**3. Get hand tracking state**
```python
import xrobotoolkit_sdk as xrt

xrt.init()

# Left Hand State
left_hand_tracking_state = xrt.get_left_hand_tracking_state()
print(f"Left Hand State: {left_hand_tracking_state}")

# Left Hand isActive
left_hand_is_active = xrt.get_left_hand_is_active()
print(f"Left Hand isActive: {left_hand_is_active}")

# Right Hand State
right_hand_tracking_state = xrt.get_right_hand_tracking_state()
print(f"Right Hand State: {right_hand_tracking_state}")

# Right Hand isActive
right_hand_is_active = xrt.get_right_hand_is_active()
print(f"Right Hand isActive: {right_hand_is_active}")

xrt.close()
```

**4. Get whole body motion tracking （please refer to this example when check Full Body tracking mode in UNITY app）**
```python
import xrobotoolkit_sdk as xrt

xrt.init()

# Check if body tracking data is available
if xrt.is_body_data_available():
    # Get body joint poses (24 joints, 7 values each: x,y,z,qx,qy,qz,qw)
    body_poses = xrt.get_body_joints_pose()
    print(f"Body joints pose data: {body_poses}")
    
    # Get body joint velocities (24 joints, 6 values each: vx,vy,vz,wx,wy,wz)
    body_velocities = xrt.get_body_joints_velocity()
    print(f"Body joints velocity data: {body_velocities}")
    
    # Get body joint accelerations (24 joints, 6 values each: ax,ay,az,wax,way,waz)
    body_accelerations = xrt.get_body_joints_acceleration()
    print(f"Body joints acceleration data: {body_accelerations}")
    
    # Get IMU timestamps for each joint
    imu_timestamps = xrt.get_body_joints_timestamp()
    print(f"IMU timestamps: {imu_timestamps}")
    
    # Get body data timestamp
    body_timestamp = xrt.get_body_timestamp_ns()
    print(f"Body data timestamp: {body_timestamp}")
    
    # Example: Get specific joint data (Head joint is index 15)
    head_pose = body_poses[15]  # Head joint
    x, y, z, qx, qy, qz, qw = head_pose
    print(f"Head pose: Position({x:.3f}, {y:.3f}, {z:.3f}) Rotation({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
else:
    print("Body tracking data not available. Make sure:")
    print("1. PICO headset is connected")
    print("2. Body tracking is enabled in the control panel")
    print("3. At least two Pico Swift devices are connected and calibrated")

xrt.close()
```

**Body Joint Indices (similar to SMPL, 24 joints total):**
- 0: Pelvis, 1: Left Hip, 2: Right Hip, 3: Spine1, 4: Left Knee, 5: Right Knee
- 6: Spine2, 7: Left Ankle, 8: Right Ankle, 9: Spine3, 10: Left Foot, 11: Right Foot
- 12: Neck, 13: Left Collar, 14: Right Collar, 15: Head, 16: Left Shoulder, 17: Right Shoulder
- 18: Left Elbow, 19: Right Elbow, 20: Left Wrist, 21: Right Wrist, 22: Left Hand, 23: Right Hand
