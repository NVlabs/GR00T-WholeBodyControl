# VR Teleop Setup (PICO)

This page covers the one-time hardware and software setup for PICO VR whole-body teleoperation.

```{note}
The steps below use XRoboToolkit. To use IsaacTeleop, complete Steps 1 and 2, then continue with the [Isaac Teleop Setup](../tutorials/isaac_teleop_publisher_setup.md).
```

---

## Required Hardware

- [PICO 4 / PICO 4 Pro headset](https://www.picoxr.com/global/products/pico4)
- [2x PICO controllers](https://www.picoxr.com/global/products/pico4)
- [2x PICO motion trackers](https://www.picoxr.com/global/products/pico-motion-tracker), strapped to the ankles
- A high-speed, low-latency Wi-Fi connection. Teleoperation performance depends heavily on network quality.

---

(motion-tracker-setup)=
## Step 1: Set Up the Motion Trackers

```{image} ../_static/pico_setup/pico_setup_screenshot.png
:width: 600px
:align: center
```

1. Strap one PICO motion tracker to your left ankle and one to your right ankle. Scrunch down any baggy clothing so that the trackers are visible. Make sure that the side with the light indicator faces up.
2. Open PICO settings. Select **Developer** in the menu and turn off **Safeguard**.
   - If the Developer option is not active, tap **Software** until it appears.
3. Select the **Wi-Fi icon** in the PICO menu. A picture of the headset appears. Select the small circular Motion Tracker icon above the headset. If the icon does not appear, open the **Motion Tracker** app.
4. Select the **i** icon next to each tracker, then unpair all trackers.
5. Select **Pair** in the top-right corner.
6. Press and hold the button on each motion tracker for 6 seconds. The lights flash red and blue in pairing mode.

### Calibrate the Motion Trackers

1. Wear the PICO headset over your eyes.
2. Select **Calibrate** and complete both calibration sequences:
   - Stand upright with the handheld controllers down by your sides.
   - Look down at the foot motion trackers until the headset cameras recognize them.
3. After calibration, wear the PICO headset around your forehead. Keep the headset facing forward so that it continues to detect the motion trackers.

---

## Step 2: Install the Teleop Environment

On the host receiving PICO tracking data, run the install script from the repository root:

```bash
bash install_scripts/install_pico.sh
```

The script creates a Python 3.10 virtual environment at `.venv_teleop`. The environment includes:

- The `gear_sonic[teleop]` extra for ZMQ, Pinocchio, and visualization
- XRoboToolkit SDK
- IsaacTeleop 1.4 with the CloudXR runtime
- The simulation extra for MuJoCo
- Unitree SDK2 Python bindings

Activate the environment:

```bash
source .venv_teleop/bin/activate
```

---

## Step 3: Set Up XRoboToolkit

XRoboToolkit uses a PC service on the host computer and an application on the PICO headset.

### Install the PC Service

The XRoboToolkit PC service must run on the computer that receives tracking data.

**Ubuntu 22.04 on x86_64:**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

**Ubuntu 24.04 on x86_64:**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

**Jetson on aarch64:**

```bash
sudo dpkg -i gear_sonic_deploy/thirdparty/roboticsservice_1.0.0.0_arm64.deb
```

See the [XRoboToolkit PC Service releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases) for other versions.

### Install the PICO Application

1. Open the browser in the PICO headset.
2. Enable **Developer Mode** in PICO settings.
3. Download [XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk). See [other releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases) if you need a different version.
4. Open the browser download manager and select the APK.
5. Select **Install**. The application appears in the **Unknown** section of the PICO library.

### Connect XRoboToolkit

1. Connect the host computer and PICO headset to the same Wi-Fi network.
2. Open the **XRoboToolkit** application in the headset.
3. Select the host computer's IP address in the server connection prompt. If it is not listed, enter its IPv4 address next to **PC Service**.
4. Verify that the status is **WORKING**.
5. Enable **Head** and **Controller** under Tracking.
6. Select **Send** under Data/Control.
7. Select **Full body** for the PICO Motion Tracker.

```{image} ../_static/pico_setup/xrrobot_setup.png
:width: 600px
:align: center
```

---

## Next Steps

Continue to the [ZMQ Manager tutorial](../tutorials/vr_wholebody_teleop.md) to run whole-body teleoperation in simulation or on a real robot.
