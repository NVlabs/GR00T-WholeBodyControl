# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project provides Python bindings for the XRoboToolkit PC Service SDK, enabling Python applications to extract XR state data including controller poses, hand tracking, and body motion capture from XR devices (primarily PICO headsets).

## Architecture

The project consists of:

- **Core C++ Bindings** (`bindings/py_bindings.cpp`): Pybind11-based C++ module that wraps the PXREARobotSDK
- **SDK Integration**: Uses the XRoboToolkit-PC-Service SDK (cloned from external repository)
- **Build System**: CMake-based build with Python setuptools integration
- **Multi-platform Support**: Linux (x86_64/aarch64) and Windows

Key components:
- `PXREARobotSDK.h`: Main SDK header providing device connectivity and data parsing
- `py_bindings.cpp`: Thread-safe C++ wrapper with mutex-protected global state variables
- JSON parsing using nlohmann/json for device state data
- Callback-based data updates from the SDK

## Build Commands

### Ubuntu/Linux Setup and Build
```bash
# Full setup (downloads dependencies and builds)
bash setup_ubuntu.sh

# Manual build after setup
python setup.py install

# Clean build artifacts
python setup.py clean
```

### Windows Setup and Build
```batch
# Full setup (downloads dependencies and builds)
setup_windows.bat

# Manual build after setup
python setup.py install
```

### Development Commands
```bash
# Uninstall existing package
pip uninstall -y xrobotoolkit_sdk

# Install pybind11 dependency
conda install -c conda-forge pybind11
# or
pip install pybind11

# Build and install
python setup.py install
```

## Data Flow and Threading

The SDK uses a callback-based architecture:
- `OnPXREAClientCallback`: Main callback function that receives JSON data from connected devices
- Global state variables (poses, button states, etc.) are updated in real-time
- Thread-safe access via mutex locks for each data category
- Data parsing from comma-separated pose strings to arrays

## Key Functions and Data Types

### Controller Data
- Poses: `std::array<double, 7>` (x,y,z,qx,qy,qz,qw)
- Buttons: Menu, Primary, Secondary, Axis Click
- Analog: Trigger, Grip, Axis (x,y)

### Hand Tracking
- 26 joints per hand with 7 values each (position + quaternion)
- Hand scale factor

### Body Tracking
- 24 body joints with pose, velocity, acceleration data
- IMU timestamps for each joint
- Availability flag for body tracking system

## Dependencies

### Required
- pybind11 (Python binding framework)
- CMake (build system)
- XRoboToolkit-PC-Service SDK (automatically downloaded during setup)

### Platform-specific Libraries
- Linux: `libPXREARobotSDK.so`
- Windows: `PXREARobotSDK.dll` and `PXREARobotSDK.lib`

## Testing

No formal test suite is included. Test functionality using the example scripts in `examples/`:
- `example.py`: Basic controller and headset pose testing
- `example_body_tracking.py`: Body tracking functionality
- `run_binding_continuous.py`: Continuous data capture

## Important Notes

- The SDK requires active XR device connection (PICO headset)
- Body tracking requires at least two Pico Swift devices
- All data access is thread-safe but real-time dependent on device connectivity
- The project builds a Python extension module that must be installed to site-packages