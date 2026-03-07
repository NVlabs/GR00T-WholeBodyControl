#!/bin/bash
# Start the RealSense camera ZMQ server on the Orin.
# Usage: ./start_camera_server.sh [--port PORT] [--mount-position POS]
#
# This script:
#   1. Kills any existing camera processes holding the device
#   2. Hardware-resets the RealSense to free it
#   3. Starts the ZMQ camera server

set -e

PORT="${1:-5555}"
MOUNT_POSITION="${2:-ego_view}"
REPO_DIR="/home/unitree/GR00T-WholeBodyControl"

echo "=== RealSense Camera Server ==="

# Step 1: Kill anything that might be holding the camera
echo "[1/3] Killing existing camera processes..."
pkill -9 -f camera_stream 2>/dev/null && echo "  Killed camera_stream" || true
pkill -9 -f "realsense.*--server" 2>/dev/null && echo "  Killed old realsense server" || true
sleep 1

# Step 2: Hardware-reset the RealSense
echo "[2/3] Resetting RealSense hardware..."
python3 -c "
import pyrealsense2 as rs, time
ctx = rs.context()
devs = ctx.query_devices()
if len(devs) == 0:
    print('  No RealSense device found! Check USB connection.')
    exit(1)
for d in devs:
    name = d.get_info(rs.camera_info.name)
    print(f'  Resetting {name}...')
    d.hardware_reset()
time.sleep(4)
print('  Reset complete.')
"

# Step 3: Start the server
echo "[3/3] Starting ZMQ server on port ${PORT} (mount: ${MOUNT_POSITION})..."
echo "  Press Ctrl+C to stop."
echo ""
cd "$REPO_DIR"
python3 -m decoupled_wbc.control.sensor.realsense \
    --server \
    --port "$PORT" \
    --mount-position "$MOUNT_POSITION"
