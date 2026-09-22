#!/usr/bin/env bash
# Start the robot-side BrainCo teleoperation processes in separate tmux windows.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SESSION="${TMUX_SESSION:-brainco_robot}"
REPLACE_SESSION=0

if [[ "${1:-}" == "--replace" ]]; then
    REPLACE_SESSION=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--replace]" >&2
    exit 2
fi

for command in tmux docker; do
    command -v "$command" >/dev/null || {
        echo "Required command is missing: $command" >&2
        exit 1
    }
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [[ $REPLACE_SESSION -eq 0 ]]; then
        echo "tmux session '$SESSION' already exists." >&2
        echo "Attach: tmux attach -t $SESSION" >&2
        echo "Replace: $0 --replace" >&2
        exit 1
    fi
    tmux kill-session -t "$SESSION"
fi

TELEOP_VENV="${TELEOP_VENV:-/home/unitree/GR00T-WholeBodyControl/.venv_teleop}"
CAMERA_VENV="${CAMERA_VENV:-/home/unitree/GR00T-WholeBodyControl/.venv_camera}"
DEPLOY_DIR="${DEPLOY_DIR:-/home/unitree/GR00T-WholeBodyControl/gear_sonic_deploy}"
BRAINCO_CONTAINER="${BRAINCO_CONTAINER:-g1-brainco-hand-server}"
BRAINCO_MAX_ATTEMPTS="${BRAINCO_MAX_ATTEMPTS:-10}"
PICO_INTERFACE="${PICO_INTERFACE:-wlxfc23cd997021}"
BRAINCO_DDS_INTERFACE="${BRAINCO_DDS_INTERFACE:-$PICO_INTERFACE}"
PICO_PORT="${PICO_PORT:-5556}"
TELEMETRY_PORT="${TELEMETRY_PORT:-5560}"
TELEMETRY_HZ="${TELEMETRY_HZ:-50}"

# Default camera profile: 1280x720 RealSense ego view and 1600x896 USB head,
# both at 15 FPS, with no RealSense depth stream.
# CAMERA_MODE: two-realsense, realsense-usb, head-realsense, or none.
CAMERA_MODE="${CAMERA_MODE:-realsense-usb}"
EGO_VIEW_DEVICE_ID="${EGO_VIEW_DEVICE_ID:-243422071979}"
HEAD_DEVICE_ID="${HEAD_DEVICE_ID:-135122071874}"
USB_HEAD_DEVICE_ID="${USB_HEAD_DEVICE_ID:-/dev/video6}"
CAMERA_FPS="${CAMERA_FPS:-15}"
REALSENSE_WIDTH="${REALSENSE_WIDTH:-1280}"
REALSENSE_HEIGHT="${REALSENSE_HEIGHT:-720}"
REALSENSE_DEPTH="${REALSENSE_DEPTH:-0}"
# HEAD_CAMERA_* overrides apply to head independently of its USB/RealSense type.
HEAD_CAMERA_WIDTH="${HEAD_CAMERA_WIDTH:-1600}"
HEAD_CAMERA_HEIGHT="${HEAD_CAMERA_HEIGHT:-896}"
HEAD_CAMERA_FPS="${HEAD_CAMERA_FPS:-15}"
HEAD_CAMERA_QUALITY="${HEAD_CAMERA_QUALITY:-80}"
HEAD_CAMERA_FOURCC="${HEAD_CAMERA_FOURCC:-MJPG}"
HEAD_REALSENSE_DEPTH_WIDTH="${HEAD_REALSENSE_DEPTH_WIDTH:-640}"
HEAD_REALSENSE_DEPTH_HEIGHT="${HEAD_REALSENSE_DEPTH_HEIGHT:-480}"

realsense_depth_args="--no-realsense-depth"
if [[ "$REALSENSE_DEPTH" == "1" ]]; then
    realsense_depth_args="--realsense-depth"
fi

new_window() {
    local name="$1"
    local command="$2"
    tmux new-window -d -t "$SESSION" -n "$name"
    tmux send-keys -t "$SESSION:$name" "$command" C-m
}

tmux new-session -d -s "$SESSION" -n brainco
tmux set-option -t "$SESSION" -g mouse on
tmux send-keys -t "$SESSION:brainco" "cd $PROJECT_DIR && bash gear_sonic/scripts/start_brainco_and_check.sh --container $BRAINCO_CONTAINER --attempts $BRAINCO_MAX_ATTEMPTS --interface $BRAINCO_DDS_INTERFACE" C-m

case "$CAMERA_MODE" in
    two-realsense)
        new_window camera \
            "source $CAMERA_VENV/bin/activate && cd $PROJECT_DIR && python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --ego-view-device-id $EGO_VIEW_DEVICE_ID --head-camera realsense --head-device-id $HEAD_DEVICE_ID --realsense-width $REALSENSE_WIDTH --realsense-height $REALSENSE_HEIGHT --head-camera-width $HEAD_CAMERA_WIDTH --head-camera-height $HEAD_CAMERA_HEIGHT --head-camera-fps $HEAD_CAMERA_FPS --head-camera-quality $HEAD_CAMERA_QUALITY $realsense_depth_args --fps $CAMERA_FPS --port 5555"
        ;;
    realsense-usb)
        new_window camera \
            "source $CAMERA_VENV/bin/activate && cd $PROJECT_DIR && python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --ego-view-device-id $EGO_VIEW_DEVICE_ID --head-camera usb --head-device-id $USB_HEAD_DEVICE_ID --realsense-width $REALSENSE_WIDTH --realsense-height $REALSENSE_HEIGHT --head-camera-width $HEAD_CAMERA_WIDTH --head-camera-height $HEAD_CAMERA_HEIGHT --head-camera-fps $HEAD_CAMERA_FPS --head-camera-quality $HEAD_CAMERA_QUALITY --head-camera-fourcc $HEAD_CAMERA_FOURCC $realsense_depth_args --fps $CAMERA_FPS --port 5555"
        ;;
    head-realsense)
        new_window camera \
            "source $CAMERA_VENV/bin/activate && cd $PROJECT_DIR && python -m gear_sonic.camera.composed_camera --ego-view-camera None --head-camera realsense --head-device-id $HEAD_DEVICE_ID --head-camera-width $HEAD_CAMERA_WIDTH --head-camera-height $HEAD_CAMERA_HEIGHT --head-camera-fps $HEAD_CAMERA_FPS --head-camera-quality $HEAD_CAMERA_QUALITY --realsense-depth-width $HEAD_REALSENSE_DEPTH_WIDTH --realsense-depth-height $HEAD_REALSENSE_DEPTH_HEIGHT $realsense_depth_args --fps $CAMERA_FPS --port 5555"
        ;;
    none)
        ;;
    *)
        echo "Unsupported CAMERA_MODE: $CAMERA_MODE" >&2
        echo "Use: two-realsense, realsense-usb, head-realsense, or none" >&2
        tmux kill-session -t "$SESSION"
        exit 2
        ;;
esac

new_window pico \
    "source $TELEOP_VENV/bin/activate && cd $PROJECT_DIR && python -m gear_sonic.scripts.pico_manager_brainco_dexterous --manager --port $PICO_PORT --target_fps 50 --brainco_network_interface $PICO_INTERFACE"

new_window telemetry \
    "source $TELEOP_VENV/bin/activate && cd $PROJECT_DIR && python -m gear_sonic.g1_upper_body_telemetry.robot_publisher --publish-hz $TELEMETRY_HZ --port $TELEMETRY_PORT"

new_window deploy \
    "export TensorRT_ROOT=\$HOME/TensorRT && cd $DEPLOY_DIR && source scripts/setup_env.sh && ./deploy.sh --input-type zmq_manager real"

tmux select-window -t "$SESSION:pico"
echo "Started robot-side session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
