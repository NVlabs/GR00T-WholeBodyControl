#!/usr/bin/env bash
# Start the external camera, BrainCo exporter, and optional viewer in tmux.

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SESSION="${TMUX_SESSION:-brainco_host}"
REPLACE_SESSION=0

if [[ "${1:-}" == "--replace" ]]; then
    REPLACE_SESSION=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--replace]" >&2
    exit 2
fi

command -v tmux >/dev/null || {
    echo "tmux is required. Install it with: sudo apt install tmux" >&2
    exit 1
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [[ $REPLACE_SESSION -eq 0 ]]; then
        echo "tmux session '$SESSION' already exists." >&2
        echo "Attach: tmux attach -t $SESSION" >&2
        echo "Replace: $0 --replace" >&2
        exit 1
    fi
    tmux kill-session -t "$SESSION"
fi

DATA_COLLECTION_VENV="${DATA_COLLECTION_VENV:-$PROJECT_DIR/.venv_data_collection}"
# Default robot endpoint and dataset profile: two robot RGB views, one local
# external view, raw G1 telemetry, and no depth recording/display.
ROBOT_HOST="${ROBOT_HOST:-192.168.50.132}"
CAMERA_HOST="${CAMERA_HOST:-$ROBOT_HOST}"
SONIC_HOST="${SONIC_HOST:-$ROBOT_HOST}"
STATE_HOST="${STATE_HOST:-$ROBOT_HOST}"
TELEMETRY_HOST="${TELEMETRY_HOST:-$ROBOT_HOST}"
CAMERA_PORT="${CAMERA_PORT:-5555}"
SONIC_PORT="${SONIC_PORT:-5556}"
STATE_PORT="${STATE_PORT:-5557}"
TELEMETRY_PORT="${TELEMETRY_PORT:-5560}"
PICO_INTERFACE="${PICO_INTERFACE:-wlp128s20f3}"
ENABLE_CAMERA_VIEWER="${ENABLE_CAMERA_VIEWER:-1}"
SHOW_MOTOR_STATES="${SHOW_MOTOR_STATES:-0}"
CAMERA_STREAMS="${CAMERA_STREAMS:-head ego_view}"
EXPORTER_EXTRA_ARGS="${EXPORTER_EXTRA_ARGS:-}"
HEAD_CAMERA_WIDTH="${HEAD_CAMERA_WIDTH:-1600}"
HEAD_CAMERA_HEIGHT="${HEAD_CAMERA_HEIGHT:-896}"
EXTERNAL_VIEW_CAMERA_DEVICE="${EXTERNAL_VIEW_CAMERA_DEVICE-/dev/video4}"
EXTERNAL_VIEW_CAMERA_WIDTH="${EXTERNAL_VIEW_CAMERA_WIDTH:-1280}"
EXTERNAL_VIEW_CAMERA_HEIGHT="${EXTERNAL_VIEW_CAMERA_HEIGHT:-960}"
EXTERNAL_VIEW_CAMERA_FPS="${EXTERNAL_VIEW_CAMERA_FPS:-15}"
EXTERNAL_VIEW_CAMERA_FOURCC="${EXTERNAL_VIEW_CAMERA_FOURCC:-MJPG}"
EXTERNAL_VIEW_CAMERA_HOST="${EXTERNAL_VIEW_CAMERA_HOST:-localhost}"
EXTERNAL_VIEW_CAMERA_PORT="${EXTERNAL_VIEW_CAMERA_PORT:-5582}"

external_view_camera_args=""
external_view_viewer_args=""
external_view_publisher_command=""
if [[ -n "$EXTERNAL_VIEW_CAMERA_DEVICE" ]]; then
    external_view_camera_args="--external-view-camera-host $EXTERNAL_VIEW_CAMERA_HOST --external-view-camera-port $EXTERNAL_VIEW_CAMERA_PORT --external-view-camera-width $EXTERNAL_VIEW_CAMERA_WIDTH --external-view-camera-height $EXTERNAL_VIEW_CAMERA_HEIGHT"
    external_view_viewer_args="--external-view-camera-host $EXTERNAL_VIEW_CAMERA_HOST --external-view-camera-port $EXTERNAL_VIEW_CAMERA_PORT"
    external_view_publisher_command="source $DATA_COLLECTION_VENV/bin/activate && cd $PROJECT_DIR && python gear_sonic/scripts/run_external_view_camera.py --device $EXTERNAL_VIEW_CAMERA_DEVICE --width $EXTERNAL_VIEW_CAMERA_WIDTH --height $EXTERNAL_VIEW_CAMERA_HEIGHT --fps $EXTERNAL_VIEW_CAMERA_FPS --port $EXTERNAL_VIEW_CAMERA_PORT"
    if [[ -n "$EXTERNAL_VIEW_CAMERA_FOURCC" ]]; then
        external_view_publisher_command+=" --fourcc $EXTERNAL_VIEW_CAMERA_FOURCC"
    fi
fi

case "$SHOW_MOTOR_STATES" in
    0) motor_states_viewer_args="--no-motor-states" ;;
    1) motor_states_viewer_args="--show-tau-plot" ;;
    *)
        echo "SHOW_MOTOR_STATES must be 0 or 1" >&2
        exit 2
        ;;
esac

tmux new-session -d -s "$SESSION" -n exporter
tmux set-option -t "$SESSION" -g mouse on

if [[ -n "$external_view_publisher_command" ]]; then
    tmux new-window -d -t "$SESSION" -n external_camera
    tmux send-keys -t "$SESSION:external_camera" "$external_view_publisher_command" C-m
fi

exporter_command="source $DATA_COLLECTION_VENV/bin/activate && cd $PROJECT_DIR && python -m gear_sonic.scripts.brainco_data_exporter --camera-host $CAMERA_HOST --camera-port $CAMERA_PORT --sonic-zmq-host $SONIC_HOST --sonic-zmq-port $SONIC_PORT --state-zmq-host $STATE_HOST --state-zmq-port $STATE_PORT --g1-telemetry-zmq-host $TELEMETRY_HOST --g1-telemetry-zmq-port $TELEMETRY_PORT --g1-telemetry-expected-hz 50 --brainco-network-interface $PICO_INTERFACE --depth-camera-width 1280 --depth-camera-height 720 --webcam-width $HEAD_CAMERA_WIDTH --webcam-height $HEAD_CAMERA_HEIGHT --record-raw-telemetry $external_view_camera_args $EXPORTER_EXTRA_ARGS"
tmux send-keys -t "$SESSION:exporter" "$exporter_command" C-m

if [[ "$ENABLE_CAMERA_VIEWER" == "1" ]]; then
    tmux new-window -d -t "$SESSION" -n viewer
    viewer_command="source $DATA_COLLECTION_VENV/bin/activate && cd $PROJECT_DIR && python gear_sonic/scripts/run_camera_viewer.py --camera-host $CAMERA_HOST --camera-port $CAMERA_PORT --camera-streams $CAMERA_STREAMS --no-depth --grid-columns 2 $external_view_viewer_args $motor_states_viewer_args"
    tmux send-keys -t "$SESSION:viewer" "$viewer_command" C-m
fi

tmux select-window -t "$SESSION:exporter"
echo "Started host-side session: $SESSION"
echo "Dataset name will be generated from the answers in the exporter window."
echo "Attach: tmux attach -t $SESSION"
