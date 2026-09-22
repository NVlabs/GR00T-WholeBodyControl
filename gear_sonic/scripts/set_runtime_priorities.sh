#!/usr/bin/env bash
# Apply CPU nice priorities to the active robot-side teleoperation processes.

set -Eeuo pipefail

DEPLOY_NICE=-5
PICO_NICE=-5
CAMERA_NICE=10
TELEMETRY_NICE=5

usage() {
    cat <<'EOF'
Usage: bash gear_sonic/scripts/set_runtime_priorities.sh

Finds the currently running GearSonic deploy binary, Pico manager, composed
camera server, and upper-body telemetry publisher. Applies these CPU nice
priorities (lower value = higher priority):

  GearSonic deploy:  -5
  Pico manager:      -5
  Camera server:     10
  Telemetry:         5

Run this on the robot after all tmux windows have started. The script requests
sudo once and intentionally fails without changing anything if it detects zero
or multiple matching processes for any required service.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
elif [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

find_one_pid() {
    local label="$1"
    local pattern="$2"
    local -a pids=()
    mapfile -t pids < <(pgrep -f "$pattern" || true)

    if [[ ${#pids[@]} -ne 1 ]]; then
        echo "Expected exactly one $label process; found ${#pids[@]}." >&2
        if [[ ${#pids[@]} -gt 0 ]]; then
            ps -o pid,ni,cmd -p "${pids[@]}" >&2
        fi
        echo "Start/stop duplicate tmux sessions, then run this script again." >&2
        return 1
    fi
    printf '%s\n' "${pids[0]}"
}

DEPLOY_PID="$(find_one_pid deploy '(^|/)g1_deploy_onnx_ref([[:space:]]|$)')"
PICO_PID="$(find_one_pid 'Pico manager' 'gear_sonic\.scripts\.pico_manager_brainco_dexterous.*--manager')"
CAMERA_PID="$(find_one_pid 'camera server' 'gear_sonic\.camera\.composed_camera')"
TELEMETRY_PID="$(find_one_pid 'telemetry publisher' 'gear_sonic\.g1_upper_body_telemetry\.robot_publisher')"

echo "Detected processes:"
ps -o pid,ni,cmd -p "$DEPLOY_PID,$PICO_PID,$CAMERA_PID,$TELEMETRY_PID"

echo "Requesting sudo to apply CPU priorities..."
sudo -v
sudo renice -n "$DEPLOY_NICE" -p "$DEPLOY_PID" >/dev/null
sudo renice -n "$PICO_NICE" -p "$PICO_PID" >/dev/null
sudo renice -n "$CAMERA_NICE" -p "$CAMERA_PID" >/dev/null
sudo renice -n "$TELEMETRY_NICE" -p "$TELEMETRY_PID" >/dev/null

echo "Applied CPU nice priorities:"
ps -o pid,ni,cls,cmd -p "$DEPLOY_PID,$PICO_PID,$CAMERA_PID,$TELEMETRY_PID"
