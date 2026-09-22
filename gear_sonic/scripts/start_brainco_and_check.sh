#!/usr/bin/env bash
# Start BrainCo from a clean stopped state and verify hand DDS states.

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONTAINER="${BRAINCO_CONTAINER:-g1-brainco-hand-server}"
MAX_ATTEMPTS="${BRAINCO_MAX_ATTEMPTS:-10}"
DDS_INTERFACE="${BRAINCO_DDS_INTERFACE:-}"
RESTART_WAIT_SEC="${BRAINCO_RESTART_WAIT_SEC:-2}"
CHECK_SCRIPT="$PROJECT_DIR/gear_sonic/scripts/pico_manager_brainco_dexterous/check_brainco_states.py"

usage() {
    cat <<EOF
Usage: $0 [options]

Stop and start the BrainCo container, then verify both hand DDS state topics.
After a failed check, repeat stop -> start and retry (10 checks by default).

Options:
  --container NAME       Docker container name (default: $CONTAINER)
  --attempts N           Total check attempts (default: $MAX_ATTEMPTS)
  --interface NAME       DDS network interface; omit for SDK auto-selection
  --restart-wait SEC     Seconds to wait after each start (default: $RESTART_WAIT_SEC)
  --python PATH          Python interpreter for the check script
  -h, --help             Show this help

Environment alternatives:
  BRAINCO_CONTAINER, BRAINCO_MAX_ATTEMPTS, BRAINCO_DDS_INTERFACE,
  BRAINCO_RESTART_WAIT_SEC, BRAINCO_PYTHON, PROJECT_DIR
EOF
}

PYTHON_BIN="${BRAINCO_PYTHON:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container) CONTAINER="$2"; shift 2 ;;
        --attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --interface) DDS_INTERFACE="$2"; shift 2 ;;
        --restart-wait) RESTART_WAIT_SEC="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--attempts must be a positive integer, got: $MAX_ATTEMPTS" >&2
    exit 2
fi

if ! [[ "$RESTART_WAIT_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--restart-wait must be non-negative, got: $RESTART_WAIT_SEC" >&2
    exit 2
fi

command -v docker >/dev/null || {
    echo "docker is required" >&2
    exit 2
}

if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in \
        /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/python \
        "$PROJECT_DIR/.venv_teleop/bin/python" \
        python3; do
        if [[ "$candidate" == "python3" ]] || [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

if [[ ! -f "$CHECK_SCRIPT" ]]; then
    echo "BrainCo check script was not found: $CHECK_SCRIPT" >&2
    exit 2
fi

if [[ "$PYTHON_BIN" == */* ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "Python interpreter is not executable: $PYTHON_BIN" >&2
        exit 2
    fi
elif ! command -v "$PYTHON_BIN" >/dev/null; then
    echo "Python interpreter was not found: $PYTHON_BIN" >&2
    exit 2
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Docker container does not exist: $CONTAINER" >&2
    exit 2
fi

stop_and_start_container() {
    echo "Stopping BrainCo container: $CONTAINER"
    docker stop "$CONTAINER" >/dev/null || return 1
    echo "Starting BrainCo container: $CONTAINER"
    docker start "$CONTAINER" >/dev/null || return 1
    sleep "$RESTART_WAIT_SEC"
}

run_check() {
    if [[ -n "$DDS_INTERFACE" ]]; then
        "$PYTHON_BIN" "$CHECK_SCRIPT" "$DDS_INTERFACE"
    else
        "$PYTHON_BIN" "$CHECK_SCRIPT"
    fi
}

if ! stop_and_start_container; then
    echo "Could not stop and start BrainCo container: $CONTAINER" >&2
    exit 2
fi

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    echo "BrainCo DDS check: attempt $attempt/$MAX_ATTEMPTS"
    if run_check; then
        echo "BrainCo hand states are ready."
        exit 0
    fi

    if (( attempt == MAX_ATTEMPTS )); then
        break
    fi

    echo "BrainCo check failed; repeating stop -> start: $CONTAINER"
    if ! stop_and_start_container; then
        echo "Could not stop and start BrainCo container: $CONTAINER" >&2
        exit 2
    fi
done

echo "BrainCo hand states were not received after $MAX_ATTEMPTS attempts." >&2
exit 1
