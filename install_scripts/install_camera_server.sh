#!/usr/bin/env bash
# install_camera_server.sh
# Sets up the .venv_camera venv for running the composed camera server
# on the robot computer.
#
# Installs gear_sonic[camera] which includes the ZMQ-based camera server
# framework and the depthai SDK (OAK cameras). For other camera SDKs
# (e.g. pyrealsense2), install them into the venv after setup.
#
# Usage:  bash install_scripts/install_camera_server.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 0. Print detected architecture ───────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

# ── 1. Ensure uv is installed and available ──────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 ─────────────────────────────────────
echo "[INFO] Installing uv-managed Python 3.10 …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ─────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_camera (if present) …"
rm -rf .venv_camera

# ── 4. Create venv & install camera extra ────────────────────────────────────
echo "[INFO] Creating .venv_camera with uv-managed Python 3.10 …"
uv venv .venv_camera --python "$MANAGED_PY" --prompt gear_sonic_camera
# shellcheck disable=SC1091
source .venv_camera/bin/activate
echo "[INFO] Installing gear_sonic[camera] …"
uv pip install -e "gear_sonic[camera]"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Camera server venv setup complete!"
echo "  depthai (OAK cameras) is included by default."
echo ""
echo "  Activate the venv with:"
echo "    source .venv_camera/bin/activate"
echo ""
echo "  For other camera SDKs, install into the venv:"
echo "    pip install pyrealsense2                    # Intel RealSense"
echo "    python /usr/local/zed/get_python_api.py     # Stereolabs ZED (install the ZED SDK first)"
echo ""
echo "  See docs/source/tutorials/data_collection.md for full setup."
echo "══════════════════════════════════════════════════════════════"

# ── 5. Optionally install the systemd service ────────────────────────────────
SERVICE_TEMPLATE="$REPO_ROOT/systemd/composed_camera_server.service"
SERVICE_NAME="composed_camera_server.service"

if [ ! -f "$SERVICE_TEMPLATE" ]; then
    echo ""
    echo "[WARN] systemd template not found at $SERVICE_TEMPLATE — skipping."
    exit 0
fi

echo ""
read -rp "Install the camera server as a systemd service (auto-start on boot)? [y/N] " INSTALL_SERVICE
if [[ ! "$INSTALL_SERVICE" =~ ^[Yy]$ ]]; then
    echo ""
    echo "  Skipped systemd install. You can run the camera server manually:"
    echo "    source .venv_camera/bin/activate"
    echo "    python -m gear_sonic.camera.composed_camera --ego-view-camera oak --port 5555"
    echo ""
    exit 0
fi

# Gather configuration
echo ""
echo "── Camera service configuration ──"
echo "  Each camera needs a type and a device ID so the server knows"
echo "  which physical camera maps to each mount position (ego, wrist, etc.)."
echo ""

detect_oak_cameras() {
    "${REPO_ROOT}/.venv_camera/bin/python" -c "
import depthai as dai
devices = dai.Device.getAllAvailableDevices()
if not devices:
    exit(1)
for i, d in enumerate(devices):
    # Try to get actual MxID; fall back to string representation
    mxid = None
    for attr in ['mxid', 'getMxId']:
        if hasattr(d, attr):
            val = getattr(d, attr)
            mxid = val() if callable(val) else val
            if mxid:
                break
    state = d.state.name if hasattr(d, 'state') else 'N/A'
    name = getattr(d, 'name', '')
    if mxid and mxid != name:
        print(f'    [{i}] MxId: {mxid}  port: {name}  state: {state}')
    else:
        # MxID not available; show all useful attributes
        print(f'    [{i}] device: {d}  state: {state}')
        print(f'         attributes: {[a for a in dir(d) if not a.startswith(\"_\")]}')
" 2>&1
}

while true; do
    echo "  Detecting connected OAK cameras …"
    OAK_DEVICES="$(detect_oak_cameras)" && OAK_FOUND=true || OAK_FOUND=false

    if $OAK_FOUND && [ -n "$OAK_DEVICES" ]; then
        echo "$OAK_DEVICES"
        break
    else
        echo "  (no OAK devices detected)"
        if [ -n "$OAK_DEVICES" ]; then
            echo "  depthai output: $OAK_DEVICES"
        fi
        echo ""
        read -rp "  Retry detection? [Y/n] (or 'n' to enter device IDs manually): " RETRY
        if [[ "$RETRY" =~ ^[Nn]$ ]]; then
            break
        fi
        echo ""
    fi
done
echo ""

# Build ExecStart args incrementally
CAMERA_ARGS=""

# --- Ego-view camera (required) ---
read -rp "  Ego-view camera type (oak, oak_mono, realsense, usb) [oak]: " EGO_TYPE
EGO_TYPE="${EGO_TYPE:-oak}"
read -rp "  Ego-view device ID (MxID or /dev/video index): " EGO_DEVICE_ID
CAMERA_ARGS="--ego-view-camera ${EGO_TYPE}"
if [ -n "$EGO_DEVICE_ID" ]; then
    CAMERA_ARGS="${CAMERA_ARGS} --ego-view-device-id ${EGO_DEVICE_ID}"
fi

# --- Left wrist camera (optional) ---
echo ""
read -rp "  Add a left-wrist camera? [y/N]: " ADD_LEFT
if [[ "$ADD_LEFT" =~ ^[Yy]$ ]]; then
    read -rp "  Left-wrist camera type [oak]: " LEFT_TYPE
    LEFT_TYPE="${LEFT_TYPE:-oak}"
    read -rp "  Left-wrist device ID (MxID): " LEFT_DEVICE_ID
    CAMERA_ARGS="${CAMERA_ARGS} --left-wrist-camera ${LEFT_TYPE}"
    if [ -n "$LEFT_DEVICE_ID" ]; then
        CAMERA_ARGS="${CAMERA_ARGS} --left-wrist-device-id ${LEFT_DEVICE_ID}"
    fi
fi

# --- Right wrist camera (optional) ---
echo ""
read -rp "  Add a right-wrist camera? [y/N]: " ADD_RIGHT
if [[ "$ADD_RIGHT" =~ ^[Yy]$ ]]; then
    read -rp "  Right-wrist camera type [oak]: " RIGHT_TYPE
    RIGHT_TYPE="${RIGHT_TYPE:-oak}"
    read -rp "  Right-wrist device ID (MxID): " RIGHT_DEVICE_ID
    CAMERA_ARGS="${CAMERA_ARGS} --right-wrist-camera ${RIGHT_TYPE}"
    if [ -n "$RIGHT_DEVICE_ID" ]; then
        CAMERA_ARGS="${CAMERA_ARGS} --right-wrist-device-id ${RIGHT_DEVICE_ID}"
    fi
fi

echo ""
read -rp "  ZMQ port [5555]: " CFG_PORT
CFG_PORT="${CFG_PORT:-5555}"
CAMERA_ARGS="${CAMERA_ARGS} --port ${CFG_PORT}"

EXEC_START="${REPO_ROOT}/.venv_camera/bin/python -m gear_sonic.camera.composed_camera ${CAMERA_ARGS}"
echo ""
echo "  ExecStart command:"
echo "    $EXEC_START"
echo ""
read -rp "  Look correct? [Y/n]: " CONFIRM
if [[ "$CONFIRM" =~ ^[Nn]$ ]]; then
    echo "  Aborted. Edit systemd/composed_camera_server.service manually."
    exit 0
fi

# Generate unit file directly (avoids fragile sed on multi-line ExecStart)
TMPUNIT="$(mktemp)"
cat > "$TMPUNIT" <<UNIT
[Unit]
Description=SONIC Composed Camera Server (ZMQ)
After=network.target

[Service]
Type=simple
User=$USER
Environment="HOME=$HOME"
Environment="REPO_DIR=$REPO_ROOT"
WorkingDirectory=$REPO_ROOT
ExecStart=$EXEC_START
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

echo ""
echo "[INFO] Installing systemd service …"
sudo cp "$TMPUNIT" "/etc/systemd/system/$SERVICE_NAME"
rm -f "$TMPUNIT"

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl start "$SERVICE_NAME"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  systemd service installed and started!"
echo ""
echo "  Check status:"
echo "    sudo systemctl status $SERVICE_NAME"
echo ""
echo "  View logs:"
echo "    journalctl -u $SERVICE_NAME -f"
echo ""
echo "  To reconfigure, edit and re-run this script, or:"
echo "    sudo systemctl edit $SERVICE_NAME"
echo "    sudo systemctl restart $SERVICE_NAME"
echo "══════════════════════════════════════════════════════════════"
