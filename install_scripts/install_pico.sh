#!/usr/bin/env bash
# install_pico.sh
# Sets up the gear_sonic_teleop venv for PICO VR teleop on any x86_64 or arm64
# machine (desktop, laptop, or G1 onboard).
# Usage:  bash install_scripts/install_pico.sh   (run from repo root)

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

    # Source the uv env so it's available in this session
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Verify uv is now reachable
    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 (includes dev headers / Python.h) ────
echo "[INFO] Installing uv-managed Python 3.10 (includes development headers) …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ──────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_teleop (if present) …"
rm -rf .venv_teleop

# ── 4. Create venv & install teleop extra ─────────────────────────────────────
echo "[INFO] Creating .venv_teleop with uv-managed Python 3.10 …"
uv venv .venv_teleop --python "$MANAGED_PY" --prompt gear_sonic_teleop
# shellcheck disable=SC1091
source .venv_teleop/bin/activate
echo "[INFO] Installing gear_sonic[teleop] …"
uv pip install -e "gear_sonic[teleop]"

# ── 5. Install xrobotoolkit_sdk (CMake-based, not a pip package) ──────────────
echo "[INFO] Installing XRoboToolkit SDK …"
# Install cmake + pybind11 into the venv so the CMake-based build can find them.
# Build with --no-build-isolation so CMake inherits the venv's pybind11.
uv pip install cmake pybind11 setuptools
echo "[OK] cmake $(cmake --version | head -1)"
# Point CMake at pybind11's cmake config so find_package(pybind11) succeeds
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
echo "[OK] pybind11 cmake dir: $CMAKE_PREFIX_PATH"

XRT_REPO="https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind"
if [ "$ARCH" = "aarch64" ]; then
    # On aarch64 (Jetson Orin), clone the repo locally so we can build the
    # PXREARobotSDK native lib from source (no pre-built aarch64 binary available).
    XRT_DIR="$REPO_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind"
    if [ ! -d "$XRT_DIR" ]; then
        echo "[INFO] Cloning XRoboToolkit-PC-Service-Pybind …"
        git clone "${XRT_REPO}.git" "$XRT_DIR"
    fi
    if [ ! -f "$XRT_DIR/lib/aarch64/libPXREARobotSDK.so" ]; then
        echo "[INFO] Building PXREARobotSDK for aarch64 (Jetson Orin) …"
        XRT_TMP="$XRT_DIR/tmp"
        mkdir -p "$XRT_TMP"
        if [ ! -d "$XRT_TMP/XRoboToolkit-PC-Service" ]; then
            git clone -b orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git "$XRT_TMP/XRoboToolkit-PC-Service"
        fi
        pushd "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK" > /dev/null
        bash build.sh
        popd > /dev/null
        mkdir -p "$XRT_DIR/lib/aarch64" "$XRT_DIR/include/aarch64"
        cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" \
           "$XRT_DIR/include/aarch64/"
        cp -r "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann" \
           "$XRT_DIR/include/aarch64/nlohmann/"
        cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" \
           "$XRT_DIR/lib/aarch64/"
        rm -rf "$XRT_TMP"
        echo "[OK] PXREARobotSDK aarch64 native library built and installed"
    fi
    uv pip install --no-build-isolation -e "$XRT_DIR"
else
    # x86_64: prebuilt .so is included in the repo; install directly from GitHub.
    echo "[INFO] Installing XRoboToolkit-PC-Service-Pybind from GitHub …"
    uv pip install --no-build-isolation "git+${XRT_REPO}.git"
fi

# ── 6 & 7: sim extra + unitree_sdk2_python ────────────────────────────────────
# On the onboard Jetson Orin (user==unitree + aarch64) these are not needed:
#   • sim extra depends on mujoco which may lack aarch64 wheels
#   • unitree_sdk2_python requires CycloneDDS C lib (already on the robot)
# They are only installed on desktop / x86 dev machines.
if [ "$ARCH" = "aarch64" ] && [ "$(whoami)" = "unitree" ]; then
    echo "[SKIP] Skipping sim extra & unitree_sdk2_python (onboard Jetson Orin)"
else
    # ── 6. Install sim extra (for run_sim_loop.py / sim2sim testing)
    echo "[INFO] Installing sim extra …"
    uv pip install -e "gear_sonic[sim]"

    # ── 7. Install unitree_sdk2_python (needed by the sim2sim bridge)
    echo "[INFO] Installing unitree_sdk2_python …"
    uv pip install -e external_dependencies/unitree_sdk2_python
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_teleop/bin/activate"
echo ""
echo "  You should see (gear_sonic_teleop) in your prompt."
echo "══════════════════════════════════════════════════════════════"
