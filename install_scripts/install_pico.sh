#!/usr/bin/env bash
# install_pico.sh
# Sets up the gear_sonic_teleop venv for PICO VR teleop on any x86_64 or arm64
# machine (desktop, laptop, or G1 onboard).
#
# Usage:
#   bash install_scripts/install_pico.sh             # full install
#   SKIP_SIM_AND_UNITREE=1 bash install_scripts/install_pico.sh
#                                                    # publisher-only profile
#                                                    # (Thor/Orin used as a
#                                                    # headless Isaac Teleop /
#                                                    # CloudXR ROS publisher
#                                                    # — neither sim nor the
#                                                    # unitree DDS bindings
#                                                    # are on that path)
#
# Optional env vars:
#   SKIP_SIM_AND_UNITREE=1   Skip the mujoco sim extra and unitree_sdk2_python.
#                            Also skips the CycloneDDS C-lib build.
#   CYCLONEDDS_HOME=<path>   Override the CycloneDDS install prefix
#                            (default: .venv_teleop/cyclonedds).
#
# Run from the repo root.

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

# ── 2. Install a uv-managed Python 3.11 (includes dev headers / Python.h) ────
echo "[INFO] Installing uv-managed Python 3.11 (includes development headers) …"
uv python install 3.11
MANAGED_PY="$(uv python find --no-project 3.11)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ──────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_teleop (if present) …"
rm -rf .venv_teleop

# ── 4. Create venv & install teleop extra ─────────────────────────────────────
echo "[INFO] Creating .venv_teleop with uv-managed Python 3.11 …"
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

# On aarch64 (Jetson), build the PXREARobotSDK native lib from source
# because pre-built aarch64 binaries are not shipped in the repo.
XRT_DIR="$REPO_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
if [ "$ARCH" = "aarch64" ] && [ ! -f "$XRT_DIR/lib/aarch64/libPXREARobotSDK.so" ]; then
    echo "[INFO] Building PXREARobotSDK for aarch64 (Jetson) …"
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

uv pip install --no-build-isolation -e external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/

# ── 5c. Install isaacteleop[cloudxr] for the in-process CloudXR / DeviceIO path
#       (--input-source isaac-teleop in pico_manager_thread_server.py).
# Hosted on pypi.nvidia.com (public index, no auth). Replaces the legacy
# multi-container path (./scripts/run_cloudxr_via_docker.sh + teleop_ros2_ref).
echo "[INFO] Installing isaacteleop[cloudxr]~=1.4.0 from pypi.nvidia.com …"
uv pip install 'isaacteleop[cloudxr]~=1.4.0' --prerelease=allow \
    --extra-index-url https://pypi.nvidia.com

# Seed ~/cloudxr.env with the device profile CloudXRLauncher negotiates against.
# Skip if the file already exists.
if [ ! -f "$HOME/cloudxr.env" ]; then
    echo "NV_DEVICE_PROFILE=Quest3" > "$HOME/cloudxr.env"
    echo "[OK] Seeded $HOME/cloudxr.env with NV_DEVICE_PROFILE=Quest3"
else
    echo "[OK] $HOME/cloudxr.env already exists (leaving as-is)"
fi

# ── 5b, 6, 7: CycloneDDS C lib + sim extra + unitree_sdk2_python ─────────────
# Skip when:
#   • onboard unitree-provisioned image (aarch64 + user==unitree): the image
#     already ships CycloneDDS, sim has no display, and the on-robot deploy
#     uses the C++ stack directly. Applies to both Orin and Thor onboards.
#   • SKIP_SIM_AND_UNITREE=1 is set explicitly: e.g. a Thor or Orin used as a
#     headless Isaac Teleop / CloudXR streamer — neither mujoco nor the
#     unitree DDS bindings are on that path.
if { [ "$ARCH" = "aarch64" ] && [ "$(whoami)" = "unitree" ]; } \
   || [ "${SKIP_SIM_AND_UNITREE:-0}" = "1" ]; then
    echo "[SKIP] Skipping CycloneDDS build, sim extra & unitree_sdk2_python"
else
    # ── 5b. Prepare the C library for Unitree's CycloneDDS Python binding.
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/setup_cyclonedds.sh"

    # ── 6. Install sim extra (for run_sim_loop.py / sim2sim testing)
    echo "[INFO] Installing sim extra …"
    uv pip install -e "gear_sonic[sim]"

    # ── 7. Install unitree_sdk2_python (needed by the sim2sim bridge)
    echo "[INFO] Installing unitree_sdk2_python …"
    # CycloneDDS wheels embed the build-time library path.
    uv pip install --no-cache -e external_dependencies/unitree_sdk2_python
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_teleop/bin/activate"
echo ""
echo "  You should see (gear_sonic_teleop) in your prompt."
echo "══════════════════════════════════════════════════════════════"
