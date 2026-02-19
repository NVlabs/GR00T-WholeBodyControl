#!/usr/bin/env bash
# install_mujoco_sim.sh
# Minimal venv setup for running the MuJoCo simulator (run_sim_loop.py).
# Skips XRoboToolkit SDK and teleop dependencies that are NOT needed for sim.
# Based on install_pico.sh — see that script for the full teleop setup.
#
# Usage:  bash install_scripts/install_mujoco_sim.sh   (run from repo root)

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
echo "[INFO] Removing old .venv_sim (if present) …"
rm -rf .venv_sim

# ── 4. Create venv & install sim extra ────────────────────────────────────────
echo "[INFO] Creating .venv_sim with uv-managed Python 3.10 …"
uv venv .venv_sim --python "$MANAGED_PY" --prompt gear_sonic_sim
# shellcheck disable=SC1091
source .venv_sim/bin/activate
echo "[INFO] Installing gear_sonic[sim] …"
uv pip install -e "gear_sonic[sim]"

# ── 5. Install unitree_sdk2_python (needed by the sim ↔ WBC bridge) ──────────
echo "[INFO] Installing unitree_sdk2_python …"
uv pip install -e external_dependencies/unitree_sdk2_python

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_sim/bin/activate"
echo ""
echo "  You should see (gear_sonic_sim) in your prompt."
echo ""
echo "  Then run the MuJoCo simulator with:"
echo "    python gear_sonic/scripts/run_sim_loop.py"
echo "══════════════════════════════════════════════════════════════"
