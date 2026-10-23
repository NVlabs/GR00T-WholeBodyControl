#!/usr/bin/env bash
# install_data_collection.sh
# Sets up the .venv_data_collection venv for recording teleop demonstrations
# in LeRobot dataset format (for post-training with Isaac-GR00T).
#
# Installs gear_sonic[data_collection] which pulls in LeRobot, PyAV, OpenCV,
# and the other dependencies needed by run_data_exporter.py.
#
# Usage:  bash install_scripts/install_data_collection.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 0. System dependencies ────────────────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

echo "[INFO] Installing system dependencies (espeak for voice feedback) …"
if command -v apt-get &>/dev/null; then
    sudo apt-get install -y espeak >/dev/null 2>&1 || echo "[WARN] Could not install espeak — voice feedback will be disabled"
fi

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
echo "[INFO] Removing old .venv_data_collection (if present) …"
rm -rf .venv_data_collection

# ── 4. Create venv & install data_collection extra ───────────────────────────
echo "[INFO] Creating .venv_data_collection with uv-managed Python 3.11 …"
uv venv .venv_data_collection --python "$MANAGED_PY" --prompt gear_sonic_data_collection
# shellcheck disable=SC1091
source .venv_data_collection/bin/activate
echo "[INFO] Installing gear_sonic[data_collection] (this may take a few minutes) …"
# LeRobot's git repo contains LFS test artifacts that aren't needed at runtime.
# Skip them to avoid download failures and save bandwidth.
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e "gear_sonic[data_collection]"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_data_collection/bin/activate"
echo ""
echo "  You should see (gear_sonic_data_collection) in your prompt."
echo ""
echo "  Then run the data exporter with:"
echo "    python gear_sonic/scripts/run_data_exporter.py"
echo "══════════════════════════════════════════════════════════════"
