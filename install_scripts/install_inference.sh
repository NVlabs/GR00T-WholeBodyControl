#!/usr/bin/env bash
# install_inference.sh
# Sets up the .venv_inference venv for running VLA inference with
# Isaac-GR00T PolicyClient against a remote or local policy server.
#
# Installs gear_sonic[inference] which pulls in the Isaac-GR00T library,
# PyZMQ, msgpack, Pinocchio, and other inference dependencies.
#
# Isaac-GR00T currently requires Python >=3.12,<3.13, so this venv is 3.12.
# Other install_* scripts stay on 3.10 because they do not install GR00T.
#
# Usage:  bash install_scripts/install_inference.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 0. System dependencies ────────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

# ── 1. Ensure uv is installed and available ─────────────────────────────
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

# ── 2. Install a uv-managed Python 3.12 (Isaac-GR00T requires >=3.12,<3.13) ──
echo "[INFO] Installing uv-managed Python 3.12 (Isaac-GR00T requires >=3.12,<3.13) …"
uv python install 3.12
MANAGED_PY="$(uv python find --no-project 3.12)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ─────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_inference (if present) …"
rm -rf .venv_inference

# ── 4. Create venv & install inference extra ────────────────────────────
echo "[INFO] Creating .venv_inference with uv-managed Python 3.12 …"
uv venv .venv_inference --python "$MANAGED_PY" --prompt gear_sonic_inference
# shellcheck disable=SC1091
source .venv_inference/bin/activate
echo "[INFO] Installing gear_sonic[inference] (this may take a few minutes) …"
uv pip install -e "gear_sonic[inference]"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_inference/bin/activate"
echo ""
echo "  You should see (gear_sonic_inference) in your prompt."
echo ""
echo "  Then run VLA inference with:"
echo "    python gear_sonic/scripts/run_vla_inference.py --help"
echo "══════════════════════════════════════════════════════════════"
