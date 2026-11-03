#!/bin/bash
# Install the GEM (GENMO) webcam pose-estimation environment for video teleoperation.
#
# Creates a dedicated venv inside external_dependencies/GENMO/.venv and installs
# everything needed by scripts/demo/demo_webcam.py + the ZMQ publisher.
#
# NOTE: this downloads several GB of dependencies. The ONNX models (~8.7 GB) are
# auto-downloaded from HuggingFace (nvidia/GEM-X) on the FIRST run of the demo.
# One manual step is required (SMPLX body model — registration needed), see the end.
#
# Usage (from repo root):
#   bash install_scripts/install_gem_webcam.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENMO_DIR="$REPO_ROOT/external_dependencies/GENMO"

# --- 1. Clone GENMO if missing ---
if [ ! -d "$GENMO_DIR" ]; then
    echo "Cloning GENMO..."
    git clone --depth 1 https://github.com/NVlabs/GENMO.git "$GENMO_DIR"
fi

cd "$GENMO_DIR"

# --- 2. Create venv (python 3.10 via uv) ---
if ! command -v uv &> /dev/null; then
    pip install uv
fi
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.10
fi
source .venv/bin/activate

# --- 3. PyTorch with CUDA 12.4 ---
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# --- 4. GEM-SMPL and dependencies ---
bash scripts/install_env.sh

# --- 5. ONNX Runtime (GPU) + cuDNN ---
uv pip install onnxruntime-gpu nvidia-cudnn-cu12

# --- 6. ZMQ for the publisher bridge ---
uv pip install pyzmq

# --- 7. Verify ---
python -c "import gem; print('GEM import OK')"
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())"

echo ""
echo "=============================================================="
echo " GEM webcam environment installed at:"
echo "   $GENMO_DIR/.venv"
echo ""
echo " MANUAL STEP REQUIRED (once):"
echo "   1. Register at https://smpl-x.is.tue.mpg.de/ and download"
echo "      SMPLX_NEUTRAL.npz"
echo "   2. Place it at:"
echo "      $GENMO_DIR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
echo ""
echo " On the FIRST demo run, ~8.7 GB of ONNX models are auto-"
echo " downloaded from HuggingFace (nvidia/GEM-X) into inputs/onnx/."
echo " Use --no_imgfeat to skip the HMR2 model (~2.7 GB) and run faster."
echo "=============================================================="
