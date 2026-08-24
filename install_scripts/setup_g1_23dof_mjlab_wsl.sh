#!/usr/bin/env bash
#
# Build the isolated CUDA 12.8 MJLab environment used by the exact SONIC
# true-23 training launcher. Run this inside Ubuntu/WSL2, never on the robot.

set -euo pipefail

unitree_commit="1425b15f73bd4095f0df53709d7c389c3eb9e790"
mjlab_commit="5af32e378dcb93c9e881ace83cc5a3f5d373fe60"
mujoco_warp_commit="5a86ec28aa07741eb2e000d158f4ca4068ec146e"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
checkout_root="${repo_root}/external_dependencies"
unitree_checkout="${checkout_root}/unitree_rl_mjlab"
mjlab_checkout="${checkout_root}/mjlab"
task_venv="${G1_MJLAB_VENV_PATH:-${HOME}/.venvs/g1_true23_mjlab}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: run this script inside Ubuntu/WSL2." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: WSL cannot see the NVIDIA GPU. Update the Windows NVIDIA driver." >&2
  exit 2
fi

if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  uv_bin="${HOME}/.local/bin/uv"
else
  echo "ERROR: uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 2
fi

mkdir -p -- "${checkout_root}"

ensure_pinned_checkout() {
  local repository_url="$1"
  local checkout_path="$2"
  local expected_commit="$3"

  if [[ ! -d "${checkout_path}/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
      "${repository_url}" "${checkout_path}"
    git -C "${checkout_path}" checkout --detach "${expected_commit}"
  fi

  local actual_commit
  actual_commit="$(git -C "${checkout_path}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "ERROR: ${checkout_path} is at ${actual_commit}; expected ${expected_commit}." >&2
    echo "Move that checkout aside, then rerun. This script will not reset it." >&2
    exit 2
  fi
  # The launcher hashes every relevant working source/asset into its immutable
  # lineage. Do not use `git status` here: WSL Git sees Windows CRLF checkouts
  # as globally dirty even when the content is the intended pinned revision.
}

ensure_pinned_checkout \
  "https://github.com/unitreerobotics/unitree_rl_mjlab.git" \
  "${unitree_checkout}" \
  "${unitree_commit}"
ensure_pinned_checkout \
  "https://github.com/mujocolab/mjlab.git" \
  "${mjlab_checkout}" \
  "${mjlab_commit}"

"${uv_bin}" python install 3.11
"${uv_bin}" venv --python 3.11 "${task_venv}"

# The released MJLab 1.2 lock references a now-unavailable MuJoCo development
# wheel. Stable MuJoCo 3.5.0 is the explicit compatibility substitution.
# The exact MuJoCo-Warp source commit from the same lock remains authoritative.
"${uv_bin}" pip install \
  --python "${task_venv}/bin/python" \
  --default-index "https://pypi.org/simple" \
  --index "https://download.pytorch.org/whl/cu128" \
  --index "https://pypi.nvidia.com/" \
  --index-strategy unsafe-best-match \
  "torch==2.9.0+cu128" \
  "torchvision==0.24.0+cu128" \
  "warp-lang==1.12.0" \
  "mujoco==3.5.0" \
  "numpy==2.3.4" \
  "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp.git@${mujoco_warp_commit}"

"${uv_bin}" pip install \
  --python "${task_venv}/bin/python" \
  "prettytable" \
  "tqdm" \
  "tyro>=1.0.1" \
  "torchrunx>=0.3.4" \
  "trimesh>=4.8.3" \
  "viser>=1.0.24" \
  "mediapy>=1.2.6" \
  "imageio-ffmpeg" \
  "scipy==1.16.2" \
  "tensordict==0.10.0" \
  "rsl-rl-lib==5.0.1" \
  "tensorboard>=2.20.0" \
  "onnxscript>=0.5.4" \
  "wandb>=0.22.3" \
  "pytest" \
  "ruff"

# Install exact source trees without allowing Unitree's stale
# mujoco-warp==3.5.0 metadata pin to replace MJLab's locked git commit.
"${uv_bin}" pip install \
  --python "${task_venv}/bin/python" \
  --no-deps \
  --editable "${mjlab_checkout}"
"${uv_bin}" pip install \
  --python "${task_venv}/bin/python" \
  --no-deps \
  --editable "${unitree_checkout}"

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

"${task_venv}/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path

import mujoco
import mjlab
import torch
import warp

assert torch.__version__ == "2.9.0+cu128", torch.__version__
assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert torch.cuda.get_device_properties(0).total_memory > 0
assert mujoco.__version__ == "3.5.0", mujoco.__version__
assert metadata.version("warp-lang") == "1.12.0"
assert metadata.version("mjlab") == "1.2.0"
assert metadata.version("scipy") == "1.16.2"
assert metadata.version("tensordict") == "0.10.0"

dist = metadata.distribution("mujoco-warp")
direct_url_path = Path(dist._path) / "direct_url.json"  # type: ignore[attr-defined]
direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
commit = direct_url.get("vcs_info", {}).get("commit_id")
assert commit == "5a86ec28aa07741eb2e000d158f4ca4068ec146e", commit

print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.get_device_name(0))
print("mujoco", mujoco.__version__)
print("mujoco-warp", metadata.version("mujoco-warp"), commit)
print("mjlab", metadata.version("mjlab"))
print("scipy", metadata.version("scipy"))
print("tensordict", metadata.version("tensordict"))
print("warp", warp.__version__)
PY

echo
echo "MJLab true-23 environment ready: ${task_venv}"
echo "Next:"
echo "  ${task_venv}/bin/python ${repo_root}/gear_sonic/scripts/train_g1_23dof_mjlab.py convert-sample"
echo "  ${task_venv}/bin/python ${repo_root}/gear_sonic/scripts/train_g1_23dof_mjlab.py preflight"
