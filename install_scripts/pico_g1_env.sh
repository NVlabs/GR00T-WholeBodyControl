#!/usr/bin/env bash
# Source this file before building or running the PICO + G1 EDU profile.

if [ -d /usr/local/cuda ]; then
    _PICO_G1_CUDA_ROOT=/usr/local/cuda
elif [ -d /usr/local/cuda-12.9 ]; then
    _PICO_G1_CUDA_ROOT=/usr/local/cuda-12.9
else
    echo "[ERROR] CUDA toolkit not found under /usr/local" >&2
    return 1
fi

export CUDAToolkit_ROOT="$_PICO_G1_CUDA_ROOT"
export CUDA_HOME="$CUDAToolkit_ROOT"
export PATH="$HOME/.local/bin:$CUDAToolkit_ROOT/bin:$PATH"

if [ -n "${TensorRT_ROOT:-}" ] \
    && [ -f "$TensorRT_ROOT/include/NvInferVersion.h" ]; then
    # Honor an explicitly selected TAR installation.
    export TensorRT_ROOT
elif [ -f /usr/include/x86_64-linux-gnu/NvInferVersion.h ] \
    || [ -f /usr/include/aarch64-linux-gnu/NvInferVersion.h ]; then
    export TensorRT_ROOT=/usr
elif [ -f "$HOME/TensorRT/include/NvInferVersion.h" ]; then
    export TensorRT_ROOT="$HOME/TensorRT"
else
    echo "[ERROR] TensorRT headers not found" >&2
    return 1
fi

_PICO_G1_LIBRARY_PATH="$CUDAToolkit_ROOT/lib64"
if [ -d /usr/lib/x86_64-linux-gnu ]; then
    _PICO_G1_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$_PICO_G1_LIBRARY_PATH"
elif [ -d /usr/lib/aarch64-linux-gnu ]; then
    _PICO_G1_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$_PICO_G1_LIBRARY_PATH"
fi
if [ -d "$TensorRT_ROOT/lib" ]; then
    # Prefer the explicitly selected TensorRT release over any system copy.
    _PICO_G1_LIBRARY_PATH="$TensorRT_ROOT/lib:$_PICO_G1_LIBRARY_PATH"
fi
if [ -d /opt/onnxruntime/lib ]; then
    _PICO_G1_LIBRARY_PATH="/opt/onnxruntime/lib:$_PICO_G1_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$_PICO_G1_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"

# This profile materializes only its required LFS assets.
export SKIP_GIT_LFS_PULL=1

unset _PICO_G1_CUDA_ROOT
unset _PICO_G1_LIBRARY_PATH
