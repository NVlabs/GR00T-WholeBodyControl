#!/bin/bash

# Environment setup script to replace shell.nix functionality
# Source this file: source scripts/setup_env.sh

echo "🔧 Setting up G1 Deploy environment..."

# Run jetson_clocks on Jetson systems (bare-metal only)
if command -v jetson_clocks &> /dev/null; then
    if [ -f "/.dockerenv" ]; then
        # Inside Docker - skip (jetson_clocks needs host access)
        echo "ℹ️  Inside Docker - jetson_clocks should be run on host"
    else
        # Bare-metal Jetson - set max performance
        echo "🚀 Setting Jetson to max performance..."
        sudo jetson_clocks 2>/dev/null || echo "⚠️  jetson_clocks failed (needs sudo)"
    fi
fi

# Detect system architecture for platform-specific setup
ARCH=$(uname -m)

# Set up ONNX Runtime environment - check multiple possible locations
ONNX_RUNTIME_PATHS=(
    "/opt/onnxruntime"
    "/usr/local/onnxruntime" 
    "/usr/lib/onnxruntime"
    "$HOME/.local/onnxruntime"
    "/opt/intel/openvino/runtime/3rdparty/onnx_runtime"
)

ONNX_FOUND=false
for path in "${ONNX_RUNTIME_PATHS[@]}"; do
    if [ -d "$path" ]; then
        export onnxruntime_DIR="$path/lib/cmake/onnxruntime"
        echo "✅ ONNX Runtime found at: $path"
        ONNX_FOUND=true
        break
    fi
done

if [ "$ONNX_FOUND" = false ]; then
    # Try to use system package manager installation
    if pkg-config --exists libonnxruntime 2>/dev/null; then
        echo "✅ ONNX Runtime found via system package manager"
    else
        echo "⚠️  ONNX Runtime not found in common locations:"
        printf "   %s\n" "${ONNX_RUNTIME_PATHS[@]}"
        echo "   Please run scripts/install_deps.sh or install ONNX Runtime manually"
    fi
fi

# Auto-detect system architecture and library paths for different distributions
DISTRO_ID=""

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    DISTRO_ID=$(grep "^ID=" /etc/os-release | cut -d'=' -f2 | tr -d '"')
elif [ -f /etc/redhat-release ]; then
    DISTRO_ID="rhel"
elif [ -f /etc/debian_version ]; then
    DISTRO_ID="debian"
fi

# Set system library directory based on distribution and architecture
case "$DISTRO_ID" in
    ubuntu|debian|linuxmint)
        # Debian-based distributions use multiarch paths
        if [ "$ARCH" = "x86_64" ]; then
            SYSTEM_LIB_DIR="/usr/lib/x86_64-linux-gnu"
        elif [ "$ARCH" = "aarch64" ]; then
            SYSTEM_LIB_DIR="/usr/lib/aarch64-linux-gnu"
        else
            SYSTEM_LIB_DIR="/usr/lib"
        fi
        ;;
    fedora|rhel|centos|rocky|almalinux)
        # Red Hat-based distributions typically use lib64 for 64-bit
        if [[ "$ARCH" == "x86_64" || "$ARCH" == "aarch64" ]]; then
            SYSTEM_LIB_DIR="/usr/lib64"
        else
            SYSTEM_LIB_DIR="/usr/lib"
        fi
        ;;
    opensuse*|sles)
        # SUSE distributions also use lib64
        if [[ "$ARCH" == "x86_64" || "$ARCH" == "aarch64" ]]; then
            SYSTEM_LIB_DIR="/usr/lib64"
        else
            SYSTEM_LIB_DIR="/usr/lib"
        fi
        ;;
    arch|manjaro)
        # Arch Linux uses standard lib structure
        SYSTEM_LIB_DIR="/usr/lib"
        ;;
    *)
        # Generic fallback - try to detect best path
        if [ -d "/usr/lib/$ARCH-linux-gnu" ]; then
            SYSTEM_LIB_DIR="/usr/lib/$ARCH-linux-gnu"
        elif [ -d "/usr/lib64" ] && [[ "$ARCH" == "x86_64" || "$ARCH" == "aarch64" ]]; then
            SYSTEM_LIB_DIR="/usr/lib64"
        else
            SYSTEM_LIB_DIR="/usr/lib"
        fi
        ;;
esac

echo "🔍 Detected: $DISTRO_ID on $ARCH, using library path: $SYSTEM_LIB_DIR"

# Set up CMake paths - use dynamically detected paths
CMAKE_PATHS="$SYSTEM_LIB_DIR/cmake"

# Add ONNX Runtime path if we found one
if [ -n "$onnxruntime_DIR" ]; then
    ONNX_BASE_PATH=$(dirname $(dirname $onnxruntime_DIR))  # Remove /lib/cmake/onnxruntime to get base path
    CMAKE_PATHS="$ONNX_BASE_PATH:$CMAKE_PATHS"
fi

export CMAKE_PREFIX_PATH="$CMAKE_PATHS:$CMAKE_PREFIX_PATH"
export OPENSSL_ROOT_DIR="/usr"

# ROS2 Environment Setup - dynamically find ROS2 installation
ROS2_FOUND=false

# Common ROS2 distributions in order of preference (newest first)
ROS2_DISTROS=("jazzy" "iron" "humble" "galactic" "foxy" "eloquent" "dashing" "crystal")
ROS2_INSTALL_PATHS=("/opt/ros" "/usr/local/ros" "$HOME/ros2_ws/install")

for install_path in "${ROS2_INSTALL_PATHS[@]}"; do
    if [ "$ROS2_FOUND" = true ]; then
        break
    fi
    
    for distro in "${ROS2_DISTROS[@]}"; do
        ros2_setup_file="$install_path/$distro/setup.bash"
        if [ -f "$ros2_setup_file" ]; then
            source "$ros2_setup_file"
            export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
            # Remove problematic system library path that conflicts with system GLIBC
            export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "$SYSTEM_LIB_DIR" | tr '\n' ':' | sed 's/:$//')
            echo "✅ ROS2 $distro found at $install_path/$distro - system manages all ROS2 dependencies"
            export HAS_ROS2=1
            export ROS_LOCALHOST_ONLY=1
            ROS2_FOUND=true
            break
        fi
    done
done

if [ "$ROS2_FOUND" = false ]; then
    echo "⚠️  ROS2 not found in common locations:"
    printf "   %s/<distro>\n" "${ROS2_INSTALL_PATHS[@]}"
    echo "   Install ROS2 system-wide for ROS2InputHandler support"
    echo "   Building will continue without ROS2InputHandler"
    export HAS_ROS2=0
fi

# Set up production FastRTPS profile
if [ -f "src/g1/g1_deploy_onnx_ref/config/fastrtps_profile.xml" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$(pwd)/src/g1/g1_deploy_onnx_ref/config/fastrtps_profile.xml"
    echo "✅ FastRTPS production profile configured"
fi

# TensorRT Environment Setup
# Check if TensorRT_ROOT is already set, if not try to load from .bashrc
if [ -z "$TensorRT_ROOT" ] && [ -f "$HOME/.bashrc" ]; then
    # Extract TensorRT_ROOT from .bashrc if it exists
    BASHRC_TENSORRT=$(grep -o 'export TensorRT_ROOT=.*' "$HOME/.bashrc" | head -n1 | cut -d'=' -f2 | tr -d '"' | envsubst)
    if [ -n "$BASHRC_TENSORRT" ]; then
        export TensorRT_ROOT="$BASHRC_TENSORRT"
        echo "📋 Loaded TensorRT_ROOT from ~/.bashrc: $TensorRT_ROOT"
    fi
fi

if [ -n "$TensorRT_ROOT" ]; then
    export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$LD_LIBRARY_PATH"
    echo "✅ TensorRT environment configured"
    
    # For Jetson systems, ensure DLA libraries are accessible for runtime
    if [[ "$ARCH" == "aarch64" ]] && [[ -f "/etc/nv_tegra_release" || -d "/usr/src/jetson_multimedia_api" ]]; then
        echo "🤖 Jetson system detected - setting up DLA library paths for runtime"
        
        # Create missing libcudla.so.1 symlink if needed (TensorRT expects this name)
        if [ -f "/usr/lib/aarch64-linux-gnu/nvidia/libnvcudla.so" ] && [ ! -f "/usr/lib/aarch64-linux-gnu/nvidia/libcudla.so.1" ]; then
            echo "   🔗 Creating libcudla.so.1 symlink for TensorRT compatibility..."
            sudo ln -sf libnvcudla.so /usr/lib/aarch64-linux-gnu/nvidia/libcudla.so.1 2>/dev/null || echo "   ⚠️  Could not create symlink (may need sudo)"
            sudo ln -sf libnvcudla.so /usr/lib/aarch64-linux-gnu/nvidia/libcudla.so 2>/dev/null || echo "   ⚠️  Could not create symlink (may need sudo)"
            echo "   ✅ libcudla.so.1 → libnvcudla.so"
        fi
        
        # CRITICAL: Add DLA library path for runtime (this is why your executable can't run)
        export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH"
        echo "   📁 Added DLA library path to current session"
        
        # Make it persistent so you don't need to run setup_env.sh every time
        if ! grep -q "/usr/lib/aarch64-linux-gnu/nvidia" ~/.bashrc 2>/dev/null; then
            echo 'export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH"' >> ~/.bashrc
            echo "   ✅ Added DLA library path to ~/.bashrc for future sessions"
        fi
        
        echo "   ℹ️  Your executable should now be able to find DLA libraries at runtime"
    fi
else
    echo "⚠️  TensorRT_ROOT is not set"
    echo "   Please install TensorRT and add 'export TensorRT_ROOT=/path/to/tensorrt' to your ~/.bashrc"
    echo "   Or source ~/.bashrc before running this script"
fi

# CUDA Environment Setup
echo "🔧 Setting up CUDA environment..."

# Function to detect and set CUDA toolkit root
setup_cuda_toolkit() {
    # Check if CUDAToolkit_ROOT is already set
    if [ -n "$CUDAToolkit_ROOT" ] && [ -f "$CUDAToolkit_ROOT/bin/nvcc" ]; then
        echo "✅ CUDA toolkit found: CUDAToolkit_ROOT already set to $CUDAToolkit_ROOT"
        return 0
    fi
    
    # Check for nvcc in PATH first
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(command -v nvcc)
        CUDA_ROOT=$(dirname $(dirname "$NVCC_PATH"))
        export CUDAToolkit_ROOT="$CUDA_ROOT"
        echo "✅ CUDA toolkit found in PATH: $CUDAToolkit_ROOT"
        return 0
    fi
    
    # Check common CUDA installation paths (including CUDA 12.6 for newer Jetson)
    CUDA_PATHS=(
        "/usr/local/cuda"
        "/usr/local/cuda-12.6"  # CUDA 12.6 on newer Jetson
        "/usr/local/cuda-12.5"  # CUDA 12.5 variant
        "/usr/local/cuda-12.4"  # CUDA 12.4 variant
        "/usr/local/cuda-12"
        "/usr/local/cuda-11.4"  # Common on older Jetson
        "/usr/local/cuda-11"
        "/usr/local/cuda-10.2"  # Common on older Jetson
        "/usr/local/cuda-10"
        "/opt/cuda"
        "/usr/cuda"
    )
    
    for cuda_path in "${CUDA_PATHS[@]}"; do
        if [ -f "$cuda_path/bin/nvcc" ]; then
            export CUDAToolkit_ROOT="$cuda_path"
            export CUDA_HOME="$cuda_path"  # Alternative name CMake might use
            export PATH="$cuda_path/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_path/lib64:$cuda_path/lib:$LD_LIBRARY_PATH"
            echo "✅ CUDA toolkit found at: $CUDAToolkit_ROOT"
            echo "   Added to PATH: $cuda_path/bin"
            echo "   Added CUDA libraries to LD_LIBRARY_PATH"
            return 0
        fi
    done
    
    return 1
}

# Set up CUDA toolkit
if setup_cuda_toolkit; then
    # Add CUDA library paths
    if [ -n "$CUDAToolkit_ROOT" ]; then
        export LD_LIBRARY_PATH="$CUDAToolkit_ROOT/lib64:$CUDAToolkit_ROOT/lib:$LD_LIBRARY_PATH"
        
        # For Jetson with CUDA 12.6+, also add the targets/aarch64-linux/lib path for libcudla
        if [[ "$ARCH" == "aarch64" ]] && [ -d "$CUDAToolkit_ROOT/targets/aarch64-linux/lib" ]; then
            export LD_LIBRARY_PATH="$CUDAToolkit_ROOT/targets/aarch64-linux/lib:$LD_LIBRARY_PATH"
            echo "   ✅ Added CUDA aarch64-linux libraries (includes libcudla for TensorRT DLA support)"
        fi
    fi
else
    echo "ℹ️  CUDA toolkit (nvcc) not found. This is OK - checking for runtime libraries..."
    echo "   Note: Full toolkit is preferred for development but runtime-only is sufficient for execution"
fi

# Set up CUDA runtime libraries (fallback)
# Note: Only add nvidia path if we don't already have CUDA toolkit paths set up
# This prevents conflicts where libnvcudla.so shadows the correct libcudla.so from CUDA toolkit
if [ -z "$CUDAToolkit_ROOT" ] && [ -d "/usr/lib/aarch64-linux-gnu/nvidia/" ]; then
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia/:$LD_LIBRARY_PATH"
    echo "✅ CUDA runtime libraries found (aarch64)"
elif [ -n "$CUDAToolkit_ROOT" ]; then
    echo "✅ Using CUDA libraries from toolkit (prioritized over system runtime libs)"
else
    cuda_so_path=$(find /usr -name libcuda.so.1 2>/dev/null | head -n1)
    if [ -n "$cuda_so_path" ]; then
        export LD_PRELOAD="$cuda_so_path"
        echo "✅ CUDA runtime library found at $cuda_so_path"
    else
        echo "========================================================================"
        echo "⚠️  Warning: CUDA libraries not found. GPU functionality may be unavailable."
        echo "   Please install NVIDIA drivers and CUDA toolkit for GPU support."
        echo "========================================================================"
    fi
fi

# Add ONNX Runtime to library path if not already there
if [ -d "/opt/onnxruntime/lib" ]; then
    export LD_LIBRARY_PATH="/opt/onnxruntime/lib:$LD_LIBRARY_PATH"
fi

# Set up Git LFS (if not already done)
if command -v git-lfs &> /dev/null; then
    git lfs install &> /dev/null
    echo "✅ Git LFS configured"
    
    # Pull large files if in git repository
    if [ -d ".git" ]; then
        echo "📥 Pulling Git LFS files..."
        git lfs pull
    fi
else
    echo "⚠️  Git LFS not found. Please install git-lfs package."
fi

# Verify essential tools
echo ""
echo "🔍 Verifying essential tools:"

check_command() {
    if command -v "$1" &> /dev/null; then
        echo "✅ $1: $(command -v $1)"
    else
        echo "❌ $1: not found"
    fi
}

check_command cmake
check_command clang
check_command just
check_command git

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "📝 You can now run:"
echo "   just build    # Build the project"
echo "   just --list   # See all available commands"
echo ""

# Optional: Add this setup to the current shell session persistence
if [ -n "$BASH_VERSION" ]; then
    export PS1="(g1_deploy) $PS1"
fi

