#!/bin/bash

set -e

echo "🚀 Installing G1 Deploy system dependencies..."
sudo apt-get update && sudo apt-get install -y libgtest-dev

# Detect system type and architecture
ARCH=$(uname -m)
OS_ID=$(lsb_release -si 2>/dev/null || echo "Unknown")
OS_VERSION=$(lsb_release -sr 2>/dev/null || echo "Unknown")

# Common CUDA installation paths (used by multiple functions)
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-12.6"  # CUDA 12.6 on newer Jetson
    "/usr/local/cuda-12.5"  # CUDA 12.5 variant  
    "/usr/local/cuda-12.4"  # CUDA 12.4 variant
    "/usr/local/cuda-12"    # Generic CUDA 12.x
    "/usr/local/cuda-11.4"  # Common on older Jetson
    "/usr/local/cuda-10.2"  # Common on older Jetson
    "/usr/local/cuda-11"
    "/usr/local/cuda-10"
    "/opt/cuda"
    "/usr/cuda"
)

# Common CUDA runtime library paths (used by multiple functions)
CUDA_RUNTIME_PATHS=(
    "/usr/local/cuda/lib64/libcudart.so"
    "/usr/local/cuda-12.6/lib64/libcudart.so"
    "/usr/local/cuda-12.5/lib64/libcudart.so"
    "/usr/local/cuda-12.4/lib64/libcudart.so"
    "/usr/local/cuda-12/lib64/libcudart.so"
    "/usr/local/cuda-11.4/lib64/libcudart.so"
    "/usr/local/cuda-10.2/lib64/libcudart.so"
    "/usr/lib/aarch64-linux-gnu/libcudart.so"
    "/usr/lib/x86_64-linux-gnu/libcudart.so" 
    "/usr/lib64/libcudart.so"
    "/usr/lib/libcudart.so"
)

# Common CUDA header paths (used by multiple functions)
CUDA_HEADER_PATHS=(
    "/usr/local/cuda/include/cuda_runtime.h"
    "/usr/local/cuda-12.6/include/cuda_runtime.h"
    "/usr/local/cuda-12.5/include/cuda_runtime.h"
    "/usr/local/cuda-12.4/include/cuda_runtime.h"
    "/usr/local/cuda-12/include/cuda_runtime.h"
    "/usr/local/cuda-11.4/include/cuda_runtime.h"
    "/usr/local/cuda-10.2/include/cuda_runtime.h"
    "/usr/local/cuda-11/include/cuda_runtime.h"
    "/usr/include/cuda_runtime.h"
    "/usr/include/cuda/cuda_runtime.h"
)

# Common utility functions
has_cuda_in_ldconfig() {
    ldconfig -p 2>/dev/null | grep -q "libcudart\|libcuda"
}

# Update package cache (with deduplication)
update_package_cache() {
    if [ "$PACKAGE_MANAGER" = "apt" ] && [ "$APT_UPDATED" != "true" ]; then
        sudo apt-get update
        export APT_UPDATED="true"
    fi
}

# Enhanced Jetson detection
IS_JETSON=false
JETSON_MODEL=""

if [[ "$ARCH" == "aarch64" ]]; then
    # Check for Jetson-specific indicators
    if [[ -f "/etc/nv_tegra_release" ]] || \
       [[ -d "/usr/src/jetson_multimedia_api" ]] || \
       ls /etc/apt/sources.list.d/ 2>/dev/null | grep -q jetson || \
       [[ -f "/proc/device-tree/model" && $(cat /proc/device-tree/model 2>/dev/null) =~ "Jetson" ]]; then
        IS_JETSON=true
        
        # Detect Jetson model for optimized package selection
        if [[ -f "/proc/device-tree/model" ]]; then
            JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' | sed 's/NVIDIA //')
            echo "🤖 Jetson system detected: $JETSON_MODEL"
        else
            echo "🤖 Jetson system detected (model unknown)"
        fi
        
        # Check JetPack version if available
        if command -v jetson_release &> /dev/null; then
            JETPACK_VERSION=$(jetson_release -v 2>/dev/null | grep "JETPACK" | cut -d' ' -f2 || echo "unknown")
            echo "📦 JetPack version: $JETPACK_VERSION"
        fi
    fi
fi

echo "🔍 System: $OS_ID $OS_VERSION ($ARCH)$([ "$IS_JETSON" = true ] && echo " - Jetson")"

# Detect the operating system and package manager
if command -v apt-get &> /dev/null; then
    PACKAGE_MANAGER="apt"
elif command -v yum &> /dev/null; then
    PACKAGE_MANAGER="yum"
elif command -v pacman &> /dev/null; then
    PACKAGE_MANAGER="pacman"
else
    echo "❌ Unsupported package manager. This script supports apt, yum, and pacman."
    exit 1
fi

# Function to install packages based on the package manager
install_packages() {
    case $PACKAGE_MANAGER in
        apt)
            update_package_cache
            sudo apt-get install -y "$@"
            ;;
        yum)
            sudo yum install -y "$@"
            ;;
        pacman)
            sudo pacman -S --noconfirm "$@"
            ;;
    esac
}

echo "📦 Installing base development tools..."
case $PACKAGE_MANAGER in
    apt)
        if [ "$IS_JETSON" = true ]; then
            echo "🤖 Installing packages optimized for Jetson system..."
            # Jetson systems may have some tools pre-installed
            JETSON_PACKAGES="build-essential clang cmake git git-lfs pkg-config patchelf zlib1g-dev curl wget"
            
            # Add optional packages if available
            if apt-cache show cmake-format &>/dev/null; then
                JETSON_PACKAGES="$JETSON_PACKAGES cmake-format"
            fi
            if apt-cache show cppcheck &>/dev/null; then
                JETSON_PACKAGES="$JETSON_PACKAGES cppcheck"
            fi
            
            install_packages $JETSON_PACKAGES
        else
        install_packages \
            build-essential \
            clang \
            cmake \
            cmake-format \
            cppcheck \
            git \
            git-lfs \
            pkg-config \
            patchelf \
            zlib1g-dev \
            curl \
            wget
        fi
        ;;
    yum)
        install_packages \
            gcc \
            gcc-c++ \
            clang \
            cmake \
            cppcheck \
            git \
            git-lfs \
            pkgconfig \
            patchelf \
            zlib-devel \
            curl \
            wget
        ;;
    pacman)
        install_packages \
            base-devel \
            clang \
            cmake \
            cppcheck \
            git \
            git-lfs \
            pkgconf \
            patchelf \
            zlib \
            curl \
            wget
        ;;
esac

echo "📚 Installing C++ libraries..."
case $PACKAGE_MANAGER in
    apt)
        install_packages \
            libyaml-cpp-dev \
            libeigen3-dev \
            libmsgpack-dev \
            libzmq3-dev \
            nlohmann-json3-dev \
            libgtest-dev
        ;;
    yum)
        install_packages \
            yaml-cpp-devel \
            eigen3-devel \
            msgpack-devel \
            zeromq-devel \
            cppzmq-devel \
            nlohmann-json-devel \
            gtest-devel
        ;;
    pacman)
        install_packages \
            yaml-cpp \
            eigen \
            msgpack-cxx \
            zeromq \
            cppzmq \
            nlohmann-json \
            gtest
        ;;
esac

# Ensure ZeroMQ C++ headers (cppzmq) are available.
# On some Ubuntu releases (e.g., 24.04), the 'cppzmq' or 'libcppzmq-dev' package may be missing.
# If not found, vendor the header-only repo to third_party/cppzmq.
echo "🧩 Ensuring ZeroMQ C++ headers (cppzmq)..."

# Resolve repository root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY_DIR="$REPO_ROOT/third_party"

have_cppzmq_header=false
if [ -f "/usr/include/zmq.hpp" ]; then
    have_cppzmq_header=true
fi

if [ "$have_cppzmq_header" = false ]; then
    if [ "$PACKAGE_MANAGER" = "apt" ]; then
        # Try distro packages if available
        if apt-cache policy cppzmq 2>/dev/null | grep -q "Candidate:" && \
           ! apt-cache policy cppzmq 2>/dev/null | grep -q "(none)"; then
            update_package_cache
            if sudo apt-get install -y cppzmq; then
                have_cppzmq_header=true
            fi
        elif apt-cache policy libcppzmq-dev 2>/dev/null | grep -q "Candidate:" && \
             ! apt-cache policy libcppzmq-dev 2>/dev/null | grep -q "(none)"; then
            update_package_cache
            if sudo apt-get install -y libcppzmq-dev; then
                have_cppzmq_header=true
            fi
        fi
    fi
fi

if [ "$have_cppzmq_header" = false ]; then
    echo "📦 Vendoring header-only cppzmq into third_party..."
    mkdir -p "$THIRD_PARTY_DIR"
    if [ ! -d "$THIRD_PARTY_DIR/cppzmq/.git" ]; then
        if git clone --depth 1 https://github.com/zeromq/cppzmq.git "$THIRD_PARTY_DIR/cppzmq"; then
            echo "✅ cppzmq headers cloned to $THIRD_PARTY_DIR/cppzmq"
            echo "   Add include path: -I$THIRD_PARTY_DIR/cppzmq"
        else
            echo "❌ Failed to clone cppzmq repo. Please install headers manually."
        fi
    else
        if git -C "$THIRD_PARTY_DIR/cppzmq" pull --ff-only; then
            echo "✅ cppzmq headers updated in $THIRD_PARTY_DIR/cppzmq"
        else
            echo "⚠️  Failed to update cppzmq headers. Using existing copy."
        fi
    fi
else
    echo "✅ ZeroMQ C++ headers found in system includes"
fi

# Ensure JSON headers (nlohmann/json) are available (similar to cppzmq ensure)
echo "🧩 Ensuring JSON headers (nlohmann/json)..."

have_json_header=false
if [ -f "/usr/include/nlohmann/json.hpp" ] || [ -f "/usr/local/include/nlohmann/json.hpp" ]; then
    have_json_header=true
fi

if [ "$have_json_header" = false ]; then
    case $PACKAGE_MANAGER in
        apt)
            update_package_cache
            sudo apt-get install -y nlohmann-json3-dev || true
            ;;
        yum)
            sudo yum install -y nlohmann-json-devel || true
            ;;
        pacman)
            sudo pacman -S --noconfirm nlohmann-json || true
            ;;
    esac
    if [ -f "/usr/include/nlohmann/json.hpp" ] || [ -f "/usr/local/include/nlohmann/json.hpp" ]; then
        have_json_header=true
    fi
fi

if [ "$have_json_header" = false ]; then
    echo "📦 Vendoring nlohmann/json into third_party..."
    # Reuse THIRD_PARTY_DIR from earlier cppzmq ensure block if set; otherwise define
    if [ -z "$THIRD_PARTY_DIR" ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
        THIRD_PARTY_DIR="$REPO_ROOT/third_party"
    fi
    mkdir -p "$THIRD_PARTY_DIR"
    if [ ! -d "$THIRD_PARTY_DIR/nlohmann_json/.git" ]; then
        if git clone --depth 1 https://github.com/nlohmann/json.git "$THIRD_PARTY_DIR/nlohmann_json"; then
            echo "✅ nlohmann/json cloned to $THIRD_PARTY_DIR/nlohmann_json"
            echo "   Add include path: -I$THIRD_PARTY_DIR/nlohmann_json/single_include"
        else
            echo "❌ Failed to clone nlohmann/json. Please install headers manually."
        fi
    else
        if git -C "$THIRD_PARTY_DIR/nlohmann_json" pull --ff-only; then
            echo "✅ nlohmann/json updated in $THIRD_PARTY_DIR/nlohmann_json"
        else
            echo "⚠️  Failed to update nlohmann/json. Using existing copy."
        fi
    fi
else
    echo "✅ JSON headers found in system includes"
fi

# Install Just command runner
echo "⚡ Installing Just command runner..."
if ! command -v just &> /dev/null; then
    case $PACKAGE_MANAGER in
        apt|yum)
            # Install Just from GitHub releases for Ubuntu/Debian and RHEL/CentOS
            JUST_VERSION="1.43.0"
            JUST_ARCH=$(uname -m)
            
            # Map architecture names to just release naming
            case $JUST_ARCH in
                x86_64)
                    JUST_ARCH_NAME="x86_64-unknown-linux-musl"
                    ;;
                aarch64|arm64)
                    JUST_ARCH_NAME="aarch64-unknown-linux-musl"
                    ;;
                armv7l)
                    JUST_ARCH_NAME="armv7-unknown-linux-musleabihf"
                    ;;
                arm*)
                    JUST_ARCH_NAME="arm-unknown-linux-musleabihf"
                    ;;
                *)
                    echo "❌ Unsupported architecture for just: $JUST_ARCH"
                    echo "   Supported architectures: x86_64, aarch64, armv7l, arm"
                    exit 1
                    ;;
            esac
            
            echo "📥 Downloading Just ${JUST_VERSION} for ${JUST_ARCH}..."
            JUST_URL="https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-${JUST_ARCH_NAME}.tar.gz"
            curl -L "$JUST_URL" | tar xz -C /tmp
            sudo mv /tmp/just /usr/local/bin/
            ;;
        pacman)
            install_packages just
            ;;
    esac
else
    echo "✅ Just is already installed"
fi

# Install ONNX Runtime
echo "🧠 Setting up ONNX Runtime..."

# Allow configurable installation path
ONNX_INSTALL_PATH="${ONNX_INSTALL_PATH:-/opt/onnxruntime}"

if [ ! -d "$ONNX_INSTALL_PATH" ]; then
    ONNX_VERSION="1.16.3"
    ONNX_ARCH=$(uname -m)
    
    if [ "$ONNX_ARCH" = "x86_64" ]; then
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
    elif [ "$ONNX_ARCH" = "aarch64" ]; then
        # Same URL for all ARM64 systems (Jetson and regular ARM64)
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz"
        if [ "$IS_JETSON" = true ]; then
            echo "🤖 Jetson system detected - using ARM64 ONNX Runtime..."
        fi
    else
        echo "❌ Unsupported architecture: $ONNX_ARCH"
        exit 1
    fi
    
    echo "📥 Downloading ONNX Runtime ${ONNX_VERSION} for ${ONNX_ARCH}$([ "$IS_JETSON" = true ] && echo " (Jetson-optimized)")..."
    cd /tmp
    
    # Download with retry for Jetson systems (sometimes have slower network)
    if [ "$IS_JETSON" = true ]; then
        curl -L --retry 3 --retry-delay 2 "$ONNX_URL" -o onnxruntime.tgz
    else
    curl -L "$ONNX_URL" -o onnxruntime.tgz
    fi
    
    tar xzf onnxruntime.tgz
    
    # Move to configured location (may need sudo for system paths)
    ONNX_DIR=$(ls -d onnxruntime-linux-* | head -n1)
    
    if [[ "$ONNX_INSTALL_PATH" =~ ^(/opt|/usr) ]]; then
        # System path - needs sudo
        sudo mv "$ONNX_DIR" "$ONNX_INSTALL_PATH"
        sudo ln -sf "$ONNX_INSTALL_PATH/lib/libonnxruntime.so" /usr/local/lib/ 2>/dev/null || true
        sudo ln -sf "$ONNX_INSTALL_PATH/include" /usr/local/include/onnxruntime 2>/dev/null || true
        echo "$ONNX_INSTALL_PATH/lib" | sudo tee /etc/ld.so.conf.d/onnxruntime.conf
        sudo ldconfig
    else
        # User path - no sudo needed
        mv "$ONNX_DIR" "$ONNX_INSTALL_PATH"
        # Create user-accessible symlinks if possible
        if [ -w /usr/local/lib ]; then
            ln -sf "$ONNX_INSTALL_PATH/lib/libonnxruntime.so" /usr/local/lib/ 2>/dev/null || true
        fi
    fi
    
    # For Jetson systems, add to environment
    if [ "$IS_JETSON" = true ]; then
        echo "# ONNX Runtime for Jetson" | sudo tee -a /etc/environment
        echo "LD_LIBRARY_PATH=\"$ONNX_INSTALL_PATH/lib:\$LD_LIBRARY_PATH\"" | sudo tee -a /etc/environment
    fi
    
    echo "✅ ONNX Runtime installed to $ONNX_INSTALL_PATH$([ "$IS_JETSON" = true ] && echo " with Jetson optimizations")"
else
    echo "✅ ONNX Runtime is already installed at $ONNX_INSTALL_PATH"
fi

echo "💡 To install to a different location, set ONNX_INSTALL_PATH environment variable"

# Function to check for CUDA toolkit (nvcc compiler) - silent check
check_cuda_toolkit() {
    # Check for nvcc in PATH
    if command -v nvcc &> /dev/null; then
        return 0
    fi
    
    # Check common CUDA installation paths (using shared variable)
    
    for cuda_path in "${CUDA_PATHS[@]}"; do
        if [ -f "$cuda_path/bin/nvcc" ]; then
            return 0
        fi
    done
    
    return 1
}

# Function to check for CUDA runtime libraries AND headers (needed for compilation)
check_cuda_runtime() {
    # Check for CUDA runtime libraries
    local has_libraries=false
    if has_cuda_in_ldconfig; then
        has_libraries=true
    else
        # Check specific paths for runtime libraries (using shared variable)
        for lib_path in "${CUDA_RUNTIME_PATHS[@]}"; do
            if [ -f "$lib_path" ]; then
                has_libraries=true
                break
            fi
        done
    fi
    
    # Check for CUDA headers (essential for compilation) - using shared variable
    local has_headers=false
    for header_path in "${CUDA_HEADER_PATHS[@]}"; do
        if [ -f "$header_path" ]; then
            has_headers=true
            break
        fi
    done
    
    # Need both libraries and headers
    if [ "$has_libraries" = true ] && [ "$has_headers" = true ]; then
        return 0
    else
        return 1
    fi
}

# Check what CUDA components we have and provide clear, non-confusing output
echo "🔍 Checking CUDA installation..."


# Function to handle non-Jetson ARM64 systems  
install_cuda_for_arm64() {
    echo "🔧 Installing CUDA packages for ARM64 system..."
    
    if sudo apt-get install -y cuda-cudart-dev-12-2 cuda-runtime-12-2 2>/dev/null || \
       sudo apt-get install -y cuda-cudart-dev-11-8 cuda-runtime-11-8 2>/dev/null || \
       sudo apt-get install -y libcudart-dev nvidia-cuda-runtime-cu12 2>/dev/null; then
        echo "✅ CUDA runtime and development packages installed for ARM64!"
        return 0
    else
        echo "❌ Failed to install CUDA packages for ARM64 system"
        return 1
    fi
}

# Function to handle x86_64 systems
install_cuda_for_x86_64() {
    echo "🔧 Installing CUDA packages for x86_64 system..."
    
    # Detect CUDA version from nvidia-smi if available
    local DETECTED_CUDA_VERSION=""
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_CUDA_VERSION=$(nvidia-smi | grep "CUDA Version:" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/' | head -n1)
        if [ -n "$DETECTED_CUDA_VERSION" ]; then
            echo "📍 Detected CUDA $DETECTED_CUDA_VERSION from nvidia-smi"
        fi
    fi
    
    # Check if NVIDIA CUDA repository is configured
    local has_cuda_repo=false
    if [ -f /etc/apt/sources.list.d/cuda*.list ] || \
       grep -r "developer.download.nvidia.com/compute/cuda" /etc/apt/sources.list* &>/dev/null; then
        has_cuda_repo=true
    fi
    
    # Add NVIDIA CUDA repository if not present
    if [ "$has_cuda_repo" = false ]; then
        echo "📦 NVIDIA CUDA repository not found, adding it now..."
        
        # Detect Ubuntu version for correct repository
        local ubuntu_version=""
        if [ "$OS_ID" = "Ubuntu" ]; then
            case "$OS_VERSION" in
                22.04)
                    ubuntu_version="ubuntu2204"
                    ;;
                20.04)
                    ubuntu_version="ubuntu2004"
                    ;;
                24.04)
                    ubuntu_version="ubuntu2404"
                    ;;
                *)
                    # Default to 22.04 for unknown versions
                    ubuntu_version="ubuntu2204"
                    echo "⚠️  Ubuntu $OS_VERSION detected, using ubuntu2204 repository (may need adjustment)"
                    ;;
            esac
        else
            ubuntu_version="ubuntu2204"
            echo "⚠️  Non-Ubuntu system detected, using ubuntu2204 repository (may not work)"
        fi
        
        # Download and install CUDA keyring
        local cuda_keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_version}/x86_64/cuda-keyring_1.1-1_all.deb"
        
        if [ -f /tmp/cuda-keyring.deb ]; then
            rm /tmp/cuda-keyring.deb
        fi
        
        echo "📥 Downloading CUDA repository keyring for $ubuntu_version..."
        if wget -q "$cuda_keyring_url" -O /tmp/cuda-keyring.deb 2>/dev/null; then
            if sudo dpkg -i /tmp/cuda-keyring.deb 2>/dev/null; then
                echo "✅ NVIDIA CUDA repository keyring installed"
                # Force apt-get update after adding new repository
                echo "🔄 Updating package cache with new CUDA repository..."
                sudo apt-get update
                has_cuda_repo=true
            else
                echo "⚠️  Failed to install CUDA keyring"
            fi
            rm -f /tmp/cuda-keyring.deb
        else
            echo "⚠️  Failed to download CUDA keyring from $cuda_keyring_url"
        fi
        
        # If keyring install failed, provide manual instructions
        if [ "$has_cuda_repo" = false ]; then
            echo ""
            echo "⚠️  Automatic CUDA repository setup failed. Please add it manually:"
            echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_version}/x86_64/cuda-keyring_1.1-1_all.deb"
            echo "   sudo dpkg -i cuda-keyring_1.1-1_all.deb"
            echo "   sudo apt-get update"
            echo ""
        fi
    else
        echo "✅ NVIDIA CUDA repository already configured"
    fi
    
    # Try to install CUDA packages - prioritize detected version
    local cuda_installed=false
    
    if [ -n "$DETECTED_CUDA_VERSION" ]; then
        # Convert version like "12.4" to package suffix like "12-4"
        local version_suffix=$(echo "$DETECTED_CUDA_VERSION" | sed 's/\./-/')
        echo "🎯 Attempting to install CUDA $DETECTED_CUDA_VERSION packages..."
        
        # IMPORTANT: We need BOTH cuda-compiler (for complete headers) AND cuda-cudart-dev (for runtime)
        # Try to install both together first
        echo "📦 Installing cuda-compiler-${version_suffix} and cuda-cudart-dev-${version_suffix}..."
        if sudo apt-get install -y cuda-compiler-${version_suffix} cuda-cudart-dev-${version_suffix}; then
            echo "✅ CUDA $DETECTED_CUDA_VERSION compiler and runtime development packages installed!"
            cuda_installed=true
        else
            # If combined install fails, try separately
            echo "⚠️  Combined install failed, trying packages separately..."
            local compiler_installed=false
            local cudart_installed=false
            
            if sudo apt-get install -y cuda-compiler-${version_suffix}; then
                echo "✅ cuda-compiler-${version_suffix} installed"
                compiler_installed=true
            else
                echo "❌ Failed to install cuda-compiler-${version_suffix}"
            fi
            
            if sudo apt-get install -y cuda-cudart-dev-${version_suffix}; then
                echo "✅ cuda-cudart-dev-${version_suffix} installed"
                cudart_installed=true
            else
                echo "❌ Failed to install cuda-cudart-dev-${version_suffix}"
            fi
            
            if [ "$compiler_installed" = true ] || [ "$cudart_installed" = true ]; then
                if [ "$compiler_installed" = false ]; then
                    echo "⚠️  Warning: cuda-compiler not installed - headers may be incomplete!"
                fi
                if [ "$cudart_installed" = false ]; then
                    echo "⚠️  Warning: cuda-cudart-dev not installed - runtime dev files missing!"
                fi
                cuda_installed=true
            fi
        fi
        
        # If version-specific install failed, try toolkit
        if [ "$cuda_installed" = false ]; then
            echo "📦 Trying cuda-toolkit-${version_suffix} as fallback..."
            if sudo apt-get install -y cuda-toolkit-${version_suffix}; then
                echo "✅ CUDA $DETECTED_CUDA_VERSION toolkit installed!"
                cuda_installed=true
            fi
        fi
    fi
    
    # Fall back to trying various package names if version-specific install failed
    if [ "$cuda_installed" = false ]; then
        echo "🔄 Trying generic CUDA package installation..."
        
        # List available CUDA packages for debugging
        echo "🔍 Checking available CUDA packages..."
        local available_cuda_pkgs=$(apt-cache search "^cuda-cudart-dev" 2>/dev/null | head -5)
        if [ -n "$available_cuda_pkgs" ]; then
            echo "   Found packages:"
            echo "$available_cuda_pkgs" | sed 's/^/     /'
        fi
        
        # Try installing with specific version numbers first, then fall back to generic
        # Always try to get both compiler (for headers) and cudart-dev (for runtime)
        if sudo apt-get install -y cuda-compiler-12-4 cuda-cudart-dev-12-4; then
            echo "✅ CUDA 12.4 compiler and runtime development installed!"
            cuda_installed=true
        elif sudo apt-get install -y cuda-compiler-12-6 cuda-cudart-dev-12-6; then
            echo "✅ CUDA 12.6 compiler and runtime development installed!"
            cuda_installed=true
        elif sudo apt-get install -y cuda-compiler-12-2 cuda-cudart-dev-12-2; then
            echo "✅ CUDA 12.2 compiler and runtime development installed!"
            cuda_installed=true
        elif sudo apt-get install -y cuda-compiler-12-0 cuda-cudart-dev-12-0; then
            echo "✅ CUDA 12.0 compiler and runtime development installed!"
            cuda_installed=true
        elif sudo apt-get install -y cuda-toolkit-12-4; then
            echo "✅ CUDA 12.4 toolkit installed!"
            cuda_installed=true
        elif sudo apt-get install -y cuda-toolkit-12-6; then
            echo "✅ CUDA 12.6 toolkit installed!"
            cuda_installed=true
        elif sudo apt-get install -y cuda-toolkit; then
            echo "✅ CUDA toolkit (latest) installed!"
            cuda_installed=true
        elif sudo apt-get install -y cuda; then
            echo "✅ CUDA (full package) installed!"
            cuda_installed=true
        fi
    fi
    
    if [ "$cuda_installed" = true ]; then
        return 0
    else
        echo "❌ Failed to install CUDA packages for x86_64 system"
        echo "   💡 You may need to manually install CUDA toolkit from:"
        echo "      https://developer.nvidia.com/cuda-downloads"
        return 1
    fi
}

# CUDA detection and installation - optimized for Jetson systems
install_cuda_for_jetson() {
    echo "🚀 Installing CUDA development packages for Jetson..."
    
    # First, try to install CUDA development headers for existing CUDA installation
    local success=false
    
    # Check if we can install from JetPack repositories (try newer versions first)
    if sudo apt-get install -y nvidia-jetpack-dev 2>/dev/null; then
        echo "✅ JetPack development packages installed!"
        success=true
    elif sudo apt-get install -y cuda-toolkit-12-6-dev 2>/dev/null || \
         sudo apt-get install -y cuda-toolkit-12-5-dev 2>/dev/null || \
         sudo apt-get install -y cuda-toolkit-12-4-dev 2>/dev/null || \
         sudo apt-get install -y cuda-toolkit-11-4-dev 2>/dev/null || \
         sudo apt-get install -y cuda-toolkit-10-2-dev 2>/dev/null; then
        echo "✅ CUDA toolkit development packages installed!"
        success=true
    elif sudo apt-get install -y cuda-compiler-12-6 cuda-cudart-dev-12-6 2>/dev/null || \
         sudo apt-get install -y cuda-compiler-12-5 cuda-cudart-dev-12-5 2>/dev/null || \
         sudo apt-get install -y cuda-compiler-12-4 cuda-cudart-dev-12-4 2>/dev/null || \
         sudo apt-get install -y cuda-compiler-11-4 cuda-cudart-dev-11-4 2>/dev/null || \
         sudo apt-get install -y cuda-compiler-10-2 cuda-cudart-dev-10-2 2>/dev/null; then
        echo "✅ CUDA compiler and runtime dev packages installed!"
        success=true
    else
        # Try generic packages that might work
        echo "⚠️  Jetson-specific packages not found. Trying generic CUDA packages..."
        if sudo apt-get install -y cuda-cudart-dev libcuda1-dev 2>/dev/null || \
           sudo apt-get install -y cuda-toolkit 2>/dev/null; then
            echo "✅ Generic CUDA development packages installed!"
            success=true
        fi
    fi
    
    # If all else fails, try to create symlinks to existing CUDA installation
    if [ "$success" = false ]; then
        echo "⚠️  Package installation failed. Trying to link existing CUDA installation..."
        
        # Find existing CUDA installation (using shared variable)
        for cuda_dir in "${CUDA_PATHS[@]}"; do
            if [ -d "$cuda_dir/include" ]; then
                echo "📍 Found CUDA at $cuda_dir, creating development symlinks..."
                sudo mkdir -p /usr/include/cuda
                sudo ln -sf $cuda_dir/include/* /usr/include/cuda/ 2>/dev/null || true
                sudo ln -sf $cuda_dir/include/cuda_runtime.h /usr/include/ 2>/dev/null || true
                echo "✅ CUDA headers linked successfully!"
                success=true
                break
            fi
        done
    fi
    
    return $([ "$success" = true ] && echo 0 || echo 1)
}

# Check CUDA installation status
if check_cuda_toolkit; then
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "✅ CUDA Toolkit found: nvcc version $CUDA_VERSION"
    else
        # Found in custom path - add to PATH suggestion (using shared variable)
        for cuda_path in "${CUDA_PATHS[@]}"; do
            if [ -f "$cuda_path/bin/nvcc" ]; then
                CUDA_VERSION=$($cuda_path/bin/nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' 2>/dev/null || echo "unknown")
                echo "✅ CUDA Toolkit found at: $cuda_path (version $CUDA_VERSION)"
                echo "   💡 Consider adding $cuda_path/bin to your PATH"
                break
            fi
        done
    fi
    
    # Even with toolkit found, verify runtime libraries and headers are available
    echo "🔍 Verifying CUDA runtime libraries and headers are complete..."
    if check_cuda_runtime; then
        echo "✅ CUDA toolkit is complete with runtime libraries and headers!"
    else
        echo "⚠️  CUDA toolkit found but runtime libraries/headers missing!"
        
        if [ "$IS_JETSON" = true ]; then
            echo "🤖 Installing missing CUDA runtime development packages for Jetson..."
            if install_cuda_for_jetson; then
                echo "✅ Jetson CUDA runtime packages installed successfully!"
            else
                echo "❌ Jetson CUDA runtime installation failed"
                echo "   💡 System may already have working CUDA runtime"
            fi
        fi
    fi
elif check_cuda_runtime; then
    echo "✅ CUDA runtime libraries and headers found (sufficient for this project!)"
    echo "   ℹ️  Full toolkit not detected, but runtime + headers are all you need"
else
    # CUDA components missing - install them
    echo "⚠️  CUDA components missing or incomplete. Installing CUDA runtime libraries..."
    
    if [ "$IS_JETSON" = true ]; then
        echo "🤖 Installing CUDA runtime development packages for Jetson..."
        update_package_cache
        
        # Install CUDA runtime development packages for Jetson
        if sudo apt install -y cuda-cudart-dev-12-6 cuda-headers-12-6 2>/dev/null; then
            echo "✅ CUDA 12.6 development packages installed!"
        elif sudo apt install -y cuda-cudart-dev cuda-toolkit-12-6 2>/dev/null; then
            echo "✅ CUDA development packages installed!"
        elif sudo apt install -y libcudart-dev cuda-toolkit 2>/dev/null; then
            echo "✅ Generic CUDA development packages installed!"
        else
            echo "❌ Failed to install CUDA development packages"
            echo "   Try manually: sudo apt install -y cuda-cudart-dev-12-6"
        fi
    else
        # Non-Jetson installation logic (existing)
        if [[ "$ARCH" == "aarch64" ]]; then
            install_cuda_for_arm64
        elif [[ "$ARCH" == "x86_64" ]]; then
            install_cuda_for_x86_64
        fi
    fi
    
    # Verify CUDA is now available
    echo "🔍 Verifying CUDA installation after package installation..."
    if check_cuda_runtime; then
        echo "✅ CUDA runtime and headers now available!"
    else
        echo "❌ CUDA verification still failing - manual intervention may be needed"
    fi
fi

# Initialize Git LFS
echo "📁 Setting up Git LFS..."
if git lfs install --force 2>/dev/null; then
    echo "✅ Git LFS hooks installed successfully"
else
    echo "⚠️  Git LFS hook installation had issues, trying manual resolution..."
    # Try to update existing hooks
    if git lfs update --force 2>/dev/null; then
        echo "✅ Git LFS hooks updated successfully"
    else
        echo "⚠️  Git LFS hooks may need manual attention"
        echo "   You can run 'git lfs update --force' later if needed"
    fi
fi

echo ""
echo "🎉 System dependencies installation complete!"
echo ""

if [ "$IS_JETSON" = true ]; then
    echo "🤖 Jetson-specific setup completed!"
    echo ""
    echo "📋 Next steps for Jetson system:"
    echo "   1. Ensure TensorRT is available (usually pre-installed with JetPack)"
    echo "   2. Run 'source scripts/setup_env.sh' to configure paths for Jetson"
    echo "   3. Run 'just build' to build the project"
    echo ""
    echo "💡 Jetson tips:"
    echo "   - CUDA is typically pre-installed at /usr/local/cuda-12.6, /usr/local/cuda-11.4 or /usr/local/cuda-10.2"
    echo "   - TensorRT is usually available at /usr/lib/aarch64-linux-gnu/"
    echo "   - If build fails, you may need to install JetPack SDK development components"
    echo "   - Monitor system temperature during compilation (use 'tegrastats')"
    echo "   - For CUDA 12.6 systems, ensure you have the latest JetPack SDK Manager"
    echo ""
    
    # Install JetPack for Jetson systems (includes all DLA libraries)
    echo "🔧 Installing JetPack for complete Jetson development environment..."
    
    update_package_cache
    
    if sudo apt install -y nvidia-jetpack; then
        echo "✅ JetPack installed successfully - includes all DLA libraries!"
        echo "   This resolves all TensorRT DLA dependencies"
    else
        echo "⚠️  JetPack installation failed (may have version conflicts)"
        echo "   💡 Try: sudo apt update && sudo apt install -y nvidia-jetpack"
        echo "   🔄 If issues persist, build should still work with existing libraries"
    fi
else
    # Non-Jetson systems
    echo "📋 Next steps for $ARCH system:"
echo "   1. Make sure TensorRT is installed and TensorRT_ROOT is set"
echo "   2. Run 'source scripts/setup_env.sh' to set up the environment"
echo "   3. Run 'just build' to build the project"
echo ""
    
    if [[ "$ARCH" == "aarch64" ]]; then
        echo "💡 ARM64 tips:"
        echo "   - CUDA may be installed at /usr/local/cuda or system locations"
        echo "   - TensorRT typically installed via package manager"
        echo "   - DLA libraries not needed on non-Jetson ARM64 systems"
    elif [[ "$ARCH" == "x86_64" ]]; then
        echo "💡 x86_64 tips:"
        echo "   - CUDA typically at /usr/local/cuda or via CUDA toolkit"
        echo "   - TensorRT available from NVIDIA or package repositories"
        echo "   - Use nvidia-smi to check GPU compatibility"
        echo "   - DLA libraries typically not needed on desktop GPUs"
    fi
fi

echo ""
echo "⚠️  If you encounter any issues:"
if [ "$IS_JETSON" = true ]; then
    echo "   🤖 Jetson-specific troubleshooting:"
    echo "   - Check JetPack version: jetson_release"
    echo "   - Verify CUDA installation: ls -la /usr/local/cuda*"
    echo "   - Monitor resources: tegrastats"
    echo "   - For DLA linking errors: Install full JetPack SDK"
    echo "   - For help: https://developer.nvidia.com/embedded/jetpack"
elif [[ "$ARCH" == "aarch64" ]]; then
    echo "   🔧 ARM64 troubleshooting:"
    echo "   - Check CUDA: ls -la /usr/local/cuda*"
    echo "   - Verify libraries: ldconfig -p | grep cuda"
    echo "   - DLA errors: DLA not supported on non-Jetson ARM64"
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "   🖥️  x86_64 troubleshooting:"
    echo "   - Check GPU: nvidia-smi"
    echo "   - Verify CUDA: nvcc --version"
    echo "   - Check environment: echo \$CUDA_HOME \$TensorRT_ROOT"
    echo "   - DLA errors: DLA typically not supported on desktop GPUs"
else
    echo "   - Check that all environment variables are set correctly"
    echo "   - Verify CUDA installation for your architecture"
fi
