#!/bin/bash

# Smart ROS2 Development Container with System Detection
# Usage: ./run-ros2-dev.sh [OPTIONS]
#   --rebuild, -r        : Force rebuild of Docker image
#   --with-opengl, -gl   : Build custom CUDA+OpenGL base image (takes ~30min first time)
#   --help, -h          : Show this help message

cd "$(dirname "$0")"

# Parse arguments
BUILD_CUDAGL=false
FORCE_REBUILD=false

for arg in "$@"; do
    case $arg in
        --rebuild|-r)
            FORCE_REBUILD=true
            shift
            ;;
        --with-opengl|-gl)
            BUILD_CUDAGL=true
            shift
            ;;
        --help|-h)
            echo "G1 Deploy Docker Environment Launcher"
            echo ""
            echo "Usage: ./run-ros2-dev.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rebuild, -r        Force rebuild of Docker image"
            echo "  --with-opengl, -gl   Build custom CUDA+OpenGL base image (for visualization/GUI)"
            echo "                       Takes ~30 minutes first time, needs ~5GB download"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run-ros2-dev.sh                    # Quick start with standard CUDA"
            echo "  ./run-ros2-dev.sh --with-opengl      # Include OpenGL for RViz/Gazebo"
            echo "  ./run-ros2-dev.sh --rebuild          # Force rebuild"
            exit 0
            ;;
    esac
done

echo "🚀 G1 Deploy - Smart ROS2 Development Environment"
echo "================================================="
echo "🏗️  Host Architecture: $(uname -m)"
if [ "$BUILD_CUDAGL" = true ]; then
    echo "🎨 Mode: CUDA + OpenGL (for visualization/rendering)"
else
    echo "⚡ Mode: Standard CUDA (fast, for inference/control)"
fi

# Check host NVIDIA driver and CUDA compatibility
if command -v nvidia-smi &> /dev/null; then
    HOST_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
    HOST_CUDA=$(nvidia-smi | grep "CUDA Version:" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/' | head -n1)
    if [ -n "$HOST_CUDA" ]; then
        echo "🔍 Host NVIDIA Driver: $HOST_DRIVER (CUDA $HOST_CUDA)"
    fi
fi

# Detect system type
SYSTEM_TYPE="generic"
IS_JETSON=false
JETSON_MODEL=""
DOCKERFILE="Dockerfile.ros2"  # Unified Dockerfile for all platforms
IMAGE_NAME="g1-deploy-dev"
CUDA_VERSION="12.4.1"  # Default for x86_64/ARM64

# Enhanced Jetson detection
if [[ "$(uname -m)" == "aarch64" ]]; then
    # Check for Jetson-specific indicators
    if [[ -f "/etc/nv_tegra_release" ]] || \
       [[ -d "/usr/src/jetson_multimedia_api" ]] || \
       ls /etc/apt/sources.list.d/ 2>/dev/null | grep -q jetson || \
       [[ -f "/proc/device-tree/model" && $(cat /proc/device-tree/model 2>/dev/null) =~ "Jetson" ]]; then
        IS_JETSON=true
        SYSTEM_TYPE="jetson"
        # Use closest available CUDA version to Jetson's CUDA
        # Note: Jetson has CUDA 12.6, but Docker images only go up to 12.4.1
        # This is OK - newer driver (12.6) can run older CUDA (12.4.1)
        CUDA_VERSION="12.4.1"  # Closest available to Jetson's 12.6
        
        # Detect Jetson model
        if [[ -f "/proc/device-tree/model" ]]; then
            JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' | sed 's/NVIDIA //')
            echo "🤖 Detected Jetson: $JETSON_MODEL"
        else
            echo "🤖 Detected Jetson System"
        fi
        
        # Check JetPack version if available
        if command -v jetson_release &> /dev/null; then
            JETPACK_VERSION=$(jetson_release -v 2>/dev/null | grep "JETPACK" | cut -d' ' -f2 || echo "unknown")
            echo "📦 JetPack Version: $JETPACK_VERSION"
        fi
    fi
fi

# System-specific configuration
case "$SYSTEM_TYPE" in
    jetson)
        echo "✅ Jetson system detected - using CUDA ${CUDA_VERSION}"
        if [ -n "$HOST_CUDA" ]; then
            echo "ℹ️  Host CUDA: $HOST_CUDA, Container CUDA: ${CUDA_VERSION}"
        fi
        echo "✅ Unified Dockerfile with Jetson auto-detection"
        echo "✅ Deep Learning Accelerator (DLA) support enabled"
        echo "✅ TensorRT integration via host mount"
        ;;
    *)
        echo "✅ Unified multi-architecture support"
        echo "✅ Container CUDA version: ${CUDA_VERSION}"
        
        # Check CUDA version compatibility
        if [ -n "$HOST_CUDA" ]; then
            CUDA_MAJOR_HOST=$(echo "$HOST_CUDA" | cut -d. -f1)
            CUDA_MAJOR_CONTAINER=$(echo "$CUDA_VERSION" | cut -d. -f1)
            
            if [ "$CUDA_MAJOR_CONTAINER" -gt "$CUDA_MAJOR_HOST" ]; then
                echo ""
                echo "⚠️  WARNING: Container CUDA ($CUDA_VERSION) > Host CUDA ($HOST_CUDA)"
                echo "   This may not work! Your driver might be too old."
                echo "   Recommended: Update NVIDIA driver to support CUDA $CUDA_VERSION"
                echo ""
            elif [ "$CUDA_VERSION" != "$HOST_CUDA" ]; then
                echo "ℹ️  Note: Container CUDA ($CUDA_VERSION) differs from host ($HOST_CUDA)"
                echo "   This is OK if your driver supports CUDA $CUDA_VERSION"
            fi
        fi
        
        case $(uname -m) in
            x86_64)
                echo "✅ Full x86_64 support with CUDA and TensorRT"
                ;;
            aarch64)
                echo "✅ ARM64 support (non-Jetson)"
                echo "✅ CUDA support from official NVIDIA images"
                ;;
            *)
                echo "⚠️  Architecture $(uname -m) - experimental support"
                ;;
        esac
        ;;
esac
echo ""

# Build custom CUDA+GL base image if requested
CUDA_BASE_IMAGE="nvidia/cuda"
if [ "$BUILD_CUDAGL" = true ]; then
    echo "🎨 Building custom CUDA+OpenGL base image..."
    echo "   This is a one-time process (takes ~30 minutes)"
    echo ""
    
    CUDAGL_IMAGE="g1-cuda-gl"
    CUDAGL_TAG="${CUDA_VERSION}-devel-ubuntu22.04"
    
    # Check if custom image already exists
    if docker image inspect "${CUDAGL_IMAGE}:${CUDAGL_TAG}" >/dev/null 2>&1 && [ "$FORCE_REBUILD" != true ]; then
        echo "✅ Custom CUDA+GL image already exists: ${CUDAGL_IMAGE}:${CUDAGL_TAG}"
    else
        # Clone NVIDIA CUDA repo if not exists
        if [ ! -d "nvidia-cuda-build" ]; then
            echo "📥 Cloning NVIDIA CUDA repository (one-time, ~2GB)..."
            git clone --depth 1 https://gitlab.com/nvidia/container-images/cuda.git nvidia-cuda-build
        fi
        
        cd nvidia-cuda-build
        
        # Determine architecture
        BUILD_ARCH=$(uname -m)
        if [ "$BUILD_ARCH" = "x86_64" ]; then
            BUILD_ARCH_FLAG="x86_64"
        elif [ "$BUILD_ARCH" = "aarch64" ]; then
            BUILD_ARCH_FLAG="arm64"
        else
            echo "⚠️  Unknown architecture: $BUILD_ARCH, using x86_64"
            BUILD_ARCH_FLAG="x86_64"
        fi
        
        echo "🔨 Building CUDA ${CUDA_VERSION} + OpenGL for ${BUILD_ARCH_FLAG}..."
        echo "   This takes 20-30 minutes on first build..."
        
        # Build with cudagl flag
        ./build.sh -d \
            --image-name "${CUDAGL_IMAGE}" \
            --cuda-version "${CUDA_VERSION}" \
            --os ubuntu \
            --os-version 22.04 \
            --arch "${BUILD_ARCH_FLAG}" \
            --cudagl
        
        if [ $? -eq 0 ]; then
            echo "✅ Custom CUDA+GL base image built successfully!"
        else
            echo "❌ Failed to build custom CUDA+GL image"
            echo "   Falling back to standard CUDA image"
            BUILD_CUDAGL=false
        fi
        
        cd ..
    fi
    
    if [ "$BUILD_CUDAGL" = true ]; then
        CUDA_BASE_IMAGE="${CUDAGL_IMAGE}"
        echo "✅ Will use custom CUDA+GL image: ${CUDA_BASE_IMAGE}:${CUDAGL_TAG}"
    fi
    
    echo ""
fi

# Clean up any existing containers (including old names)
docker rm -f g1-deploy-dev g1-ros2-dev g1-jetson-dev 2>/dev/null || true

# Smart Docker image build - only rebuild if needed
BUILD_IMAGE=false
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "📦 Image $IMAGE_NAME not found - building..."
    BUILD_IMAGE=true
elif [ "$FORCE_REBUILD" = true ]; then
    echo "📦 Forced rebuild requested..."
    BUILD_IMAGE=true
else
    echo "✅ Using existing Docker image: $IMAGE_NAME"
    echo "   (Use --rebuild or -r to force rebuild)"
fi

if [ "$BUILD_IMAGE" = true ]; then
    echo "📦 Building Docker image: $IMAGE_NAME"
    echo "   Dockerfile: $DOCKERFILE (unified for all platforms)"
    echo "   CUDA Version: $CUDA_VERSION"
    echo "   Base Image: $CUDA_BASE_IMAGE"
    echo "   System Type: $SYSTEM_TYPE"
    
    # Build with parent directory as context (to access ../scripts/)
    # Use -f to specify Dockerfile in current directory
    # Use host networking during build to avoid Jetson iptables-bridge issues and DNS failures
    docker build --network host -f "$DOCKERFILE" -t "$IMAGE_NAME" .. \
        --build-arg CUDA_VERSION=$CUDA_VERSION \
        --build-arg CUDA_BASE_IMAGE=$CUDA_BASE_IMAGE
fi

# Check for TensorRT - unified for all platforms
TENSORRT_MOUNT=""
ADDITIONAL_MOUNTS=""
DEVICE_MOUNTS=""

if [ -n "$TensorRT_ROOT" ] && [ -d "$TensorRT_ROOT" ]; then
    echo "✅ TensorRT found: $TensorRT_ROOT (mounting to container)"
    TENSORRT_MOUNT="-v $TensorRT_ROOT:/opt/TensorRT:ro"
else
    echo "⚠️  TensorRT not found - Set \$TensorRT_ROOT environment variable for GPU inference"
    echo "    Example: export TensorRT_ROOT=/path/to/TensorRT"
fi

if [[ "$SYSTEM_TYPE" == "jetson" ]]; then
    # Jetson: Add system library mounts
    if [ -n "$TENSORRT_MOUNT" ]; then
        # Mount NVIDIA DLA libraries from host
        ADDITIONAL_MOUNTS="-v /usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/nvidia:ro"

        # Mount host CUDA toolkit (Jetson commonly needs libcudla with cudla* symbols)
        # Avoid hardcoding CUDA versions (JetPack 5/6 differ); pick newest cuda-* that has targets/aarch64-linux/lib.
        HOST_CUDA_TOOLKIT=""
        if [ -d "/usr/local/cuda/targets/aarch64-linux/lib" ]; then
            HOST_CUDA_TOOLKIT="/usr/local/cuda"
        else
            # Sort so cuda-12.6 > cuda-12.4 > cuda-11.4, etc.
            for cuda_dir in $(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -V -r); do
                if [ -d "$cuda_dir/targets/aarch64-linux/lib" ]; then
                    HOST_CUDA_TOOLKIT="$cuda_dir"
                    break
                fi
            done
        fi

        if [ -n "$HOST_CUDA_TOOLKIT" ]; then
            echo "✅ Host CUDA toolkit found: $HOST_CUDA_TOOLKIT (mounting to container)"
            ADDITIONAL_MOUNTS="$ADDITIONAL_MOUNTS -v $HOST_CUDA_TOOLKIT:$HOST_CUDA_TOOLKIT:ro"
        else
            echo "⚠️  Host CUDA toolkit not found under /usr/local/cuda-*/targets/aarch64-linux/lib"
            echo "    If DLA (cudla) linking fails, install JetPack dev components on host"
        fi
        
        # Mount Jetson info if available
        if [ -f "/etc/nv_tegra_release" ]; then
            ADDITIONAL_MOUNTS="$ADDITIONAL_MOUNTS -v /etc/nv_tegra_release:/etc/nv_tegra_release:ro"
        fi
        
        # Add Jetson device mounts for GPU access (only if they exist)
        DEVICE_MOUNTS=""
        JETSON_DEVICES=(
            "/dev/nvidia0"
            "/dev/nvidiactl"
            "/dev/nvidia-modeset"
            "/dev/nvhost-ctrl"
            "/dev/nvhost-ctrl-gpu"
            "/dev/nvhost-prof-gpu"
            "/dev/nvmap"
            "/dev/nvhost-gpu"
            "/dev/nvhost-as-gpu"
            "/dev/nvhost-vic"
            "/dev/tegra-crypto"
        )
        
        for device in "${JETSON_DEVICES[@]}"; do
            if [ -e "$device" ]; then
                DEVICE_MOUNTS="$DEVICE_MOUNTS --device $device:$device"
            fi
        done
        
        if [ -z "$DEVICE_MOUNTS" ]; then
            echo "⚠️  Warning: No Jetson GPU devices found in /dev/"
        fi
        
        # Mount cuDNN if available
        if ls /usr/lib/aarch64-linux-gnu/libcudnn* >/dev/null 2>&1; then
            echo "✅ cuDNN libraries found on host"
        fi
    fi
    
    # Check for DLA libraries
    if [ -f "/usr/lib/aarch64-linux-gnu/nvidia/libnvcudla.so" ]; then
        echo "✅ Deep Learning Accelerator (DLA) libraries available"
    else
        echo "⚠️  DLA libraries not found - will run without DLA acceleration"
    fi
fi

# Check FastRTPS profile
if [ -f "../src/g1/g1_deploy_onnx_ref/config/fastrtps_profile.xml" ]; then
    echo "✅ FastRTPS production profile ready"
else
    echo "⚠️  FastRTPS profile missing - using defaults"
fi

echo ""
echo "🐳 Launching $SYSTEM_TYPE-optimized container..."

# Determine GPU runtime settings
GPU_SETTINGS=""
if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    # nvidia runtime is available
    if [[ "$SYSTEM_TYPE" == "jetson" ]]; then
        GPU_SETTINGS="--runtime nvidia --gpus all"
    else
        GPU_SETTINGS="--gpus all"
    fi
else
    # nvidia runtime not available, use --gpus all (works with Docker 19.03+)
    if command -v nvidia-smi &> /dev/null; then
        echo "ℹ️  Using --gpus all (nvidia runtime not configured)"
        GPU_SETTINGS="--gpus all"
    else
        echo "⚠️  Warning: No NVIDIA GPU support detected"
        GPU_SETTINGS=""
    fi
fi

# Run the container with system-specific configuration
docker run -it --rm \
    --name "$IMAGE_NAME" \
    --network host \
    $GPU_SETTINGS \
    -v "$(cd .. && pwd):/workspace/g1_deploy:rw" \
    $TENSORRT_MOUNT \
    $ADDITIONAL_MOUNTS \
    $DEVICE_MOUNTS \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_DOMAIN_ID=0 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e SYSTEM_TYPE="$SYSTEM_TYPE" \
    -e IS_JETSON="$IS_JETSON" \
    -e JETSON_MODEL="$JETSON_MODEL" \
    -w /workspace/g1_deploy \
    "$IMAGE_NAME" \
    bash -c "
        echo ''
        if [[ \"\$SYSTEM_TYPE\" == \"jetson\" ]]; then
            echo '🤖 G1 Deploy Jetson Development Environment'
            echo '============================================'
            echo '📦 Jetson CUDA + DLA Integration'

            # Match bare-metal builds: if a host CUDA toolkit is mounted, prefer it.
            # Choose /usr/local/cuda if it has targets/aarch64-linux; otherwise pick newest cuda-*.
            if [ -d '/usr/local/cuda/targets/aarch64-linux/lib' ]; then
                export CUDAToolkit_ROOT='/usr/local/cuda'
                export CUDA_HOME='/usr/local/cuda'
            else
                BEST_CUDA=\$(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -V -r | head -n1)
                if [ -n \"\$BEST_CUDA\" ] && [ -d \"\$BEST_CUDA/targets/aarch64-linux/lib\" ]; then
                    export CUDAToolkit_ROOT=\"\$BEST_CUDA\"
                    export CUDA_HOME=\"\$BEST_CUDA\"
                fi
            fi
            
            # Ensure NVIDIA library path is in linker configuration for DLA
            if [ -d '/usr/lib/aarch64-linux-gnu/nvidia' ]; then
                echo '/usr/lib/aarch64-linux-gnu/nvidia' > /etc/ld.so.conf.d/nvidia.conf
                ldconfig
                echo '✅ NVIDIA DLA libraries configured'
            fi
        else
            echo '🎯 G1 Deploy ROS2 Development Environment'
            echo '========================================='
            echo '📦 NVIDIA CUDA Multi-Architecture'
        fi
        echo ''
        
        # Fix Git ownership issue for mounted directory
        git config --global --add safe.directory /workspace/g1_deploy
        
        # Set up environment using the project's setup script
        # This ensures consistency between Docker and bare-metal installations
        if [ -f 'scripts/setup_env.sh' ]; then
            echo '🔧 Running setup_env.sh for environment configuration...'
            source scripts/setup_env.sh
        else
            # Fallback if setup script not found
            echo '⚠️  setup_env.sh not found, using basic environment'
            source /opt/ros/humble/setup.bash
            export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
            export LD_LIBRARY_PATH='/opt/onnxruntime/lib:\$LD_LIBRARY_PATH'
        fi
        
        # Status check
        echo '📊 Environment Status:'
        if [[ \"\$SYSTEM_TYPE\" == \"jetson\" ]]; then
            echo \"  🤖 System: Jetson \$JETSON_MODEL\"
        fi
        echo \"  🏗️  Architecture: $(uname -m)\"
        command -v just >/dev/null && echo '  ✅ Just command runner' || echo '  ❌ Just not found'
        command -v cmake >/dev/null && echo '  ✅ CMake build system' || echo '  ❌ CMake not found'
        echo \"  ✅ ROS2 Humble ($(ros2 --version 2>/dev/null || echo 'version unknown'))\"
        echo '  ✅ ONNX Runtime 1.16.3'
        command -v nvcc >/dev/null 2>&1 && echo \"  ✅ CUDA \$(nvcc --version | grep release | cut -d' ' -f5 | cut -d',' -f1 2>/dev/null || echo 'Toolkit')\" || echo '  ⚠️  CUDA Toolkit not found'
        
        # Check TensorRT (unified location for all platforms)
        ls /opt/TensorRT >/dev/null 2>&1 && echo '  ✅ TensorRT (mounted from host)' || echo '  ⚠️  TensorRT not mounted - set \$TensorRT_ROOT on host'
        
        # Jetson-specific checks
        if [[ \"\$SYSTEM_TYPE\" == \"jetson\" ]]; then
            [ -f '/usr/lib/aarch64-linux-gnu/nvidia/libnvcudla.so' ] && echo '  ✅ Deep Learning Accelerator (DLA) ready' || echo '  ⚠️  DLA libraries not found'
            nvidia-smi 2>/dev/null | grep -q 'GPU' && echo '  ✅ GPU acceleration available' || echo '  ⚠️  GPU not detected (check device mounts)'
        fi
        
        echo ''
        echo '🛠️  Quick Commands:'
        if [[ \"\$SYSTEM_TYPE\" == \"jetson\" ]]; then
            echo '   just build       # Build with ARM64 + DLA optimizations'
            echo '   just test-ros2   # Test ROS2 integration on Jetson'
            echo '   just run freq_test model.onnx  # Test inference with DLA'
        else
            echo '   just build       # Build with ROS2 support'
            echo '   just test-ros2   # Test ROS2 integration'
        fi
        echo '   just --list      # Show all commands'
        echo ''
        echo 'Ready for development! 🚀'
        echo ''
        
        exec bash
    "
