#!/bin/bash

# Docker run script for decoupled_wbc with branch-based container isolation
# 
# Usage:
#   ./docker/run_docker.sh [OPTIONS]    (run from inside decoupled_wbc/)
#
# Options:
#   --build           Build Docker image
#   --clean           Clean containers
#   --deploy          Run in deploy mode
#   --install         Pull prebuilt Docker image
#   --push            Push built image to Docker Hub
#   --branch          Use branch-specific container names
#
# Branch-based Container Isolation (when --branch flag is used):
#   - Each git branch gets its own isolated containers
#   - Container names include branch identifier (e.g., decoupled_wbc-deploy-user-main)
#   - Works with git worktrees, separate clones, or nested repositories
#   - Clean and build operations only affect the current branch

# Exit on error
set -e

# Default values
BUILD=false
CLEAN=false
DEPLOY=false
INSTALL=false
# Flag to push the built Docker image to Docker Hub
# This should be used when someone updates the Docker image dependencies
# because this image is used for CI/CD pipelines
# When true, the image will be tagged and pushed to docker.io/nvgear/gr00t_wbc:latest
DOCKER_HUB_PUSH=false
# Flag to build the docker with root user
# This could cause some of your local files to be owned by root
# If you get error like "PermissionError: [Errno 13] Permission denied:"
# You can run `sudo chown -R $USER:$USER .` in local machine to fix it
ROOT=false
BRANCH_MODE=false
EXTRA_ARGS=()
PROJECT_NAME="decoupled_wbc"
PROJECT_SLUG=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]')
REMOTE_IMAGE="nvgear/gr00t_wbc:latest"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --deploy)
            DEPLOY=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --push)
            DOCKER_HUB_PUSH=true
            shift
            ;;
        --root)
            ROOT=true
            shift
            ;;
        --branch)
            BRANCH_MODE=true
            shift
            ;;
        *)
            # Collect all unknown arguments as extra args for the deployment script
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$INSTALL" = true ] && [ "$BUILD" = true ]; then
    echo "Cannot use --install and --build together. Choose one."
    exit 1
fi


# Function to get branch name for container naming
function get_branch_id {
    # Check if we're in a git repository
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        # Get current branch name (returns "HEAD" in detached state)
        local branch_name=$(git rev-parse --abbrev-ref HEAD)
        # Replace forward slashes with dashes for valid container names
        echo "${branch_name//\//-}"
    else
        # Default: no branch identifier (not in git repo)
        echo ""
    fi
}

# Architecture detection helpers
is_arm64() { [ "$(dpkg --print-architecture)" = "arm64" ]; }
is_amd64() { [ "$(dpkg --print-architecture)" = "amd64" ]; }

# Get current user's username and UID
if [ "$ROOT" = true ]; then
    USERNAME=root
    USERID=0
    DOCKER_HOME_DIR=/root
    CACHE_FROM=${PROJECT_SLUG}-deploy-cache-root
else
    USERNAME=$(whoami)
    USERID=$(id -u)
    DOCKER_HOME_DIR=/home/${USERNAME}
    CACHE_FROM=${PROJECT_SLUG}-deploy-cache
fi
# Get input group ID for device access
INPUT_GID=$(getent group input | cut -d: -f3)

# Get script directory for path calculations
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to get the actual project directory (worktree-aware)
function get_project_dir {
    # For worktrees, use the actual worktree root path
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        git rev-parse --show-toplevel
    else
        # Fallback to script-based detection (go up two levels: docker/ -> decoupled_wbc/ -> project root)
        dirname "$(dirname "$SCRIPT_DIR")"
    fi
}

# Get branch identifier
BRANCH_ID=$(get_branch_id)

# Set project directory (needs to be after branch detection)
PROJECT_DIR="$(get_project_dir)"

# Function to generate container name with optional branch support
function get_container_name {
    local container_type="$1"
    if [[ -n "$BRANCH_ID" ]] && [[ "$BRANCH_MODE" = true ]]; then
        echo "${PROJECT_SLUG}-${container_type}-${USERNAME}-${BRANCH_ID}"
    else
        echo "${PROJECT_SLUG}-${container_type}-${USERNAME}"
    fi
}

# Set common variables used throughout the script
DEPLOY_CONTAINER=$(get_container_name "deploy")
BASH_CONTAINER=$(get_container_name "bash")
WORKTREE_NAME=$(basename "$PROJECT_DIR")

# Debug output for branch detection
if [[ -n "$BRANCH_ID" ]] && [[ "$BRANCH_MODE" = true ]]; then
    echo "Branch mode enabled - using branch: $BRANCH_ID"
    echo "Project directory: $PROJECT_DIR"
elif [[ -n "$BRANCH_ID" ]]; then
    echo "Branch mode disabled - using default containers"
    echo "Project directory: $PROJECT_DIR"
else
    echo "Running outside git repository"
    echo "Project directory: $PROJECT_DIR"
fi

# Get host's hostname and append -docker
HOSTNAME=$(hostname)-docker

function clean_container {
    echo "Cleaning up Docker containers..."
    
    # Stop containers
    sudo docker stop $DEPLOY_CONTAINER 2>/dev/null || true
    sudo docker stop $BASH_CONTAINER 2>/dev/null || true
    # Remove containers
    echo "Removing containers..."
    sudo docker rm $DEPLOY_CONTAINER 2>/dev/null || true
    sudo docker rm $BASH_CONTAINER 2>/dev/null || true
    echo "Containers cleaned!"
}


# Function to install Docker Buildx if needed
function install_docker_buildx {
    # Check if Docker Buildx is already installed
    if sudo docker buildx version &> /dev/null; then
        echo "Docker Buildx is already installed."
        return 0
    fi
    
    echo "Installing Docker Buildx..."
    
    # Create directories and detect architecture
    mkdir -p ~/.docker/cli-plugins/ && sudo mkdir -p /root/.docker/cli-plugins/
    ARCH=$(dpkg --print-architecture)
    [[ "$ARCH" == "arm64" ]] && BUILDX_ARCH="linux-arm64" || BUILDX_ARCH="linux-amd64"
    
    # Get version (with fallback)
    BUILDX_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep tag_name | cut -d '"' -f 4)
    BUILDX_VERSION=${BUILDX_VERSION:-v0.13.1}
    
    # Download and install for both user and root
    curl -L "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.${BUILDX_ARCH}" -o ~/.docker/cli-plugins/docker-buildx
    sudo cp ~/.docker/cli-plugins/docker-buildx /root/.docker/cli-plugins/docker-buildx
    chmod +x ~/.docker/cli-plugins/docker-buildx && sudo chmod +x /root/.docker/cli-plugins/docker-buildx
    
    # Create builder
    sudo docker buildx create --use --name mybuilder || true
    sudo docker buildx inspect --bootstrap
    
    echo "Docker Buildx installation complete!"
}

# Function to install NVIDIA Container Toolkit if needed
function install_nvidia_toolkit {
    # Check if NVIDIA Container Toolkit is already installed
    if command -v nvidia-container-toolkit &> /dev/null; then
        echo "NVIDIA Container Toolkit is already installed."
        return 0
    fi
    
    echo "Installing NVIDIA Container Toolkit..."

    # Add the package repositories
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

    # Check if GPG key exists and remove it if it does
    if [ -f "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg" ]; then
        echo "Removing existing NVIDIA GPG key..."
        sudo rm /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    fi

    # Add new GPG key
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    # Add repository
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

    # Install nvidia-container-toolkit and docker if needed
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Install docker if not already installed
    if ! command -v docker &> /dev/null; then
        sudo apt-get install -y docker.io
    fi

    # Configure Docker to use the NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker

    # Restart the Docker daemon
    sudo systemctl restart docker

    echo "NVIDIA Container Toolkit installation complete!"
}


# Function to build Docker image for current branch
function build_docker_image {
    echo "Building Docker image: $DEPLOY_CONTAINER"

    sudo docker buildx build \
        --build-arg USERNAME=$USERNAME \
        --build-arg USERID=$USERID \
        --build-arg HOME_DIR=$DOCKER_HOME_DIR \
        --build-arg WORKTREE_NAME=$WORKTREE_NAME \
        --cache-from $CACHE_FROM \
        -t $DEPLOY_CONTAINER \
        -f "$SCRIPT_DIR/Dockerfile.deploy" \
        --load \
        "$PROJECT_DIR"

    # Tag for persistent cache
    # sudo docker tag $DEPLOY_CONTAINER $CACHE_FROM
    echo "Docker image build complete!"
}

# Build function 
function build_with_cleanup {
    echo "Building Docker image..."
    echo "Removing existing containers and images..."
    clean_container
    # Tag for persistent cache before deleting the image
    sudo docker tag $DEPLOY_CONTAINER $CACHE_FROM 2>/dev/null || true
    sudo docker rmi $DEPLOY_CONTAINER 2>/dev/null || true
    echo "Images cleaned!"
    
    install_docker_buildx
    install_nvidia_toolkit
    build_docker_image
}

function install_remote_image {
    echo "Installing Docker image from remote registry: $REMOTE_IMAGE"
    echo "Removing existing containers to ensure a clean install..."
    clean_container
    sudo docker pull "$REMOTE_IMAGE"
    sudo docker tag "$REMOTE_IMAGE" "$DEPLOY_CONTAINER"
    sudo docker tag "$REMOTE_IMAGE" "$CACHE_FROM" 2>/dev/null || true
    echo "Docker image install complete!"
}

# Clean up if requested
if [ "$CLEAN" = true ]; then
    clean_container
    exit 0
fi

# Build if requested
if [ "$BUILD" = true ]; then
    build_with_cleanup
fi

if [ "$INSTALL" = true ]; then
    install_remote_image
fi

if [ "$DOCKER_HUB_PUSH" = true ]; then
    echo "Pushing Docker image to Docker Hub: docker.io/${REMOTE_IMAGE}"
    sudo docker tag $DEPLOY_CONTAINER docker.io/${REMOTE_IMAGE}
    sudo docker push docker.io/${REMOTE_IMAGE}
    echo "Docker image pushed to Docker Hub!"
    exit 0
fi

# Setup X11 display forwarding
setup_x11() {
    # Set display if missing and X server available
    if [ -z "$DISPLAY" ] && command -v xset >/dev/null 2>&1 && xset q >/dev/null 2>&1; then
        export DISPLAY=:1
        echo "No DISPLAY set, using :1"
    fi
    
    # Enable X11 forwarding if possible
    if [ -n "$DISPLAY" ] && command -v xhost >/dev/null 2>&1 && xhost +local:docker 2>/dev/null; then
        echo "X11 forwarding enabled"
        return 0
    else
        echo "Headless environment - X11 disabled"
        export DISPLAY=""
        return 1
    fi
}

X11_ENABLED=false
setup_x11 && X11_ENABLED=true

# Mount entire /dev directory for dynamic device access (including hidraw for joycon)
# This allows JoyCon controllers to be detected even when connected after container launch
sudo chmod g+r+w /dev/input/*

# Detect GPU setup and set appropriate environment variables
echo "Detecting GPU setup..."
GPU_ENV_VARS=""

# Check if we have both integrated and discrete GPUs (hybrid/Optimus setup)
HAS_AMD_GPU=$(lspci | grep -i "vga\|3d\|display" | grep -i amd | wc -l)
HAS_INTEL_GPU=$(lspci | grep -i "vga\|3d\|display" | grep -i intel | wc -l)
HAS_NVIDIA_GPU=$(lspci | grep -i "vga\|3d\|display" | grep -i nvidia | wc -l)

if [[ "$HAS_INTEL_GPU" -gt 0 ]] || [[ "$HAS_AMD_GPU" -gt 0 ]] && [[ "$HAS_NVIDIA_GPU" -gt 0 ]]; then
    echo "Detected hybrid GPU setup (Intel/AMD integrated + NVIDIA discrete)"
    echo "Setting NVIDIA Optimus environment variables for proper rendering offload..."
    GPU_ENV_VARS="-e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __VK_LAYER_NV_optimus=NVIDIA_only"
else
    GPU_ENV_VARS=""
fi

# Set GPU runtime based on architecture
if is_arm64; then
    echo "Detected ARM64 architecture (Jetson Orin), using device access instead of nvidia runtime..."
    GPU_RUNTIME_ARGS="--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools"
else
    GPU_RUNTIME_ARGS="--gpus all --runtime=nvidia"
fi

# Common Docker run parameters
DOCKER_RUN_ARGS="--hostname $HOSTNAME \
    --user $USERNAME \
    --group-add $INPUT_GID \
    $GPU_RUNTIME_ARGS \
    --ipc=host \
    --network=host \
    --privileged \
    --device=/dev \
    $GPU_ENV_VARS \
    -p 5678:5678 \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -e USERNAME=$USERNAME \
    -e DECOUPLED_WBC_DIR="$DOCKER_HOME_DIR/Projects/$WORKTREE_NAME" \
    -e PYTHONPATH="$DOCKER_HOME_DIR/Projects/$WORKTREE_NAME" \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.ssh:$DOCKER_HOME_DIR/.ssh \
    -v $HOME/.gear:$DOCKER_HOME_DIR/.gear \
    -v $HOME/.Xauthority:$DOCKER_HOME_DIR/.Xauthority \
    -v $PROJECT_DIR:$DOCKER_HOME_DIR/Projects/$(basename "$PROJECT_DIR")
    --device /dev/snd \
    --group-add audio \
    -e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
    -v /run/user/$(id -u)/pulse/native:/run/user/$(id -u)/pulse/native \
    -v $HOME/.config/pulse/cookie:/home/$USERNAME/.config/pulse/cookie"

# Check if RL mode first, then handle container logic
if [ "$DEPLOY" = true ]; then
    # Deploy mode - use decoupled_wbc-deploy-${USERNAME} container
   
    # Always clean up old processes and create a new container
    # Kill all decoupled_wbc processes across containers to prevent message passing conflicts
    "$SCRIPT_DIR/kill_decoupled_wbc_processors.sh"
    echo "Creating new deploy container..."
       
    # Clean up old processes and create a fresh deploy container
    # Remove existing deploy container if it exists
    if sudo docker ps -a --format '{{.Names}}' | grep -q "^$DEPLOY_CONTAINER$"; then
        echo "Removing existing deploy container..."
        sudo docker rm -f $DEPLOY_CONTAINER
    fi
    sudo docker run -it --rm $DOCKER_RUN_ARGS \
        -w $DOCKER_HOME_DIR/Projects/$WORKTREE_NAME \
        --name $DEPLOY_CONTAINER \
        $DEPLOY_CONTAINER \
        /bin/bash -ic 'exec "$0" "$@"' \
        "${DOCKER_HOME_DIR}/Projects/${WORKTREE_NAME}/decoupled_wbc/docker/entrypoint/deploy.sh" \
        "${EXTRA_ARGS[@]}"
else
    # Bash mode - use decoupled_wbc-bash-${USERNAME} container
    if sudo docker ps -a --format '{{.Names}}' | grep -q "^$BASH_CONTAINER$"; then
        echo "Bash container exists, starting it..."
        sudo docker start $BASH_CONTAINER > /dev/null
        sudo docker exec -it $BASH_CONTAINER /bin/bash
    else
        echo "Creating new bash container with auto-install decoupled_wbc..."
        sudo docker run -it $DOCKER_RUN_ARGS \
            -w $DOCKER_HOME_DIR/Projects/$WORKTREE_NAME \
            --name $BASH_CONTAINER \
            $DEPLOY_CONTAINER \
            /bin/bash -ic 'exec "$0"' \
            "${DOCKER_HOME_DIR}/Projects/${WORKTREE_NAME}/decoupled_wbc/docker/entrypoint/bash.sh"
    fi
fi

# Cleanup X11 permissions
$X11_ENABLED && xhost -local:docker 2>/dev/null
