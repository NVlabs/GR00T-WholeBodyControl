# Installation (Deployment)

## Prerequisites

**Required for all setups:**
- **Ubuntu 20.04/22.04/24.04** or other Debian-based Linux distributions
- **CUDA Toolkit** (for GPU acceleration)
- **TensorRT** (for inference optimization) — **Install this first!**
- **JetPack 6 on Jetson AGX Orin** (for onboard deployment)
- **JetPack 7.1 on Jetson AGX Thor** (for onboard deployment)
- Python 3.8+
- Git with LFS support

**Download TensorRT** from [NVIDIA Developer](https://developer.nvidia.com/tensorrt/download/10x):

| Platform | TensorRT Version |
|---|---|
| x86_64 (Desktop) | **10.13** (required) |
| Jetson / G1 onboard Orin | **10.7** (required; requires JetPack 6 — [flashing guide](../references/jetpack6.md)) |
| Jetson / G1 onboard Thor | **10.13** (required) |

```{tip}
Download the **TAR** package (not the DEB one) so you can extract TensorRT to any location. The archive is ~10 GB; consider using `pv` to monitor progress:
```

```{danger}
You **must** use the exact TensorRT versions listed above. Using a different version is known to produce incorrect inference results — the planner will output wrong motion, which can cause dangerous robot behavior.
```

```sh
sudo apt-get install -y pv
pv TensorRT-*.tar.gz | tar -xz -f -
```

Move the unzipped TensorRT to `~/TensorRT` (or similar) and add to your `~/.bashrc`:

```sh
export TensorRT_ROOT=$HOME/TensorRT
```

## Clone the Repository

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull          # make sure all large files are fetched
```

## Setup

### Native Development (Recommended)

**Advantages:** Direct system installation, faster builds, production-ready.

```{warning}
For G1 onboard deployment, we require the onboard Orin to be upgraded to Jetpack 6 to support TensorRT. Please follow the [flashing guide](../references/jetpack6.md) for upgrading!
AGX Thor requires JetPack 7.1.
```

**Prerequisites:**
- Basic development tools (cmake, git, etc.)
- (Optional) ROS2 if you plan to use ROS2-based input/output

**Setup steps:**

1. **Install system dependencies:**

```sh
cd gear_sonic_deploy
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh
```

2. **Set up environment:**

```sh
source scripts/setup_env.sh
```

The setup script will automatically:
- Configure TensorRT environment
- Set up all necessary paths

For convenience, you can add the environment setup to your shell profile:

```sh
echo "source $(pwd)/scripts/setup_env.sh" >> ~/.bashrc
```

3. **Build the project:**

```sh
just build
```

### Docker (ROS2 Development Environment)

We provide a unified Docker environment with ROS2 Humble, supporting x86_64 and Jetson platforms.

**Prerequisites:**
- Docker installed and user added to docker group
- `TensorRT_ROOT` environment variable set on x86_64 hosts
- For Jetson: JetPack 6.1+ on AGX Orin or JetPack 7.1 on AGX Thor

**Quick Setup:**

```sh
# 1. Add user to docker group (one-time setup)
sudo usermod -aG docker $USER
newgrp docker

# 2. Set TensorRT path on x86_64 (add to ~/.bashrc for persistence)
export TensorRT_ROOT=/path/to/TensorRT

# 3. Launch container
cd gear_sonic_deploy
./docker/run-ros2-dev.sh
```

**Options:**

```sh
./docker/run-ros2-dev.sh               # Standard build (fast)
./docker/run-ros2-dev.sh --rebuild     # Force rebuild
./docker/run-ros2-dev.sh --with-opengl # Include OpenGL for visualization (RViz, Gazebo)
```

**Architecture Support:**
- **x86_64**: CUDA 12.4.1 (requires NVIDIA driver 550+)
- **Jetson**: AGX Orin and AGX Thor

**Inside the container:**

```sh
source scripts/setup_env.sh # set up dependency
just build                  # Build
just --list                 # Show all commands
```

**Troubleshooting:**
- If you get "permission denied", ensure you're in the docker group
- On x86_64, `TensorRT_ROOT` must be set on the **host** before starting the container
- For Jetson: Run `source scripts/setup_env.sh` on host first (sets jetson_clocks)




