#!/bin/bash
set -e

# Source virtual environment and ROS2
source ${HOME}/venv/bin/activate
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1

# Install external dependencies
echo "Current directory: $(pwd)"
echo "Installing dependencies..."

# Install Unitree SDK and LeRobot
if [ -d "external_dependencies/unitree_sdk2_python" ]; then
    cd external_dependencies/unitree_sdk2_python/
    uv pip install -e . --no-deps
    cd ../..
fi

# Install project
if [ -f "pyproject.toml" ]; then
    UV_GIT_LFS=1 uv pip install -e ".[dev]"
fi
