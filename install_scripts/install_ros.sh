#!/bin/bash

set -e  # Exit immediately on error

echo "🔄 Cleaning up incomplete or cached packages..."
conda clean --packages --tarballs --yes

echo "🔧 Adding RoboStack and conda-forge channels to the current environment..."
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging

# Optional: remove defaults to avoid conflicts (ignore error if not present)
echo "⚙️  Removing 'defaults' channel if present..."
conda config --env --remove channels defaults || true

echo "📦 Installing ROS 2 Humble Desktop from RoboStack..."
conda install -y ros-humble-desktop

echo "✅ Sourcing ROS environment from current conda env..."
source "$CONDA_PREFIX/setup.bash"

# Add ROS setup to bashrc if not already present
SETUP_LINE="source \"\$CONDA_PREFIX/setup.bash\" && export ROS_LOCALHOST_ONLY=1"
if ! grep -q "$SETUP_LINE" ~/.bashrc; then
    echo "📝 Adding ROS setup to ~/.bashrc..."
    echo "$SETUP_LINE" >> ~/.bashrc
    echo "✅ Added ROS setup to ~/.bashrc"
else
    echo "ℹ️ ROS setup already exists in ~/.bashrc"
fi

echo "🚀 Launching rviz2 to verify installation (will auto-close in 5 seconds)..."
timeout 5s rviz2
