#!/bin/bash

set -e

echo "🤖 Installing ROS2 Humble..."

# Detect system type and architecture
ARCH=$(uname -m)
OS_ID=$(lsb_release -si 2>/dev/null || echo "Unknown")
OS_VERSION=$(lsb_release -sr 2>/dev/null || echo "Unknown")

echo "🔍 System: $OS_ID $OS_VERSION ($ARCH)"

# Check if running Ubuntu (ROS2 Humble officially supports Ubuntu 22.04)
if [ "$OS_ID" != "Ubuntu" ]; then
    echo "⚠️  Warning: ROS2 Humble is officially supported on Ubuntu 22.04 (Jammy Jellyfish)"
    echo "   Your system ($OS_ID $OS_VERSION) may have compatibility issues"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [ "$OS_VERSION" != "22.04" ] && [ "$OS_ID" = "Ubuntu" ]; then
    echo "⚠️  Warning: ROS2 Humble is officially supported on Ubuntu 22.04"
    echo "   Your version ($OS_VERSION) may have compatibility issues"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi



# Set locale
echo "🌍 Setting up locale..."
sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "✅ Locale configured"

# Enable Ubuntu Universe repository
echo "📦 Enabling Ubuntu Universe repository..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository universe -y

# Add ROS2 GPG key
echo "🔑 Adding ROS2 GPG key..."
sudo apt-get update
sudo apt-get install -y curl gnupg lsb-release

if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "✅ ROS2 GPG key added"
else
    echo "✅ ROS2 GPG key already exists"
fi

# Add ROS2 repository to sources list
echo "📋 Adding ROS2 repository..."
ROS2_LIST="/etc/apt/sources.list.d/ros2.list"

if [ ! -f "$ROS2_LIST" ]; then
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee $ROS2_LIST > /dev/null
    echo "✅ ROS2 repository added"
else
    echo "✅ ROS2 repository already configured"
fi

# Update apt cache
echo "🔄 Updating package cache..."
sudo apt-get update

# Upgrade packages (optional but recommended)
echo "⬆️  Upgrading packages..."
sudo apt-get upgrade -y

# Install ROS2 Humble Desktop (full installation with GUI tools)
echo "📦 Installing ROS2 Humble Desktop..."
echo "   This may take several minutes..."

if sudo apt-get install -y ros-humble-desktop; then
    echo "✅ ROS2 Humble Desktop installed successfully!"
else
    echo "❌ Failed to install ROS2 Humble Desktop"
    echo "   Trying ROS2 Humble Base instead..."
    if sudo apt-get install -y ros-humble-ros-base; then
        echo "✅ ROS2 Humble Base installed successfully!"
    else
        echo "❌ Failed to install ROS2 Humble"
        exit 1
    fi
fi

# Install development tools
echo "🔧 Installing ROS2 development tools..."
sudo apt-get install -y \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-argcomplete

echo "✅ Development tools installed"

# Initialize rosdep
echo "🔧 Initializing rosdep..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
    echo "✅ rosdep initialized"
else
    echo "✅ rosdep already initialized"
fi

rosdep update
echo "✅ rosdep updated"

# Set up environment
echo "🌱 Setting up ROS2 environment..."

SETUP_SCRIPT="/opt/ros/humble/setup.bash"
BASHRC="$HOME/.bashrc"

# Check if ROS2 sourcing is already in .bashrc
if ! grep -q "source $SETUP_SCRIPT" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# ROS2 Humble setup" >> "$BASHRC"
    echo "source $SETUP_SCRIPT" >> "$BASHRC"
    echo "✅ Added ROS2 Humble to .bashrc"
else
    echo "✅ ROS2 Humble already in .bashrc"
fi

# Source ROS2 setup for current session
source $SETUP_SCRIPT

echo ""
echo "🎉 ROS2 Humble installation complete!"
echo ""
echo "📋 Installation Summary:"
echo "   - ROS2 Humble Desktop: Installed"
echo "   - Development tools: Installed"
echo "   - rosdep: Initialized and updated"
echo "   - Environment: Configured in ~/.bashrc"
echo ""
echo "🚀 Next steps:"
echo "   1. Open a new terminal or run: source ~/.bashrc"
echo "   2. Verify installation: ros2 --help"
echo "   3. Test with: ros2 run demo_nodes_cpp talker"
echo ""
echo "📚 Useful commands:"
echo "   - Check ROS2 version: ros2 --version"
echo "   - List available packages: ros2 pkg list"
echo "   - Create a workspace: mkdir -p ~/ros2_ws/src && cd ~/ros2_ws && colcon build"
echo ""
echo "📖 Documentation: https://docs.ros.org/en/humble/"
echo ""

