#!/bin/bash

echo "Setting up Qt environment for ARM64..."
QT_GCC_ARM64=/media/bytedance/newSpace/Qt6
export QT6_TOOLS=/media/bytedance/newSpace/Qt/Tools

export PATH=/media/bytedance/newSpace/Qt6/bin:$PATH
export PATH=/media/bytedance/newSpace/Qt6/include:$PATH
export PATH=/media/bytedance/newSpace/Qt/Tools/QtCreator/bin:$PATH
export PATH=/media/bytedance/newSpace/Qt/Tools/CMake/bin:$PATH

# Get the directory of the script
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $DIR
echo "Working directory: $DIR"

# Create build directory if it doesn't exist
BUILD_DIR="$DIR/build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p $BUILD_DIR
fi

# Clean build if requested
if [ "$1" = "--clean" ]; then
    echo "Cleaning build directory..."
    rm -rf $BUILD_DIR/*
fi

# Configure with CMake if CMakeCache.txt doesn't exist or --clean was specified
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ "$1" = "--clean" ]; then
    echo "Configuring project with CMake..."
    cmake -S $DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=$QT_GCC_ARM64
fi

# Build the project
echo "Building project..."
cmake --build $BUILD_DIR --config RelWithDebInfo

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"

    # The executable is directly output to the Redistributable directory
    TARGET_DIR="$DIR/../../../Redistributable/linux_aarch64"
    if [ -f "$TARGET_DIR/RobotDemoQt" ]; then
        echo "Build successful! Executable is at $TARGET_DIR/RobotDemoQt"
    else
        # Check if it was built to the linux directory instead
        LINUX_DIR="$DIR/../../../Redistributable/linux"
        if [ -f "$LINUX_DIR/RobotDemoQt" ]; then
            echo "Build successful! Executable is at $LINUX_DIR/RobotDemoQt"
            # Create linux_aarch64 directory if it doesn't exist
            if [ ! -d "$TARGET_DIR" ]; then
                mkdir -p "$TARGET_DIR"
            fi
            # Copy the executable to the correct location
            echo "Copying executable to the correct ARM64 location..."
            cp "$LINUX_DIR/RobotDemoQt" "$TARGET_DIR/"
            echo "Executable copied to $TARGET_DIR/RobotDemoQt"
        else
            echo "Warning: Executable not found at expected locations"
            echo "Searching for the executable..."
            find $DIR/../../../ -type f -executable -name "RobotDemoQt*"
        fi
    fi
else
    echo "Build failed!"
    exit 1
fi

echo "Build process completed."
