#!/bin/bash

echo "Setting up Qt environment..."
QT_GCC_64=/home/pico/Qt6/6.6.3/gcc_64/
export QT6_TOOLS=/home/pico/Qt6/Tools

export PATH=/home/pico/Qt6/6.6.3/gcc_64/bin:$PATH
export PATH=/home/pico/Qt6/6.6.3/gcc_64/include:$PATH
export PATH=/home/pico/Qt6/Tools/QtCreator/bin:$PATH
export PATH=/home/pico/Qt6/Tools/CMake/bin:$PATH

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
    cmake -S $DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=$QT_GCC_64
fi

# Build the project
echo "Building project..."
cmake --build $BUILD_DIR --config RelWithDebInfo

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"

    # The executable is directly output to the Redistributable directory
    TARGET_DIR="$DIR/../../../Redistributable/linux"
    if [ -f "$TARGET_DIR/RobotDataRecorder" ]; then
        echo "Build successful! Executable is at $TARGET_DIR/RobotDataRecorder"
    else
        echo "Warning: Executable not found at expected location: $TARGET_DIR/RobotDataRecorder"
        echo "Searching for the executable..."
        find $DIR -type f -executable -name "RobotDataRecorder*"
    fi
else
    echo "Build failed!"
    exit 1
fi

echo "Build process completed."
