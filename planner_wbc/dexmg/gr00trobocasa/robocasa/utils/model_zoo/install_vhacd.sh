#!/bin/bash
#
# Install V-HACD 4.0.
# Adapted from https://github.com/kevinzakka/obj2mjcf/blob/main/install_vhacd.sh

# Check that cmake is installed.
t=`which cmake`
if [ -z "$t" ]; then
  echo "You need cmake to install V-HACD." 1>&2
  exit 1
fi

# Clone and build executable.
# TODO(kevin): Peg to a 4.0 release when available, see #113.
git clone https://github.com/kmammou/v-hacd.git
cd v-hacd/app
cmake CMakeLists.txt
cmake --build .

# Add executable to /usr/local/bin.
sudo ln -s "$PWD/TestVHACD" /usr/local/bin/TestVHACD
