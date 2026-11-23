#!/usr/bin/env bash
# CycloneDDS C library, required to build the cyclonedds Python package.

CDDS_PREFIX="${CYCLONEDDS_HOME:-$VIRTUAL_ENV/cyclonedds}"
if [ ! -f "$CDDS_PREFIX/lib/libddsc.so" ]; then
    if ! command -v cmake &>/dev/null; then
        uv pip install cmake
    fi
    echo "[INFO] Building CycloneDDS releases/0.10.x → $CDDS_PREFIX …"
    # The maintenance branch fixes glibc FORTIFY_SOURCE failures in the
    # original 0.10.2 C library on modern Ubuntu.
    CDDS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sonic-cyclonedds.XXXXXX")"
    git clone -b releases/0.10.x --depth 1 \
        https://github.com/eclipse-cyclonedds/cyclonedds.git "$CDDS_DIR/src"
    cmake -S "$CDDS_DIR/src" -B "$CDDS_DIR/build" \
        -DCMAKE_INSTALL_PREFIX="$CDDS_PREFIX" \
        -DENABLE_SHM=OFF \
        -DBUILD_DDSPERF=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTING=OFF
    cmake --build "$CDDS_DIR/build" -j"$(nproc)"
    cmake --install "$CDDS_DIR/build"
    rm -rf "$CDDS_DIR"
    echo "[OK] CycloneDDS installed at $CDDS_PREFIX"
else
    echo "[OK] CycloneDDS already present at $CDDS_PREFIX (libddsc.so found)"
fi
export CYCLONEDDS_HOME="$CDDS_PREFIX"
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
