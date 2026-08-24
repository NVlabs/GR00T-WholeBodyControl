set(EXECUTABLE_NAME g1_deploy_onnx_ref)

# ROS2 discovery can be controlled via env var HAS_ROS2:
# - 1: try ROS2 (find_package QUIET)
# - 0 or unset: disable ROS2 entirely (skip find_package)
set(ROS2_ENV_REQUEST "off")
if(DEFINED ENV{HAS_ROS2})
  if("$ENV{HAS_ROS2}" STREQUAL "1")
    set(ROS2_ENV_REQUEST "on")
  elseif("$ENV{HAS_ROS2}" STREQUAL "0")
    set(ROS2_ENV_REQUEST "off")
  else()
    message(WARNING "HAS_ROS2 should be 0 or 1; got '$ENV{HAS_ROS2}'. Falling back to auto-detect.")
  endif()
endif()

if(ROS2_ENV_REQUEST STREQUAL "off")
  set(rclcpp_FOUND FALSE)
  set(std_msgs_FOUND FALSE)
  message(STATUS "ROS2 disabled (HAS_ROS2 unset or 0)")
else()
  # Try to locate ROS2 only when explicitly enabled
  find_package(rclcpp QUIET)
  find_package(std_msgs QUIET)
endif()

# Find msgpack for message serialization (used by ROS2 and ZMQ handlers)
# msgpack is header-only and doesn't provide pkg-config, so we check for the header directly
find_path(MSGPACK_INCLUDE_DIR msgpack.hpp
  PATHS /usr/include /usr/local/include
  DOC "msgpack C++ header location"
)

if(MSGPACK_INCLUDE_DIR)
  message(STATUS "✅ msgpack found at: ${MSGPACK_INCLUDE_DIR}")
  set(MSGPACK_INCLUDE_DIRS ${MSGPACK_INCLUDE_DIR})
else()
  message(FATAL_ERROR "msgpack not found. Install libmsgpack-dev (Ubuntu/Debian) or msgpack-cxx (Arch) or msgpack-devel (RHEL/CentOS)")
endif()

file(GLOB_RECURSE PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_deploy_onnx_ref.cpp)
list(REMOVE_ITEM PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_shadow_gate.cpp)
list(REMOVE_ITEM PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_live_shadow.cpp)
list(REMOVE_ITEM PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_active_gantry.cpp)
list(REMOVE_ITEM PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_native124_active_gantry.cpp)
list(REMOVE_ITEM PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_gantry_hold_smoke.cpp)


# Optional AddressSanitizer support
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
  message(STATUS "🔬 Building ${EXECUTABLE_NAME} with AddressSanitizer")
  target_compile_options(${EXECUTABLE_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(${EXECUTABLE_NAME} PRIVATE -fsanitize=address)
endif()

# Detect architecture and conditionally set CUDA library paths
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # Check for Thor (no DLA support)
  set(IS_THOR FALSE)
  if(DEFINED ENV{IS_THOR} AND "$ENV{IS_THOR}" STREQUAL "true")
    set(IS_THOR TRUE)
  elseif(EXISTS "/proc/device-tree/model")
    file(READ "/proc/device-tree/model" DEVICE_MODEL)
    string(FIND "${DEVICE_MODEL}" "Thor" THOR_POS)
    if(NOT THOR_POS EQUAL -1)
      set(IS_THOR TRUE)
    endif()
  endif()

  # ARM/Jetson platform - conditionally add DLA support
  # Dynamically find CUDA installation for ARM
  if(DEFINED ENV{CUDAToolkit_ROOT})
    set(CUDA_ARM_LIB_PATH "$ENV{CUDAToolkit_ROOT}/targets/aarch64-linux/lib")
  elseif(EXISTS "/usr/local/cuda/targets/aarch64-linux/lib")
    set(CUDA_ARM_LIB_PATH "/usr/local/cuda/targets/aarch64-linux/lib")
  else()
    # Find any available CUDA version for ARM
    file(GLOB CUDA_ARM_DIRS "/usr/local/cuda-*/targets/aarch64-linux/lib")
    if(CUDA_ARM_DIRS)
      list(GET CUDA_ARM_DIRS 0 CUDA_ARM_LIB_PATH)
    else()
      message(WARNING "No CUDA ARM libraries found, using fallback path")
      set(CUDA_ARM_LIB_PATH "/usr/local/cuda/lib64")
    endif()
  endif()

  set(CUDA_LINK_DIRS ${CUDA_ARM_LIB_PATH})

  # DLA support: only for non-Thor Jetson devices
  if(IS_THOR)
    set(CUDLA_LIBRARY "")
    message(STATUS "🚀 ARM64 (Thor) detected - building without DLA support")
    message(STATUS "   Using CUDA library path: ${CUDA_ARM_LIB_PATH}")
    message(STATUS "   Thor doesn't have DLA engines")
  else()
    # Match bare-metal Jetson builds: link cudla from the CUDA toolkit (targets/aarch64-linux/lib)
    set(CUDLA_LIBRARY "cudla")
    message(STATUS "✅ ARM64 detected - enabling DLA (cudla) support")
    message(STATUS "   Using CUDA library path: ${CUDA_ARM_LIB_PATH}")
    message(STATUS "   Using CUDLA library: -l${CUDLA_LIBRARY}")
  endif()
  
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  # x86_64 platform - no DLA support needed
  # Dynamically find CUDA installation for x86_64
  if(DEFINED ENV{CUDAToolkit_ROOT})
    set(CUDA_X64_LIB_PATH "$ENV{CUDAToolkit_ROOT}/lib64")
  elseif(EXISTS "/usr/local/cuda/lib64")
    set(CUDA_X64_LIB_PATH "/usr/local/cuda/lib64")
  else()
    # Find any available CUDA version for x86_64
    file(GLOB CUDA_X64_DIRS "/usr/local/cuda-*/lib64")
    if(CUDA_X64_DIRS)
      list(GET CUDA_X64_DIRS 0 CUDA_X64_LIB_PATH)
    else()
      message(WARNING "No CUDA x86_64 libraries found, using fallback path")
      set(CUDA_X64_LIB_PATH "/usr/local/cuda/lib64")
    endif()
  endif()
  
  set(CUDA_LINK_DIRS ${CUDA_X64_LIB_PATH})
  set(CUDLA_LIBRARY "")
  message(STATUS "✅ x86_64 detected - building without DLA support")
  message(STATUS "   Using CUDA library path: ${CUDA_X64_LIB_PATH}")
endif()

# ZMQ support (required for main executable and tests)
find_package(PkgConfig QUIET)
if(PkgConfig)
  pkg_check_modules(ZMQ QUIET libzmq)
endif()

find_path(ZMQ_INCLUDE_DIR zmq.h
  PATHS /usr/include /usr/local/include)
find_library(ZMQ_LIBRARY zmq
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)

if(NOT ZMQ_INCLUDE_DIR OR NOT ZMQ_LIBRARY)
  message(FATAL_ERROR "❌ libzmq is required but not found. Please install libzmq-dev (apt) or zeromq (brew)")
endif()

# Add ZMQ to main executable
message(STATUS "✅ ZMQ support enabled for main executable")

# Build ZMQ tests
if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
  add_executable(zmq_pose_subscriber_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/zmq_pose_subscriber_test.cpp)
  target_include_directories(zmq_pose_subscriber_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIR})
  target_link_libraries(zmq_pose_subscriber_test PRIVATE ${ZMQ_LIBRARY})
  set_target_properties(zmq_pose_subscriber_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  message(STATUS "✅ Added zmq_pose_subscriber_test (ZMQ found)")
  
  # Test for connecting to Python pose estimation server
  add_executable(zmq_python_sender_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/zmq_python_sender_test.cpp)
  target_include_directories(zmq_python_sender_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIR})
  target_link_libraries(zmq_python_sender_test PRIVATE ${ZMQ_LIBRARY})
  set_target_properties(zmq_python_sender_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  message(STATUS "✅ Added zmq_python_sender_test (ZMQ found)")
else()
  message(WARNING "⚠️  libzmq not found - skipping zmq_pose_subscriber_test")
endif()

find_package(Eigen3 REQUIRED)

set(EXE_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${ZMQ_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)


# Build a static library, used by both the main executable and the tests: 
add_library(g1_deploy_shared STATIC ${PROJECT_SRCS})
target_include_directories(g1_deploy_shared PUBLIC ${EXE_INCLUDES})

# List of libs to link for both the main executable and the tests
set(EXE_LIBS
  g1_deploy_shared
  audio_thread
  unitree_sdk2 ZLIB::ZLIB
  onnxruntime::onnxruntime TRTInference
  CUDA::cudart
  ${TensorRT_nvinfer_LIBRARY} ${TensorRT_nvonnxparser_LIBRARY}
  ${CUDLA_LIBRARY}
  ${ZMQ_LIBRARY}
)


# build the main executable:
add_executable(${EXECUTABLE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_deploy_onnx_ref.cpp)
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/src)

target_link_directories(${EXECUTABLE_NAME} PRIVATE ${CUDA_LINK_DIRS})
target_link_libraries(
  ${EXECUTABLE_NAME} PRIVATE ${PROJECT_NAME}
  ${EXE_LIBS}
)

# Setting our output directory and C++ standard
set_target_properties(
  ${EXECUTABLE_NAME}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
             OUTPUT_NAME ${EXECUTABLE_NAME}
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)

# Read-only native 23-DoF artifact/embodiment probe. Keep this target isolated
# from g1_deploy_shared and all command, TensorRT, audio, input, and output
# surfaces. Its translation unit includes only the Unitree subscriber API.
add_executable(
  g1_true23_shadow_gate
  ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_shadow_gate.cpp)
target_include_directories(
  g1_true23_shadow_gate
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_link_libraries(
  g1_true23_shadow_gate
  PRIVATE unitree_sdk2 onnxruntime::onnxruntime Threads::Threads)
target_compile_options(
  g1_true23_shadow_gate
  PRIVATE -fno-fast-math -fno-finite-math-only)
set_target_properties(
  g1_true23_shadow_gate
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    OUTPUT_NAME g1_true23_shadow_gate
    # Windows checkouts preserve Linux SONAME symlinks as tiny text files.
    # Prefer the installer-staged native libraries before transitive vendored
    # link directories so the read-only probe remains runnable from WSL.
    BUILD_RPATH "/usr/local/lib;/usr/local/lib64"
    INSTALL_RPATH "/usr/local/lib;/usr/local/lib64"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

# Dependency-free fail-closed contract harness. This never initializes DDS.
add_executable(
  true23_shadow_gate_harness
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/true23_shadow_gate_harness.cpp)
target_include_directories(
  true23_shadow_gate_harness
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_compile_options(
  true23_shadow_gate_harness
  PRIVATE -fno-fast-math -fno-finite-math-only)
set_target_properties(
  true23_shadow_gate_harness
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    OUTPUT_NAME true23_shadow_gate_harness
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
enable_testing()
add_test(
  NAME true23_shadow_gate_harness
  COMMAND true23_shadow_gate_harness)
add_test(
  NAME true23_mujoco_promotion_harness
  COMMAND true23_shadow_gate_harness)
add_test(
  NAME true23_shadow_gate_help
  COMMAND g1_true23_shadow_gate --help)
add_test(
  NAME true23_shadow_gate_forbidden_surface
  COMMAND
    ${CMAKE_COMMAND}
    -DBINARY=$<TARGET_FILE:g1_true23_shadow_gate>
    -P
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_true23_shadow_binary.cmake)

# Read-only causal PICO + LowState ONNX inference. This target links only the
# subscriber/runtime dependencies and is audited for absence of command APIs.
add_executable(
  g1_true23_live_shadow
  ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_live_shadow.cpp)
target_include_directories(
  g1_true23_live_shadow
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${ZMQ_INCLUDE_DIR})
target_link_libraries(
  g1_true23_live_shadow
  PRIVATE
    unitree_sdk2
    onnxruntime::onnxruntime
    Threads::Threads
    ${ZMQ_LIBRARY})
target_compile_options(
  g1_true23_live_shadow
  PRIVATE -fno-fast-math -fno-finite-math-only)
set_target_properties(
  g1_true23_live_shadow
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    OUTPUT_NAME g1_true23_live_shadow
    BUILD_RPATH "/usr/local/lib;/usr/local/lib64"
    INSTALL_RPATH "/usr/local/lib;/usr/local/lib64"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

add_executable(
  true23_live_shadow_core_harness
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/true23_live_shadow_core_harness.cpp)
target_include_directories(
  true23_live_shadow_core_harness
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_compile_options(
  true23_live_shadow_core_harness
  PRIVATE -fno-fast-math -fno-finite-math-only)
set_target_properties(
  true23_live_shadow_core_harness
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    OUTPUT_NAME true23_live_shadow_core_harness
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
add_test(
  NAME true23_live_shadow_core_harness
  COMMAND true23_live_shadow_core_harness)
add_test(
  NAME true23_live_shadow_help
  COMMAND g1_true23_live_shadow --help)
add_test(
  NAME true23_live_shadow_forbidden_surface
  COMMAND
    ${CMAKE_COMMAND}
    -DBINARY=$<TARGET_FILE:g1_true23_live_shadow>
    -DREQUIRE_SAFE_OUTPUT=ON
    -P
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_true23_shadow_binary.cmake)

# Native True23 first-actuation runtime.  Unlike shadow targets this contains
# LowCmd by design, but creates that mutation surface only after its internal
# five-sample CRC/mode/telemetry gate.  It intentionally does not link the
# generic 29-DoF deployment stack.
add_executable(
  g1_true23_active_gantry
  ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_active_gantry.cpp)
target_include_directories(
  g1_true23_active_gantry
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${ZMQ_INCLUDE_DIR})
target_link_libraries(
  g1_true23_active_gantry
  PRIVATE
    unitree_sdk2
    onnxruntime::onnxruntime
    Threads::Threads
    ${ZMQ_LIBRARY})
target_compile_options(
  g1_true23_active_gantry
  PRIVATE -fno-fast-math -fno-finite-math-only)
set_target_properties(
  g1_true23_active_gantry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    OUTPUT_NAME g1_true23_active_gantry
    BUILD_RPATH "/usr/local/lib;/usr/local/lib64"
    INSTALL_RPATH "/usr/local/lib;/usr/local/lib64"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

# Separate pinned native124 stage-one gantry runtime. It reuses the reviewed
# active safety core but cannot accept causal-pair promotion artifacts.
add_executable(
  g1_true23_native124_active_gantry
  ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_native124_active_gantry.cpp)
target_include_directories(
  g1_true23_native124_active_gantry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/ ${ZMQ_INCLUDE_DIR})
target_link_libraries(
  g1_true23_native124_active_gantry
  PRIVATE unitree_sdk2 onnxruntime::onnxruntime Threads::Threads ${ZMQ_LIBRARY})
target_compile_options(
  g1_true23_native124_active_gantry
  PRIVATE -fno-fast-math -fno-finite-math-only)
set_target_properties(
  g1_true23_native124_active_gantry
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    OUTPUT_NAME g1_true23_native124_active_gantry
    BUILD_RPATH "/usr/local/lib;/usr/local/lib64"
    INSTALL_RPATH "/usr/local/lib;/usr/local/lib64"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# Keep exact ORT used at link time beside standalone true23 binaries. Robot
# test hosts may have an older system ORT that cannot load current exports.
foreach(TRUE23_ORT_TARGET
    g1_true23_shadow_gate
    g1_true23_live_shadow
    g1_true23_active_gantry
    g1_true23_native124_active_gantry)
  set_property(
    TARGET ${TRUE23_ORT_TARGET}
    APPEND PROPERTY BUILD_RPATH "\$ORIGIN/lib")
  add_custom_command(
    TARGET ${TRUE23_ORT_TARGET}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${PROJECT_SOURCE_DIR}/target/release/lib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${onnxruntime_LIBRARY}"
            "${PROJECT_SOURCE_DIR}/target/release/lib/libonnxruntime.so.1"
    VERBATIM)
endforeach()

add_executable(
  true23_active_gantry_core_harness
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/true23_active_gantry_core_harness.cpp)
target_include_directories(
  true23_active_gantry_core_harness
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_compile_options(
  true23_active_gantry_core_harness
  PRIVATE -fno-fast-math -fno-finite-math-only)
set_target_properties(
  true23_active_gantry_core_harness
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    OUTPUT_NAME true23_active_gantry_core_harness
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
add_test(
  NAME true23_active_gantry_core_harness
  COMMAND true23_active_gantry_core_harness)
add_test(
  NAME true23_active_gantry_help
  COMMAND g1_true23_active_gantry --help)
add_test(
  NAME true23_native124_active_gantry_help
  COMMAND g1_true23_native124_active_gantry --help)
add_test(
  NAME true23_active_gantry_forbidden_surface
  COMMAND
    ${CMAKE_COMMAND}
    -DBINARY=$<TARGET_FILE:g1_true23_active_gantry>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_active_gantry.cpp
    -P
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_true23_active_gantry_binary.cmake)

# Policy-free, fixed-gain, five-second gantry hold smoke.  This provides a
# truthful first DDS/motor contact test when no policy is promotion-eligible.
add_executable(
  g1_true23_gantry_hold_smoke
  ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_gantry_hold_smoke.cpp)
target_include_directories(
  g1_true23_gantry_hold_smoke
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_link_libraries(
  g1_true23_gantry_hold_smoke
  PRIVATE unitree_sdk2 Threads::Threads)
target_compile_options(
  g1_true23_gantry_hold_smoke
  PRIVATE -fno-fast-math -fno-finite-math-only)
set_target_properties(
  g1_true23_gantry_hold_smoke
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    OUTPUT_NAME g1_true23_gantry_hold_smoke
    BUILD_RPATH "/usr/local/lib;/usr/local/lib64"
    INSTALL_RPATH "/usr/local/lib;/usr/local/lib64"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
add_test(
  NAME true23_gantry_hold_smoke_help
  COMMAND g1_true23_gantry_hold_smoke --help)
add_test(
  NAME true23_gantry_hold_smoke_forbidden_surface
  COMMAND
    ${CMAKE_COMMAND}
    -DBINARY=$<TARGET_FILE:g1_true23_gantry_hold_smoke>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/g1_true23_gantry_hold_smoke.cpp
    -P
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_true23_hold_smoke_binary.cmake)

# Optional ROS2 support for main executable and tests
if(rclcpp_FOUND AND std_msgs_FOUND)
  # Load ROS2 configuration (handles both main executable and tests)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS2.cmake)
  
  # Add ROS2 support to main executable (requires msgpack)
  target_include_directories(${EXECUTABLE_NAME} PRIVATE ${ROS2_INCLUDE_DIRS} ${MSGPACK_INCLUDE_DIRS})
  target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${ROS2_LIBS} pthread)
  target_compile_definitions(${EXECUTABLE_NAME} PRIVATE HAS_ROS2=1)
  
  message(STATUS "✅ ROS2InputHandler support enabled with msgpack")
else()
  target_compile_definitions(${EXECUTABLE_NAME} PRIVATE HAS_ROS2=0)
  message(STATUS "ROS2 not found - building without ROS2InputHandler support")
endif()

# test executable:
include(FetchContent)

# Download and make GoogleTest available
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

find_package(GTest REQUIRED)

file(GLOB_RECURSE TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/*.cpp)

add_executable(run_tests ${TEST_SRCS})
target_link_directories(run_tests PRIVATE ${CUDA_LINK_DIRS})
target_link_libraries(run_tests GTest::GTest GTest::Main ${EXE_LIBS})
target_include_directories(run_tests PUBLIC ${EXE_INCLUDES})
target_compile_definitions(
  run_tests
  PRIVATE GEAR_SONIC_DEPLOY_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# Setting our output directory and C++ standard
set_target_properties(
  run_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
             OUTPUT_NAME run_tests
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)
