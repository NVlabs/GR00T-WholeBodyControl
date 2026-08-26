# FindONNXRuntime.cmake
# Find ONNX Runtime installation
#
# This module defines:
#  onnxruntime_FOUND - True if ONNX Runtime is found
#  onnxruntime_INCLUDE_DIRS - Include directories for ONNX Runtime
#  onnxruntime_LIBRARIES - Libraries for ONNX Runtime
#  onnxruntime::onnxruntime - Imported target for ONNX Runtime

# Try to find ONNX Runtime installation
find_path(onnxruntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS
    ENV onnxruntime_ROOT
    ENV onnxruntime_DIR
    /opt/onnxruntime
    /usr/local
    /usr
  PATH_SUFFIXES include
)

find_library(onnxruntime_LIBRARY
  NAMES onnxruntime
  PATHS
    ENV onnxruntime_ROOT
    ENV onnxruntime_DIR
    /opt/onnxruntime
    /usr/local
    /usr
  PATH_SUFFIXES lib lib64
)

# Handle the QUIETLY and REQUIRED arguments and set onnxruntime_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(onnxruntime
  FOUND_VAR onnxruntime_FOUND
  REQUIRED_VARS onnxruntime_LIBRARY onnxruntime_INCLUDE_DIR
  VERSION_VAR onnxruntime_VERSION
)

if(onnxruntime_FOUND)
  set(onnxruntime_INCLUDE_DIRS ${onnxruntime_INCLUDE_DIR})
  set(onnxruntime_LIBRARIES ${onnxruntime_LIBRARY})

  # Create imported target
  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}"
    )
  endif()

  mark_as_advanced(onnxruntime_INCLUDE_DIR onnxruntime_LIBRARY)
endif()

