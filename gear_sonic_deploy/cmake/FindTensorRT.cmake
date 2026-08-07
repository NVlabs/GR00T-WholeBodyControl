# ~~~
# Copyright 2021 Olivier Le Doeuff
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# This module defines the following variables:
#
# - TensorRT_FOUND: A boolean specifying whether or not TensorRT was found.
# - TensorRT_VERSION: The exact version of TensorRT found
# - TensorRT_VERSION_MAJOR: The major version of TensorRT.
# - TensorRT_VERSION_MINOR: The minor version of TensorRT.
# - TensorRT_VERSION_PATCH: The patch version of TensorRT.
# - TensorRT_VERSION_TWEAK: The tweak version of TensorRT.
# - TensorRT_INCLUDE_DIRS: The path to TensorRT ``include`` folder containing the header files    required to compile a project linking against TensorRT.
# - TensorRT_LIBRARY_DIRS: The path to TensorRT library directory that contains libraries.
#
# This module create following targets:
# - trt::nvinfer
# - trt::nvinfer_plugin
# - trt::nvonnxparser
# - trt::nvparsers
# This script was inspired from https://github.com/NicolasIRAGNE/CMakeScripts
# This script was inspired from https://github.com/NVIDIA/tensorrt-laboratory/blob/master/cmake/FindTensorRT.cmake
#
# Hints
# ^^^^^
# A user may set ``TensorRT_ROOT`` to an installation root to tell this module where to look.
# ~~~

if(NOT TensorRT_FIND_COMPONENTS)
  set(TensorRT_FIND_COMPONENTS nvinfer nvinfer_plugin nvonnxparser nvparsers)
endif()
set(TensorRT_LIBRARIES)

# find the include directory of TensorRT
find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES include
)

string(FIND ${TensorRT_INCLUDE_DIR} "NOTFOUND" _include_dir_notfound)
if(NOT _include_dir_notfound EQUAL -1)
  if(TensorRT_FIND_REQUIRED)
    message(FATAL_ERROR "Fail to find TensorRT, please set TensorRT_ROOT. Include path not found.")
  endif()
  return()
endif()
set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

# Extract version of tensorrt
if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_TWEAK REGEX "^#define NV_TENSORRT_BUILD [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
  string(REGEX REPLACE "^#define NV_TENSORRT_BUILD ([0-9]+).*$" "\\1" TensorRT_VERSION_TWEAK "${TensorRT_TWEAK}")
  set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_TWEAK}")
endif()

function(_find_trt_component component)

  # Find library for component (ie nvinfer, nvparsers, etc...)
  find_library(
    TensorRT_${component}_LIBRARY
    NAMES ${component}
    PATHS ${TensorRT_ROOT} ${TENSORRT_LIBRARY_DIR} ENV TensorRT_ROOT
  )

  string(FIND ${TensorRT_${component}_LIBRARY} "NOTFOUND" _library_not_found)

  if(NOT TensorRT_LIBRARY_DIR)
    get_filename_component(_path ${TensorRT_${component}_LIBRARY} DIRECTORY)
    set(TensorRT_LIBRARY_DIR
        "${_path}"
        CACHE INTERNAL "TensorRT_LIBRARY_DIR"
    )
  endif()

  if(NOT TensorRT_LIBRARY_DIRS)
    get_filename_component(_path ${TensorRT_${component}_LIBRARY} DIRECTORY)
    set(TensorRT_LIBRARY_DIRS
        "${_path}"
        CACHE INTERNAL "TensorRT_LIBRARY_DIRS"
    )
  endif()

  # Library found, and doesn't already exists
  if(_library_not_found EQUAL -1 AND NOT TARGET trt::${component})
    set(TensorRT_${component}_FOUND
        TRUE
        CACHE INTERNAL "Found ${component}"
    )

    # Create a target
    add_library(trt::${component} IMPORTED INTERFACE)
    target_include_directories(trt::${component} SYSTEM INTERFACE "${TensorRT_INCLUDE_DIRS}")
    target_link_libraries(trt::${component} INTERFACE "${TensorRT_${component}_LIBRARY}")
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES} ${TensorRT_${component}_LIBRARY})
  endif()

endfunction()

# Find each components
foreach(component IN LISTS TensorRT_FIND_COMPONENTS)
  _find_trt_component(${component})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT HANDLE_COMPONENTS VERSION_VAR TensorRT_VERSION REQUIRED_VARS TensorRT_INCLUDE_DIR)

