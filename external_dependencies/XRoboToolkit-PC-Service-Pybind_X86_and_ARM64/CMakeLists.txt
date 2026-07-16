cmake_minimum_required(VERSION 3.14)

project(MyPybind11Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_host_system_information(RESULT ISA_NAME QUERY OS_PLATFORM) # Added: Important for UNIX specific logic
message(STATUS "OS_PLATFORM (ISA_NAME): ${ISA_NAME}")

include(GNUInstallDirs) # Add this line
find_package(pybind11 REQUIRED)

# Python Bindings for py_bindings.cpp
pybind11_add_module(xrobotoolkit_sdk MODULE bindings/py_bindings.cpp)

# Link xrobotoolkit_sdk module against pybind11
target_link_libraries(xrobotoolkit_sdk PRIVATE pybind11::module)

# Add include directories and link libraries for PXREARobotSDK to xrobotoolkit_sdk target
if(WIN32)
  target_include_directories(xrobotoolkit_sdk PUBLIC
      ${PROJECT_SOURCE_DIR}/include
  )
  target_link_directories(xrobotoolkit_sdk PUBLIC ${PROJECT_SOURCE_DIR}/lib)
  target_link_libraries(xrobotoolkit_sdk PUBLIC
      PXREARobotSDK.dll # Assuming this is how PXREARobotSDK is linked, mirroring ConsoleDemo
  )
endif()

if(UNIX)
  # ISA_NAME is set by cmake_host_system_information above
  if(ISA_NAME STREQUAL "aarch64")
    target_include_directories(xrobotoolkit_sdk PUBLIC
        ${PROJECT_SOURCE_DIR}/include/aarch64
    )
    target_link_directories(xrobotoolkit_sdk PUBLIC ${PROJECT_SOURCE_DIR}/lib/aarch64)
  else()
    target_include_directories(xrobotoolkit_sdk PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    )
    target_link_directories(xrobotoolkit_sdk PUBLIC ${PROJECT_SOURCE_DIR}/lib)
  endif()
  target_link_libraries(xrobotoolkit_sdk PUBLIC
      PXREARobotSDK
  )
endif()

# Install the Python module
install(TARGETS xrobotoolkit_sdk
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # Installs to <prefix>/lib
                                                # You might want a Python-specific path like:
                                                # DESTINATION lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
)
