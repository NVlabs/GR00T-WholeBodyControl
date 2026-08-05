cmake_minimum_required(VERSION 3.14)

project(ConsoleDemo LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_host_system_information(RESULT ISA_NAME QUERY OS_PLATFORM)
message("OS_PLATFORM: ${ISA_NAME}")

add_executable(ConsoleDemo
  main.cpp
)

# Set output directory for Windows
if(WIN32)
  set_target_properties(ConsoleDemo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../../../Redistributable/win"
    RUNTIME_OUTPUT_NAME "ConsoleDemo"
  )
endif()

# Set output directory for Linux
if(UNIX)
  if(ISA_NAME STREQUAL "aarch64")
    set_target_properties(ConsoleDemo PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../../../Redistributable/linux_aarch64"
      RUNTIME_OUTPUT_NAME "ConsoleDemo"
    )
  else()
    set_target_properties(ConsoleDemo PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../../../Redistributable/linux"
      RUNTIME_OUTPUT_NAME "ConsoleDemo"
    )
  endif()
endif()



if(WIN32)
  target_include_directories(ConsoleDemo PUBLIC
      ${PROJECT_SOURCE_DIR}/../../../SDK/include
  )

   target_link_directories(ConsoleDemo PUBLIC ${PROJECT_SOURCE_DIR}/../../../SDK/win/64)

    target_link_libraries(ConsoleDemo PUBLIC
        PXREARobotSDK.dll
    )
endif()

if(UNIX)
  if(ISA_NAME STREQUAL "aarch64")
    target_include_directories(ConsoleDemo PUBLIC
        ${PROJECT_SOURCE_DIR}/../../../SDK/include
    )
    target_link_directories(ConsoleDemo PUBLIC ${PROJECT_SOURCE_DIR}/../../../SDK/linux_aarch64/64)
  else()
    target_include_directories(ConsoleDemo PUBLIC
        ${PROJECT_SOURCE_DIR}/../../../SDK/include
    )

    target_link_directories(ConsoleDemo PUBLIC ${PROJECT_SOURCE_DIR}/../../../SDK/linux/64)
  endif()
    target_link_libraries(ConsoleDemo PUBLIC
        PXREARobotSDK
    )
endif()


include(GNUInstallDirs)
install(TARGETS ConsoleDemo
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

