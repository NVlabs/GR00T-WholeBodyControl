cmake_minimum_required(VERSION 3.16)

project(RobotDataRecorder VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Quick)
qt_standard_project_setup()


if(WIN32)

    qt_add_executable(RobotDataRecorder
        main.cpp
    )

    set_target_properties(RobotDataRecorder PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../../../Redistributable/win"
        RUNTIME_OUTPUT_NAME "RobotDataRecorder"
    )

    qt_add_qml_module(RobotDataRecorder
        URI RobotDataRecorder
        VERSION 1.0
        QML_FILES main.qml
        SOURCES SDKCaller.cpp SDKCaller.h
        SOURCES SensorDataHandler.cpp SensorDataHandler.h SensorDataSaver.cpp SensorDataSaver.h
    )

    target_link_libraries(RobotDataRecorder
        PRIVATE Qt6::Quick
    )

    set_target_properties(RobotDataRecorder PROPERTIES
    #    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.RobotDataRecorder
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
        MACOSX_BUNDLE TRUE
        WIN32_EXECUTABLE TRUE
    )


    target_include_directories(RobotDataRecorder PUBLIC
        ${PROJECT_SOURCE_DIR}/../../../SDK/include
    )

    target_link_directories(RobotDataRecorder PUBLIC ${PROJECT_SOURCE_DIR}/../../../SDK/win/64)
    target_link_libraries(RobotDataRecorder PUBLIC
        PXREARobotSDK.dll
    )

    include(GNUInstallDirs)
    install(TARGETS RobotDataRecorder
        BUNDLE DESTINATION .
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()


if(UNIX)
    set(CMAKE_CXX_STANDARD 17)

    # Get system architecture
    cmake_host_system_information(RESULT ISA_NAME QUERY OS_PLATFORM)
    message("OS_PLATFORM: ${ISA_NAME}")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--copy-dt-needed-entries")
    qt_add_executable(RobotDataRecorder
        main.cpp
    )

    # Set output directory based on architecture
    if(ISA_NAME STREQUAL "aarch64")
        set_target_properties(RobotDataRecorder PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../../../Redistributable/linux_aarch64"
            RUNTIME_OUTPUT_NAME "RobotDataRecorder"
        )
    else()
        set_target_properties(RobotDataRecorder PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/../../../Redistributable/linux"
            RUNTIME_OUTPUT_NAME "RobotDataRecorder"
        )
    endif()

    qt_add_qml_module(RobotDataRecorder
        URI RobotDataRecorder
        VERSION 1.0
        QML_FILES main.qml
        SOURCES SDKCaller.cpp SDKCaller.h
        SOURCES SensorDataHandler.cpp SensorDataHandler.h SensorDataSaver.cpp SensorDataSaver.h
    )

    target_link_libraries(RobotDataRecorder
        PRIVATE Qt6::Quick
    )

    # Set include and link directories based on architecture
    target_include_directories(RobotDataRecorder PUBLIC
        ${PROJECT_SOURCE_DIR}/../../../SDK/include
    )

    if(ISA_NAME STREQUAL "aarch64")
        target_link_directories(RobotDataRecorder PUBLIC ${PROJECT_SOURCE_DIR}/../../../SDK/linux_aarch64/64)
    else()
        target_link_directories(RobotDataRecorder PUBLIC ${PROJECT_SOURCE_DIR}/../../../SDK/linux/64)
    endif()

    target_link_libraries(RobotDataRecorder PUBLIC
        PXREARobotSDK
    )
endif()
