add_executable(g1_loco_client high_level/g1_loco_client_example.cpp)
target_link_libraries(g1_loco_client unitree_sdk2)

add_executable(g1_arm5_sdk_dds_example high_level/g1_arm5_sdk_dds_example.cpp)
target_link_libraries(g1_arm5_sdk_dds_example unitree_sdk2)

add_executable(g1_arm7_sdk_dds_example high_level/g1_arm7_sdk_dds_example.cpp)
target_link_libraries(g1_arm7_sdk_dds_example unitree_sdk2)

add_executable(g1_arm_action_example high_level/g1_arm_action_example.cpp)
target_link_libraries(g1_arm_action_example unitree_sdk2)

add_executable(g1_ankle_swing_example low_level/g1_ankle_swing_example.cpp)
target_link_libraries(g1_ankle_swing_example unitree_sdk2)

add_executable(g1_audio_client_example audio/g1_audio_client_example.cpp)
target_link_libraries(g1_audio_client_example unitree_sdk2)

add_executable(g1_dex3_example dex3/g1_dex3_example.cpp)
target_link_libraries(g1_dex3_example unitree_sdk2)

find_package(Boost COMPONENTS program_options)
if(Boost_FOUND)
    add_executable(g1_termination low_level/terminations.cpp)
    target_link_libraries(g1_termination unitree_sdk2 Boost::program_options)
endif()

find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    if (${yaml-cpp_VERSION} VERSION_GREATER_EQUAL "0.6")
        message(STATUS "Found yaml-cpp version ${yaml-cpp_VERSION}")
        add_executable(g1_dual_arm_example low_level/g1_dual_arm_example.cpp)
        target_link_libraries(g1_dual_arm_example PRIVATE unitree_sdk2 yaml-cpp)
        target_compile_definitions(g1_dual_arm_example PUBLIC BLIB_DIR="${CMAKE_CURRENT_SOURCE_DIR}/low_level/behavior_lib/")
    else()
        message(STATUS "yaml-cpp version ${yaml-cpp_VERSION} is too old, skipping build of g1_dual_arm_example.")
    endif()
endif()
