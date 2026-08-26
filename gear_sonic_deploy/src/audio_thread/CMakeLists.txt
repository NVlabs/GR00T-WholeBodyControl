add_library(audio_thread audio_thread.cpp)
target_include_directories(audio_thread PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(audio_thread PRIVATE unitree_sdk2)
