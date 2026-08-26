add_executable(test_publisher 
    publisher.cpp 
    HelloWorldData.cpp
)
target_link_libraries(test_publisher unitree_sdk2)

add_executable(test_subscriber 
    subscriber.cpp 
    HelloWorldData.cpp
)
target_link_libraries(test_subscriber unitree_sdk2)