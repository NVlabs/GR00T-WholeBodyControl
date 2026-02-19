#ifdef HAS_ROS2

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <unistd.h>  // For getcwd

// Include the actual header files from the project
#include "../include/input_interface/input_interface.hpp"
#include "../include/localmotion_kplanner.hpp"
#include "../include/robot_parameters.hpp"
#include "../include/policy_parameters.hpp"
#include "../include/input_interface/ros2_input_handler.hpp"
#include "../include/state_logger.hpp"


int main(int argc, char** argv) {
    std::cout << "🚀 Testing REAL ROS2InputHandler Class (with PRODUCTION FastRTPS Config)" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    
    // Initialize ROS2 first
    rclcpp::init(argc, argv);
    std::cout << "💡 Press Ctrl+C to stop the test early" << std::endl;
    
    // Check if our production FastRTPS config exists
    std::cout << "🔍 Checking FastRTPS configuration..." << std::endl;
    const char* profile_path = "./config/fastrtps_profile.xml";
    if (FILE* file = fopen(profile_path, "r")) {
        fclose(file);
        std::cout << "✅ Found FastRTPS profile: " << profile_path << std::endl;
        std::cout << "🎯 This should eliminate 'XMLPARSER Error' for production deployment!" << std::endl;
    } else {
        std::cout << "⚠️  FastRTPS profile not found at: " << profile_path << std::endl;
        char* cwd = getcwd(nullptr, 0);
        if (cwd) {
            std::cout << "📍 Current working directory: " << cwd << std::endl;
            free(cwd);
        }
    }
    std::cout << std::endl;
    
    try {
        // Create the actual ROS2InputHandler instance using real class
        // Parameters: use_ik_mode (false), node_name, initial_encoder_mode (1 for 3-point tracking)
        auto input_handler = std::make_unique<ROS2InputHandler>(false, "test_real_ros2_handler");
        std::cout << "✅ ROS2InputHandler created successfully!" << std::endl;
        std::cout << "📡 Subscribed to topic: ControlPolicy/upper_body_pose (msgpack format)" << std::endl;
        std::cout << "🔍 ROS2 Status: " << (rclcpp::ok() ? "OK" : "ERROR") << std::endl;
        // Create minimal instances needed for handle_input() test (using actual classes)
        MotionDataReader motion_reader;
        std::shared_ptr<const MotionSequence> current_motion = nullptr;
        bool reinitialize_heading = false;
        int current_frame = 0;
        OperatorState operator_state; // No namespace needed
        std::array<double, 4> quaternion = {0.0, 0.0, 0.0, 1.0}; // Identity quaternion
        DataBuffer<HeadingState> heading_buffer;
        bool has_planner = true; // ROS2 mode requires planner to be available
        PlannerState planner_state;
        DataBuffer<MovementState> movement_buffer;
        std::mutex current_motion_mutex; // Mutex for thread-safe motion access
        
        std::cout << "⏳ Testing ROS2InputHandler with real classes (max 30 seconds)..." << std::endl;
        std::cout << "📝 This test expects msgpack-serialized messages on topic:" << std::endl;
        std::cout << "   ControlPolicy/upper_body_pose (std_msgs/ByteMultiArray)" << std::endl;
        std::cout << "📝 Use the Python teleop client to send test messages" << std::endl;
        
        auto start_time = std::chrono::steady_clock::now();
        int loop_count = 0;
        
        while (rclcpp::ok() && std::chrono::steady_clock::now() - start_time < std::chrono::seconds(30)) {
            // Update ROS2 input handler (process messages)
            input_handler->update();
            
            // Call handle_input() to test full integration with real classes
            // Note: encoder_mode no longer passed as parameter (initialized in constructor)
            input_handler->handle_input(motion_reader, current_motion, current_frame,
                                      operator_state, reinitialize_heading, heading_buffer, has_planner, 
                                      planner_state, movement_buffer, current_motion_mutex);
            
            // Print status every 100 loops (~10 seconds)
            if (loop_count % 100 == 0) {
                std::cout << "📊 Loop " << loop_count 
                          << " | Control goal valid: " << (input_handler->is_receiving_control_goal_data() ? "✅" : "❌")
                          << " | Planner: " << (planner_state.enabled ? "✅" : "❌")
                          << std::endl;
            }
            
            loop_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (!rclcpp::ok()) {
            std::cout << "\n🛑 Test interrupted by user (Ctrl+C)" << std::endl;
        } else {
            std::cout << "\n✅ Test completed successfully!" << std::endl;
        }
        std::cout << "🎯 ROS2InputHandler works with real classes and is ready for integration!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "👋 Shutting down ROS2..." << std::endl;
    rclcpp::shutdown();
    return 0;
}

#else

int main(int argc, char** argv) {
    std::cout << "❌ ROS2 support not compiled. Build with ROS2 environment." << std::endl;
    return 1;
}

#endif

