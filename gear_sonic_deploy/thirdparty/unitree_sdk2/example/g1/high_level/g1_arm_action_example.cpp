/**
 * @file g1_arm_action_example.cpp
 * @brief This example demonstrates how to use the G1 Arm Action Client to execute predefined arm actions.
 */
#include "unitree/robot/g1/arm/g1_arm_action_error.hpp"
#include "unitree/robot/g1/arm/g1_arm_action_client.hpp"

using namespace unitree::robot::g1;

int main(int argc, const char** argv) 
{
    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1 Arm Action Example      \n\n";

    // Unitree DDS Initialization; arg[1] is the network interface
    unitree::robot::ChannelFactory::Instance()->Init(0, argc > 1 ? argv[1] : "");

    auto client = std::make_shared<unitree::robot::g1::G1ArmActionClient>();
    client->Init();
    client->SetTimeout(10.f); // All actions will last less than 10 seconds.

    std::cout << "Usage: \n";
    std::cout << "  - 0: print supported actions.\n";
    std::cout << "  - an id: execute an action.\n";
    std::cout << "Attention: \n";
    std::cout << "  Some actions will not be displayed on the APP, \n"; 
    std::cout << "  but can be executed by the program.\n";
    std::cout << "  These actions may cause the robot to fall,\n";
    std::cout << "  so please execute them with caution.\n";

    int32_t action_id = 0;
    std::string line;
    while (true) {
        std::cout << "\nEnter action ID: .\n";
        std::getline(std::cin, line);
        try {
            action_id = std::stoi(line);
        } catch (const std::exception&) {
            std::cout << "Invalid input. Please enter an integer.\n";
            continue;
        }

        if (action_id == 0) {
            std::string action_list_data;
            int32_t ret = client->GetActionList(action_list_data);
            if (ret != 0) {
                std::cerr << "Failed to get action list, error code: " << ret << "\n";
                continue;
            }
            std::cout << "Available actions:\n" << action_list_data << std::endl;
        } else {
            int32_t ret = client->ExecuteAction(action_id);
            if(ret != 0) {
                switch (ret)
                {
                case UT_ROBOT_ARM_ACTION_ERR_ARMSDK:
                    std::cout << UT_ROBOT_ARM_ACTION_ERR_ARMSDK_DESC << std::endl;
                    break;
                case UT_ROBOT_ARM_ACTION_ERR_HOLDING:
                    std::cout << UT_ROBOT_ARM_ACTION_ERR_HOLDING_DESC << std::endl;
                    break;
                case UT_ROBOT_ARM_ACTION_ERR_INVALID_ACTION_ID:
                    std::cout << UT_ROBOT_ARM_ACTION_ERR_INVALID_ACTION_ID_DESC << std::endl;
                    break;
                case UT_ROBOT_ARM_ACTION_ERR_INVALID_FSM_ID:
                    std::cout << "The actions are only supported in fsm id {500, 501, 801}" << std::endl;
                    std::cout << "You can subscribe the topic rt/sportmodestate to check the fsm id." << std::endl;
                    std::cout << "And in the state 801, the actions are only supported in the fsm mode {0, 3}." << std::endl;
                    std::cout << "If an error is still returned at this point, ignore this action.";
                    break;
                default:
                    std::cerr << "Execute action failed, error code: " << ret << std::endl;
                    break;
                }
            }
        }
    }

    return 0;
};
