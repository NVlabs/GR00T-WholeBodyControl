#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <thread>
#include <pthread.h>
#include <stdexcept>
#include <unitree/robot/b2/sport/sport_client.hpp>

using namespace std;

struct TestOption
{
    std::string name; 
    int id;           
};

const vector<TestOption> option_list =
    {{"damp", 0},        
     {"stand_up", 1},    
     {"stand_down", 2},  
     {"move forward", 3},
     {"move lateral", 4},         
     {"move rotate", 5},        
     {"stop_move", 6},   
     {"switch_gait", 7}, 
     {"switch_gait", 8},
     {"recovery", 9},   
    }; 

int ConvertToInt(const std::string &str)
{
    try
    {
        std::stoi(str);
        return std::stoi(str);
    }
    catch (const std::invalid_argument &)
    {
        return -1;
    }
    catch (const std::out_of_range &)
    {
        return -1; 
    }
}

class UserInterface
{
public:
    UserInterface(){};
    ~UserInterface(){};

    void terminalHandle()
    {
        std::string input;
        std::getline(std::cin, input);

        if (input.compare("list") == 0)
        {
            for (TestOption option : option_list)
            {
                std::cout << option.name << ", id: " << option.id << std::endl;
            }
        }

        for (TestOption option : option_list)
        {
            if (input.compare(option.name) == 0 || ConvertToInt(input) == option.id)
            {
                test_option_->id = option.id;
                test_option_->name = option.name;
                std::cout << "Test: " << test_option_->name << ", test_id: " << test_option_->id << std::endl;
            }
        }
    };

    TestOption *test_option_;
};

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

    TestOption test_option;
    test_option.id = 1;

    unitree::robot::b2::SportClient sport_client;
    sport_client.SetTimeout(25.0f);
    sport_client.Init();

    UserInterface user_interface;
    user_interface.test_option_ = &test_option;

    std::cout << "Input \"list \" to list all test option ..." << std::endl;
    long res_count = 0;
    while (1)
    {
        auto time_start_trick = std::chrono::high_resolution_clock::now();
        static const constexpr auto dt = std::chrono::microseconds(20000); // 50Hz

        user_interface.terminalHandle();

        int res = 1;
        if (test_option.id == 0)
        {
            res = sport_client.Damp();
        }
        else if (test_option.id == 1)
        {
            res = sport_client.StandUp();
        }
        else if (test_option.id == 2)
        {
            res = sport_client.StandDown();
        }
        else if (test_option.id == 3)
        {
            res = sport_client.Move(0.3, 0, 0);// the robot will move for 0.5 seconds before stopping.
                                               // If the Move command is called in a loop, the robot will continue moving. 
                                               // If no Move request is received within 0.5 seconds, the robot will automatically stop.
        }
        else if (test_option.id == 4)
        {
            res = sport_client.Move(0, 0.3, 0);// the robot will move for 0.5 seconds before stopping.
                                               // If the Move command is called in a loop, the robot will continue moving. 
                                               // If no Move request is received within 0.5 seconds, the robot will automatically stop.
        }
        else if (test_option.id == 5)
        {
            res = sport_client.Move(0, 0, 0.5);// the robot will move for 0.5 seconds before stopping.
                                               // If the Move command is called in a loop, the robot will continue moving. 
                                               // If no Move request is received within 0.5 seconds, the robot will automatically stop.
        }
        else if (test_option.id == 6)
        {
            res = sport_client.StopMove();
        }
        else if (test_option.id == 7)
        {
            res = sport_client.SwitchGait(0);
        }
        else if (test_option.id == 8)
        {
            res = sport_client.SwitchGait(1);
        }
        else if (test_option.id == 9)
        {
            res = sport_client.RecoveryStand();
        }

        if (res < 0)
        {
            res_count += 1;
            std::cout << "Request error for: " << option_list[test_option.id].name << ", code: " << res << ", count: " << res_count << std::endl;
        }
        else
        {
            res_count = 0;
            std::cout << "Request successed: " << option_list[test_option.id].name << ", code: " << res << std::endl;
        }
        std::this_thread::sleep_until(time_start_trick + dt);
    }
    return 0;
}