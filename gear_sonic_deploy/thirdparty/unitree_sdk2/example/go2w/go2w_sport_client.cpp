#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <thread>
#include <pthread.h>
#include <stdexcept>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#define TOPIC_HIGHSTATE "rt/lf/sportmodestate"

using namespace std;

// 测试选项
struct TestOption
{
    std::string name; // 名称
    int id;           // id
};

const vector<TestOption> option_list =
    {{"damp", 0},        // 阻尼
     {"stand_up", 1},    // 站立锁定
     {"stand_down", 2},  // 趴下
     {"move", 3},        // 速度控制
     {"stop_move", 4},   // 停止运动
     {"speed_level", 5}, // 速度档位
     {"switch_gait", 6}, // 步态切换
     {"get_state", 7},   // 获取状态
     {"recovery", 8},    // 恢复站立
     {"balance", 9}};    // 平衡站立

int ConvertToInt(const std::string &str)
{
    try
    {
        std::stoi(str); // 尝试转换字符串为整数
        return std::stoi(str);
    }
    catch (const std::invalid_argument &)
    {
        return -1; // 字符串包含非数字字符
    }
    catch (const std::out_of_range &)
    {
        return -1; // 字符串表示的数字超出int范围
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

        // 如果输入的是 list 则输出所有的测试选项名称和id
        if (input.compare("list") == 0)
        {
            for (TestOption option : option_list)
            {
                std::cout << option.name << ", id: " << option.id << std::endl;
            }
        }

        // 如果输入的名名称或id在枚举的测试选项中，则记录测试选项名称和id
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

    // 待测试的功能的指针
    TestOption *test_option_;
};

void HighStateHandler(const void *message)
{
    unitree_go::msg::dds_::SportModeState_ hs = *(unitree_go::msg::dds_::SportModeState_ *)message;

    //  std::cout << "mode: " << (int)hs.mode() << ", "
    //            << "gait: " << (int)hs.gait_type() << ", "
    //            << "progress: " << hs.progress() << std::endl;
}

int main(int argc, char **argv)
{
    // 初始化 dds
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

    unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_> suber(TOPIC_HIGHSTATE);
    suber.InitChannel(HighStateHandler);

    // 待测试的选项
    TestOption test_option;
    test_option.id = 1;

    // 初始化sportclient
    unitree::robot::go2::SportClient sport_client;
    sport_client.SetTimeout(20.0f);
    sport_client.Init();

    // 初始化用户终端
    UserInterface user_interface;
    // 将user_interface.test_option_ 指向 test_option，用于传递终端输入的结果
    user_interface.test_option_ = &test_option;

    std::cout << "Input \"list \" to list all test option ..." << std::endl;
    long res_count = 0;
    while (1)
    {
        auto time_start_trick = std::chrono::high_resolution_clock::now();
        static const constexpr auto dt = std::chrono::microseconds(20000); // 50Hz

        // 等待终端输入，并解析
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
            // 此处为调用一次 Move 指令，机器人会移动1s后停下
            // 如果循环调用 Move 则可以持续移动，1s 中内无Move请求，机器人会自动停下。
            res = sport_client.Move(0.5, 0, 0);
        }
        else if (test_option.id == 4)
        {
            res = sport_client.StopMove();
        }
        else if (test_option.id == 5)
        {
            res = sport_client.SpeedLevel(1);
        }
        else if (test_option.id == 6)
        {
            // res = sport_client.SwitchGait(1);
        }
        else if (test_option.id == 7)
        {
            // std::map<std::string, std::string> state_map;
            // std::vector<std::string> state_name = {"speedLevel", "gait "};
            // res = sport_client.GetState(state_name, state_map);
            // std::cout << "Speed level: " << state_map["speedLevel"] << ", Gait: " << state_map["gait"] << std::endl;
        }
        else if (test_option.id == 8)
        {
            res = sport_client.RecoveryStand();
        }
        else if (test_option.id == 9)
        {
            res = sport_client.BalanceStand();
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