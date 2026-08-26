#include <mutex>

// #include <go2_idl/WirelessController_.hpp>
#include "unitree/idl/go2/WirelessController_.hpp"
#include "unitree/common/thread/thread.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"

#include "advanced_gamepad.hpp"

#define TOPIC_JOYSTICK "rt/wirelesscontroller"

using namespace unitree::common;
using namespace unitree::robot;

class GamepadExample
{
public:
    GamepadExample() {}

    // setup dds model
    void InitDdsModel(const std::string &networkInterface = "")
    {
        ChannelFactory::Instance()->Init(0, networkInterface);
        joystick_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));

        joystick_subscriber->InitChannel(std::bind(&GamepadExample::MessageHandler, this, std::placeholders::_1), 1);
    }

    // set gamepad dead_zone parameter
    void SetGamepadDeadZone(float deadzone)
    {
        gamepad.dead_zone = deadzone;
    }

    // set gamepad smooth parameter
    void setGamepadSmooth(float smooth)
    {
        gamepad.smooth = smooth;
    }

    // callback function to save joystick message
    void MessageHandler(const void *message)
    {
        std::lock_guard<std::mutex> lock(joystick_mutex);
        joystick_msg = *(unitree_go::msg::dds_::WirelessController_ *)message;
    }

    // work thread
    void Step()
    {
        {
            std::lock_guard<std::mutex> lock(joystick_mutex);
            gamepad.Update(joystick_msg);
        }

        // some operations
        if (gamepad.A.on_press)
        {
            press_count += 1;
        }

        // print gamepad state
        std::cout << "lx: " << gamepad.lx << std::endl
                  << "A: pressed: " << gamepad.A.pressed
                  << "; on_press: " << gamepad.A.on_press
                  << "; on_release: " << gamepad.A.on_release
                  << std::endl << "press count: " << press_count
                  << std::endl << "===========================" << std::endl;
    }

    // start the work thread
    void Start()
    {
        control_thread_ptr = CreateRecurrentThreadEx("nn_ctrl", UT_CPU_ID_NONE, 40000, &GamepadExample::Step, this);
    }

protected:
    ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;
    unitree_go::msg::dds_::WirelessController_ joystick_msg;

    Gamepad gamepad;

    ThreadPtr control_thread_ptr;

    std::mutex joystick_mutex;

    int press_count = 0;
};

int main()
{
    // create example object
    GamepadExample example;

    // set gamepad params
    example.setGamepadSmooth(0.2);
    example.SetGamepadDeadZone(0.5);

    // start program
    example.InitDdsModel();
    example.Start();

    while (1)
    {
        usleep(20000);
    }
    return 0;
}