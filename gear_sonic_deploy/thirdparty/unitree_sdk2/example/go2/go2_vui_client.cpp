#include <unitree/robot/go2/vui/vui_client.hpp>

int main(int32_t argc, const char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: vui_client_example network_interface_name" << std::endl;
        exit(0);
    }
    /*
     * Initilaize ChannelFactory
     */
    std::string networkInterface = argv[1];
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);
    unitree::robot::go2::VuiClient vc;

    /*
     * Set request timeout 1.0s
     */
    vc.SetTimeout(1.0f);
    vc.Init();

    //Test Api

    int level = 0, value = 0;
    int ret;

    while (true)
    {
        ret = vc.SetBrightness(level);
        std::cout << "SetBrightness  level=" << level << ", api return:" << ret << std::endl;
        ++level %= 11;
        sleep(1);
        ret = vc.GetBrightness(value);
        std::cout << "GetBrightness value=" << value << ", api return:" << ret << std::endl;
        sleep(1);
    }

    return 0;
}
