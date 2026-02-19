#include <unitree/robot/go2/sport/sport_client.hpp>

int main()
{
    /*
     * Initilaize ChannelFactory
     */
    unitree::robot::ChannelFactory::Instance()->Init(0);
    unitree::robot::go2::SportClient sc;

    sc.SetTimeout(5.0f);
    sc.Init();

    //Test Api
    while (true)
    {
        int32_t ret = sc.Move(0.5, 0.0, 0.0);
        std::cout << "Call Move ret:" << ret << std::endl;

        usleep(1000);
    }

    return 0;
}
