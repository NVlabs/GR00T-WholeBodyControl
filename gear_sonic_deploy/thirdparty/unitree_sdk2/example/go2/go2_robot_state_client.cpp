#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
#include <unitree/common/time/time_tool.hpp>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::go2;

int main(int32_t argc, const char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: robot_state_client_example [NetWorkInterface(eth0)] [ServiceName(sport_mode)]" << std::endl;
        exit(0);
    }

    std::string networkInterface = "eth0", serviceName = "sport_mode";

    if (argc > 1)
    {
        networkInterface = argv[1];
    }

    if (argc > 2)
    {
        serviceName = argv[2];
    }

    std::cout << "NetWorkInterface:" << networkInterface << std::endl;
    std::cout << "Switch ServiceName:" << serviceName << std::endl; 

    ChannelFactory::Instance()->Init(0, networkInterface);

    RobotStateClient rsc;
    rsc.SetTimeout(10.0f);
    rsc.Init();

    std::string clientApiVersion = rsc.GetApiVersion();
    std::string serverApiVersion = rsc.GetServerApiVersion();

    if (clientApiVersion != serverApiVersion)
    {
        std::cout << "client and server api versions are not equal." << std::endl;
    }

    Timer timer;

    int32_t status;
    int32_t ret = rsc.SetReportFreq(3, 30);
    std::cout << "Call SetReportFreq[3,30] ret:" << ret << ", cost:" << timer.Stop() << " (us)" << std::endl;

    sleep(5);
    timer.Restart();

    ret = rsc.ServiceSwitch(serviceName, 0, status);
    std::cout << "Call ServiceSwitch[" << serviceName << ",0] ret:" << ret << ", cost:" << timer.Stop() << " (us)" << std::endl;

    sleep(5);
    timer.Restart();

    ret = rsc.ServiceSwitch(serviceName, 1, status);
    std::cout << "Call ServiceSwitch[" << serviceName << ",1] ret:" << ret << ", cost:" << timer.Stop() << " (us)" << std::endl;

    sleep(5);
    timer.Restart();

    std::vector<ServiceState> serviceStateList;
    ret = rsc.ServiceList(serviceStateList);
    std::cout << "Call ServiceList ret:" << ret << ", cost:" << timer.Stop() << " (us)" << std::endl;

    size_t i, count=serviceStateList.size();
    std::cout << "serviceStateList size:" << count << std::endl;

    for (i=0; i<count; i++)
    {
        const ServiceState& serviceState = serviceStateList[i];
        std::cout << "name:" << serviceState.name << ", status:" << serviceState.status << ", protect:" << serviceState.protect << std::endl;
    }

    ChannelFactory::Instance()->Release();

    return 0;
}
