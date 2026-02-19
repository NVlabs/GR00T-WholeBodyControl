#ifndef __UT_ROBOT_GO2_ROBOT_STATE_CLIENT_HPP__
#define __UT_ROBOT_GO2_ROBOT_STATE_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*
 * @brief ServiceState
 */
class ServiceState
{
public:
    ServiceState() : status(0), protect(0)
    {}

    std::string name;
    int32_t status;
    int32_t protect;
};

/*
 * @brief RobotStateClient
 */
class RobotStateClient : public Client
{
public:
    explicit RobotStateClient();
    ~RobotStateClient();

    void Init();

    int32_t ServiceList(std::vector<ServiceState>& serviceStateList);
    int32_t ServiceSwitch(const std::string& name, int32_t swit, int32_t& status);
    int32_t SetReportFreq(int32_t interval, int32_t duration);
};

}
}
}

#endif//__UT_ROBOT_GO2_ROBOT_STATE_CLIENT_HPP__

