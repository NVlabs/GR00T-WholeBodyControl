#ifndef __UT_ROBOT_B2_MOTION_SWITCHER_CLIENT_HPP__
#define __UT_ROBOT_B2_MOTION_SWITCHER_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
/*
 * @brief MotionSwitcherClient
 */
class MotionSwitcherClient : public Client
{
public:
    explicit MotionSwitcherClient();
    ~MotionSwitcherClient();

    void Init();

    int32_t CheckMode(std::string& form, std::string& name);
    int32_t SelectMode(const std::string& nameOrAlias);
    int32_t ReleaseMode();
    int32_t SetSilent(bool silent);
    int32_t GetSilent(bool& silent);
};

}
}
}

#endif//__UT_ROBOT_B2_MOTION_SWITCHER_CLIENT_HPP__

