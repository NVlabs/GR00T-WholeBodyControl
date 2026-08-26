#ifndef __UT_ROBOT_B2_ROBOT_STATE_CLIENT_ERROR_HPP__
#define __UT_ROBOT_B2_ROBOT_STATE_CLIENT_ERROR_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
UT_DECL_ERR(UT_ROBOT_STATE_CLIENT_ERR_SERVICE_SWITCH,      5201,   "service switch error.")
UT_DECL_ERR(UT_ROBOT_STATE_CLIENT_ERR_SERVICE_PROTECTED,   5202,   "service is protected.")
UT_DECL_ERR(UT_ROBOT_STATE_CLIENT_ERR_LOWPOWER_SWITCH,     5203,   "low power switch error.")
UT_DECL_ERR(UT_ROBOT_STATE_CLIENT_ERR_LOWPOWER_STATE,      5204,   "low power state error.")
}
}
}

#endif//__UT_ROBOT_B2_ROBOT_STATE_CLIENT_ERROR_HPP__
