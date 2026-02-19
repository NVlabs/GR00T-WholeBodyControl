#ifndef __UT_ROBOT_B2_MOTION_SWITCHER_ERROR_HPP__
#define __UT_ROBOT_B2_MOTION_SWITCHER_ERROR_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
    UT_DECL_ERR(UT_SWITCH_ERR_PARAMETR,         7001,  "parameter is invalid.")
    UT_DECL_ERR(UT_SWITCH_ERR_BUSY,             7002,  "switcher is busy.")
    UT_DECL_ERR(UT_SWITCH_ERR_EVENT,            7003,  "event is invalid.")
    UT_DECL_ERR(UT_SWITCH_ERR_NAME,             7004,  "name or alias is invalid.")
    UT_DECL_ERR(UT_SWITCH_ERR_CMD,              7005,  "name or alias is invalid.")
    UT_DECL_ERR(UT_SWITCH_ERR_EXEC_CHECK,       7006,  "check cmd execute error.")
    UT_DECL_ERR(UT_SWITCH_ERR_EXEC_SELECT,      7007,  "select cmd execute error.")
    UT_DECL_ERR(UT_SWITCH_ERR_EXEC_RELEASE,     7008,  "release cmd execute error.")
    UT_DECL_ERR(UT_SWITCH_ERR_CUSTOMIZE,        7009,  "save customize data error.")
}
}
}

#endif//__UT_ROBOT_B2_MOTION_SWITCHER_ERROR_HPP__
