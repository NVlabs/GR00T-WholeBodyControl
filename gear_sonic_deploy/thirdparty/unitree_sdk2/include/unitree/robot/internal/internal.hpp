#ifndef __UT_ROBOT_SDK_INERNAL_HPP__
#define __UT_ROBOT_SDK_INERNAL_HPP__

#include <unitree/robot/internal/internal_api.hpp>
#include <unitree/robot/internal/internal_error.hpp>
#include <unitree/robot/internal/internal_request_response.hpp>

namespace unitree
{
namespace robot
{
using RequestPtr = std::shared_ptr<Request>;
using ResponsePtr = std::shared_ptr<Response>;
}
}

#endif//__UT_ROBOT_SDK_INERNAL_HPP__
