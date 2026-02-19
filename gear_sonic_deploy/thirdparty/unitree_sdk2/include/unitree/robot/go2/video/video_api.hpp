#ifndef __UT_ROBOT_GO2_VIDEO_API_HPP__
#define __UT_ROBOT_GO2_VIDEO_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*service name*/
const std::string ROBOT_VIDEO_SERVICE_NAME = "videohub";

/*api version*/
const std::string ROBOT_VIDEO_API_VERSION = "1.0.0.0";

/*api id*/
const int32_t ROBOT_VIDEO_API_ID_GETIMAGESAMPLE       = 1001;
}
}
}

#endif //__UT_ROBOT_GO2_VIDEO_API_HPP__
