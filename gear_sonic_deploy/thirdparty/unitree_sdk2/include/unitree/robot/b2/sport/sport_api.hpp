#ifndef __UT_ROBOT_B2_SPORT_API_HPP__
#define __UT_ROBOT_B2_SPORT_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
/*service name*/
const std::string ROBOT_SPORT_SERVICE_NAME = "sport";

/*api version*/
const std::string ROBOT_SPORT_API_VERSION = "1.0.0.1";

/*api id*/
const int32_t ROBOT_SPORT_API_ID_DAMP              = 1001;
const int32_t ROBOT_SPORT_API_ID_BALANCESTAND      = 1002;
const int32_t ROBOT_SPORT_API_ID_STOPMOVE          = 1003;
const int32_t ROBOT_SPORT_API_ID_STANDUP           = 1004;
const int32_t ROBOT_SPORT_API_ID_STANDDOWN         = 1005;
const int32_t ROBOT_SPORT_API_ID_RECOVERYSTAND     = 1006;
const int32_t ROBOT_SPORT_API_ID_MOVE              = 1008;
const int32_t ROBOT_SPORT_API_ID_SWITCHGAIT        = 1011;
const int32_t ROBOT_SPORT_API_ID_BODYHEIGHT        = 1013;
const int32_t ROBOT_SPORT_API_ID_SPEEDLEVEL        = 1015;
const int32_t ROBOT_SPORT_API_ID_TRAJECTORYFOLLOW  = 1018;
const int32_t ROBOT_SPORT_API_ID_CONTINUOUSGAIT    = 1019;
const int32_t ROBOT_SPORT_API_ID_MOVETOPOS            = 1036;
const int32_t ROBOT_SPORT_API_ID_SWITCHMOVEMODE       = 1038;
const int32_t ROBOT_SPORT_API_ID_VISIONWALK        = 1101;
const int32_t ROBOT_SPORT_API_ID_HANDSTAND       = 1039;
const int32_t ROBOT_SPORT_API_ID_AUTORECOVERY_SET = 1040;
const int32_t ROBOT_SPORT_API_ID_FREEWALK  = 1045;
const int32_t ROBOT_SPORT_API_ID_CLASSICWALK = 1049;
const int32_t ROBOT_SPORT_API_ID_FASTWALK  = 1050;
const int32_t ROBOT_SPORT_API_ID_FREEEULER = 1051;

}
}
}

#endif //__UT_ROBOT_B2_SPORT_API_HPP__
