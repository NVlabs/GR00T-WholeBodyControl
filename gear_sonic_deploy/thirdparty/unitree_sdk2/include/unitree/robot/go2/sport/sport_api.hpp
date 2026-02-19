#ifndef __UT_ROBOT_GO2_SPORT_API_HPP__
#define __UT_ROBOT_GO2_SPORT_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*service name*/
const std::string ROBOT_SPORT_SERVICE_NAME = "sport";

/*api version*/
const std::string ROBOT_SPORT_API_VERSION = "1.0.0.1";

/*api id*/
const int32_t ROBOT_SPORT_API_ID_DAMP               = 1001;
const int32_t ROBOT_SPORT_API_ID_BALANCESTAND       = 1002;
const int32_t ROBOT_SPORT_API_ID_STOPMOVE           = 1003;
const int32_t ROBOT_SPORT_API_ID_STANDUP            = 1004;
const int32_t ROBOT_SPORT_API_ID_STANDDOWN          = 1005;
const int32_t ROBOT_SPORT_API_ID_RECOVERYSTAND      = 1006;
const int32_t ROBOT_SPORT_API_ID_EULER              = 1007;
const int32_t ROBOT_SPORT_API_ID_MOVE               = 1008;
const int32_t ROBOT_SPORT_API_ID_SIT                = 1009;
const int32_t ROBOT_SPORT_API_ID_RISESIT            = 1010;
const int32_t ROBOT_SPORT_API_ID_SPEEDLEVEL         = 1015;
const int32_t ROBOT_SPORT_API_ID_HELLO              = 1016;
const int32_t ROBOT_SPORT_API_ID_STRETCH            = 1017;
const int32_t ROBOT_SPORT_API_ID_CONTENT            = 1020;
const int32_t ROBOT_SPORT_API_ID_DANCE1             = 1022;
const int32_t ROBOT_SPORT_API_ID_DANCE2             = 1023;
const int32_t ROBOT_SPORT_API_ID_SWITCHJOYSTICK     = 1027;
const int32_t ROBOT_SPORT_API_ID_POSE               = 1028;
const int32_t ROBOT_SPORT_API_ID_SCRAPE             = 1029;
const int32_t ROBOT_SPORT_API_ID_FRONTFLIP          = 1030;
const int32_t ROBOT_SPORT_API_ID_FRONTJUMP          = 1031;
const int32_t ROBOT_SPORT_API_ID_FRONTPOUNCE        = 1032;
const int32_t ROBOT_SPORT_API_ID_HEART              = 1036;
const int32_t ROBOT_SPORT_API_ID_STATICWALK         = 1061;
const int32_t ROBOT_SPORT_API_ID_TROTRUN            = 1062;
const int32_t ROBOT_SPORT_API_ID_ECONOMICGAIT       = 1063;
const int32_t ROBOT_SPORT_API_ID_LEFTFLIP           = 2041;
const int32_t ROBOT_SPORT_API_ID_BACKFLIP           = 2043;
const int32_t ROBOT_SPORT_API_ID_HANDSTAND          = 2044;
const int32_t ROBOT_SPORT_API_ID_FREEWALK           = 2045;
const int32_t ROBOT_SPORT_API_ID_FREEBOUND          = 2046;
const int32_t ROBOT_SPORT_API_ID_FREEJUMP           = 2047;
const int32_t ROBOT_SPORT_API_ID_FREEAVOID          = 2048;
const int32_t ROBOT_SPORT_API_ID_CLASSICWALK        = 2049;
const int32_t ROBOT_SPORT_API_ID_WALKUPRIGHT        = 2050;
const int32_t ROBOT_SPORT_API_ID_CROSSSTEP          = 2051;
const int32_t ROBOT_SPORT_API_ID_AUTORECOVERY_SET   = 2054;
const int32_t ROBOT_SPORT_API_ID_AUTORECOVERY_GET   = 2055;
const int32_t ROBOT_SPORT_API_ID_SWITCHAVOIDMODE    = 2058;


}
}
}

#endif //__UT_ROBOT_GO2_SPORT_API_HPP__
