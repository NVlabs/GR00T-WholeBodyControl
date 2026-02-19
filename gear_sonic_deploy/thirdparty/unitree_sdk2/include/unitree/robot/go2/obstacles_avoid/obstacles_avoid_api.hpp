#ifndef __UT_ROBOT_GO2_OBSTACLES_AVOID_API_HPP__
#define __UT_ROBOT_GO2_OBSTACLES_AVOID_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
const std::string ROBOT_OBSTACLES_AVOID_SERVICE_NAME = "obstacles_avoid";
const std::string ROBOT_OBSTACLES_AVOID_API_VERSION = "1.0.0.2";

const int32_t ROBOT_API_ID_OBSTACLES_AVOID_SWITCH_SET = 1001;
const int32_t ROBOT_API_ID_OBSTACLES_AVOID_SWITCH_GET = 1002;
const int32_t ROBOT_API_ID_OBSTACLES_AVOID_MOVE = 1003;
const int32_t ROBOT_API_ID_OBSTACLES_AVOID_USE_REMOTE_COMMAND_FROM_API = 1004;

class ObstaclesAvoidSwitchSetParameter : public common::Jsonize
{
public:
    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["enable"], mEnable);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(mEnable, json["enable"]);
    }

    bool mEnable = true;
};

class ObstaclesAvoidSwitchGetData : public common::Jsonize
{
public:
    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["enable"], mEnable);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(mEnable, json["enable"]);
    }

    bool mEnable = true;
};

class ObstaclesAvoidMoveParameter : public common::Jsonize
{
public:
    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["x"], mX);
        common::FromJson(json["y"], mY);
        common::FromJson(json["yaw"], mYaw);
        common::FromJson(json["mode"], mMode);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(mX, json["x"]);
        common::ToJson(mY, json["y"]);
        common::ToJson(mYaw, json["yaw"]);
        common::ToJson(mMode, json["mode"]);
    }

    float mX = 0.0;
    float mY = 0.0;
    float mYaw = 0.0;

    /*
     * mode:
     * 0 for vel
     * 1 for increment pose
     * 2 for absolute pose
     */
    int mMode = 0;
};

class ObstaclesAvoidRemoteCommandSource : public common::Jsonize
{
public:
    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["is_remote_commands_from_api"], mIsRemoteCommandsFromApi);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(mIsRemoteCommandsFromApi, json["is_remote_commands_from_api"]);
    }

    bool mIsRemoteCommandsFromApi = true;
};

}
}
}

#endif//__UT_ROBOT_GO2_OBSTACLES_AVOID_API_HPP__
