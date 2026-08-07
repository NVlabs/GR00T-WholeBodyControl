#ifndef __UT_ROBOT_GO2_UTRACK_API_HPP__
#define __UT_ROBOT_GO2_UTRACK_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
const std::string ROBOT_UTRACK_SERVICE_NAME = "uwbswitch";
const std::string ROBOT_UTRACK_API_VERSION = "1.0.0.1";

const int32_t ROBOT_API_ID_UTRACK_SWITCH_SET = 1001;
const int32_t ROBOT_API_ID_UTRACK_SWITCH_GET = 1002;
const int32_t ROBOT_API_ID_UTRACK_IS_TRACKING = 1003;

class UtrackSwitchSetParameter : public common::Jsonize
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

    int32_t mEnable = true;
};

class UtrackSwitchGetData : public common::Jsonize
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

    int32_t mEnable = true;
};

}
}
}

#endif//__UT_ROBOT_GO2_UTRACK_API_HPP__
