#ifndef __UT_ROBOT_GO2_ROBOT_STATE_API_HPP__
#define __UT_ROBOT_GO2_ROBOT_STATE_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*
 * service name
 */
const std::string ROBOT_STATE_SERVICE_NAME = "robot_state";

/*
 * api version
 */
const std::string ROBOT_STATE_API_VERSION = "1.0.0.1";

/*
 * api id
 */
const int32_t ROBOT_STATE_API_ID_SERVICE_SWITCH = 1001;
const int32_t ROBOT_STATE_API_ID_SET_REPORT_FREQ = 1002;
const int32_t ROBOT_STATE_API_ID_SERVICE_LIST = 1003;

/*
 * request parameter for 1001
 */
class ServiceSwitchParameter : public common::Jsonize
{
public:
    ServiceSwitchParameter() : swit(0)
    {}

    ~ServiceSwitchParameter()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);
        common::FromJson(json["switch"], swit);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);
        common::ToJson(swit, json["switch"]);
    }

public:
    std::string name;
    int32_t swit;
};

/*
 * response data for 1001
 */
class ServiceSwitchData : public common::Jsonize
{
public:
    ServiceSwitchData() : status(0)
    {}

    ~ServiceSwitchData()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);
        common::FromJson(json["status"], status);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);
        common::ToJson(status, json["status"]);
    }

public:
    std::string name;
    int32_t status;
};

/*
 * request parameter for 1002
 */
class SetReportFreqParameter : public common::Jsonize
{
public:
    SetReportFreqParameter()
    {}

    ~SetReportFreqParameter()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["interval"], interval);
        common::FromJson(json["duration"], duration);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(interval, json["interval"]);
        common::ToJson(duration, json["duration"]);
    }

public:
    int32_t interval;
    int32_t duration;
};

/*
 * request data for 1003
 */
class ServiceStateData : public common::Jsonize
{
public:
    ServiceStateData()
    {}

    ~ServiceStateData()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);
        common::FromJson(json["status"], status);
        common::FromJson(json["protect"], protect);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);
        common::ToJson(status, json["status"]);
        common::ToJson(protect, json["protect"]);
    }

public:
    std::string name;
    int32_t status;
    int32_t protect;
};

}
}
}

#endif//__UT_ROBOT_GO2_ROBOT_STATE_API_HPP__
