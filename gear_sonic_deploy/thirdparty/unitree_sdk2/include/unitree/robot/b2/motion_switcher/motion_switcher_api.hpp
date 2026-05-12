#ifndef __UT_ROBOT_B2_MOTION_SWITCHER_API_HPP__
#define __UT_ROBOT_B2_MOTION_SWITCHER_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
/*
 * service name
 */
const std::string MOTION_SWITCHER_SERVICE_NAME = "motion_switcher";

/*
 * api version
 */
const std::string MOTION_SWITCHER_API_VERSION = "1.0.0.1";

/*
 * api id
 */
const int32_t MOTION_SWITCHER_API_ID_CHECK_MODE = 1001;
const int32_t MOTION_SWITCHER_API_ID_SELECT_MODE = 1002;
const int32_t MOTION_SWITCHER_API_ID_RELEASE_MODE = 1003;
const int32_t MOTION_SWITCHER_API_ID_SET_SILENT = 1004;
const int32_t MOTION_SWITCHER_API_ID_GET_SILENT = 1005;

/*
 * api data type
 */
class JsonizeModeName : public common::Jsonize
{
public:
    JsonizeModeName()
    {}

    ~JsonizeModeName()
    {}

    /*override*/
    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);

        if (json.find("form") != json.end())
        {
            common::FromJson(json["form"], form);
        }
    }

    /*override*/
    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);

        if (!form.empty())
        {
            common::ToJson(form, json["form"]);
        }
    }

public:
    std::string name;
    std::string form;
};

class JsonizeSilent : public common::Jsonize
{
public:
    JsonizeSilent() : silent(false)
    {}

    ~JsonizeSilent()
    {}

    /*override*/
    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["silent"], silent);
    }

    /*override*/
    void toJson(common::JsonMap& json) const
    {
        common::ToJson(silent, json["silent"]);
    }

public:
    bool silent;
};

}
}
}

#endif//__UT_ROBOT_B2_MOTION_SWITCHER_API_HPP__
