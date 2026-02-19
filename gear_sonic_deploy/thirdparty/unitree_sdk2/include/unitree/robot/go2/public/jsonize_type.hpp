#ifndef __UT_ROBOT_GO2_SDK_JSON_DATA_TYPE_HPP__
#define __UT_ROBOT_GO2_SDK_JSON_DATA_TYPE_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*
 * @brief Jonsize for simple flag type bool
 */
class JsonizeFlagBool : public common::Jsonize
{
public:
    JsonizeFlagBool() : flag(false)
    {}

    ~JsonizeFlagBool()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["flag"], flag);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(flag, json["flag"]);
    }

public:
    bool flag;
};

/*
 * @brief Jonsize for simple data type bool
 */
class JsonizeDataBool : public common::Jsonize
{
public:
    JsonizeDataBool() : data(false)
    {}

    ~JsonizeDataBool()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["data"], data);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(data, json["data"]);
    }

public:
    bool data;
};

/*
 * @brief Jonsize for simple data type int
 */
class JsonizeDataInt : public common::Jsonize
{
public:
    JsonizeDataInt() : data(0)
    {}

    ~JsonizeDataInt()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["data"], data);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(data, json["data"]);
    }

public:
    int data;
};

/*
 * @brief Jonsize for simple data type float
 */
class JsonizeDataFloat : public common::Jsonize
{
public:
    JsonizeDataFloat() : data(0.0F)
    {}

    ~JsonizeDataFloat()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["data"], data);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(data, json["data"]);
    }

public:
    float data;
};

/*
 * @brief Jonsize for simple data type double
 */
class JsonizeDataDouble : public common::Jsonize
{
public:
    JsonizeDataDouble() : data(0.0F)
    {}

    ~JsonizeDataDouble()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["data"], data);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(data, json["data"]);
    }

public:
    double data;
};

/*
 * @brief Jonsize for simple data type string
 */
class JsonizeDataString : public common::Jsonize
{
public:
    JsonizeDataString()
    {}

    ~JsonizeDataString()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["data"], data);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(data, json["data"]);
    }

public:
    std::string data;
};

/*
 * @brief Jonsize for data type vec3
 */
class JsonizeVec3 : public common::Jsonize
{
public:
    JsonizeVec3() : x(0.0F), y(0.0F), z(0.0F)
    {}

    ~JsonizeVec3()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["x"], x);
        common::FromJson(json["y"], y);
        common::FromJson(json["z"], z);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(x, json["x"]);
        common::ToJson(y, json["y"]);
        common::ToJson(z, json["z"]);
    }

public:
    float x;
    float y;
    float z;
};

/*
 * @brief Jonsize for data type quaternion
 */
class JsonizeQuat : public common::Jsonize
{
public:
    JsonizeQuat() : x(0.0F), y(0.0F), z(0.0F), w(0.0F)
    {}

    ~JsonizeQuat()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["x"], x);
        common::FromJson(json["y"], y);
        common::FromJson(json["z"], z);
        common::FromJson(json["w"], w);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(x, json["x"]);
        common::ToJson(y, json["y"]);
        common::ToJson(z, json["z"]);
        common::ToJson(w, json["w"]);
    }

public:
    float x;
    float y;
    float z;
    float w;
};

/*
 * Jsonize class for PathPoint
 */
class JsonizePathPoint : public common::Jsonize
{
public:
    JsonizePathPoint() :
        timeFromStart(0.0F), x(0.0F), y(0.0F), yaw(0.0F), vx(0.0F), vy(0.0F), vyaw(0.0F)
    {}

    ~JsonizePathPoint()
    {}

public:
    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["t_from_start"], timeFromStart);
        common::FromJson(json["x"], x);
        common::FromJson(json["y"], y);
        common::FromJson(json["yaw"], yaw);
        common::FromJson(json["vx"], vx);
        common::FromJson(json["vy"], vy);
        common::FromJson(json["vyaw"], vyaw);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(timeFromStart, json["t_from_start"]);
        common::ToJson(x, json["x"]);
        common::ToJson(y, json["y"]);
        common::ToJson(yaw, json["yaw"]);
        common::ToJson(vx, json["vx"]);
        common::ToJson(vy, json["vy"]);
        common::ToJson(vyaw, json["vyaw"]);
  }

public:
    float timeFromStart;
    float x;
    float y;
    float yaw;
    float vx;
    float vy;
    float vyaw;
};

/*
 * @brief Jonsize for simple common object type int
 */
class JsonizeCommObjInt : public common::Jsonize
{
public:
    JsonizeCommObjInt() : value(0)
    {}

    ~JsonizeCommObjInt()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json[name], value);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(value, json[name]);
    }

public:
    int value;
    std::string name;
};

}
}
}
#endif//__UT_ROBOT_GO2_SDK_JSON_DATA_TYPE_HPP__
