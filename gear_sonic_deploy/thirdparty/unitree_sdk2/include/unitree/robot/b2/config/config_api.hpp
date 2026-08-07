#ifndef __UT_ROBOT_B2_CONFIG_API_HPP__
#define __UT_ROBOT_B2_CONFIG_API_HPP__

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
const std::string CONFIG_SERVICE_NAME = "config";

/*
 * api version
 */
const std::string CONFIG_API_VERSION = "1.0.0.1";

/*
 * api id
 */
const int32_t CONFIG_API_ID_SET = 1001;
const int32_t CONFIG_API_ID_GET = 1002;
const int32_t CONFIG_API_ID_DEL = 1003;
const int32_t CONFIG_API_ID_META = 1004;

/*
 * data type
 */
class JsonizeConfigMeta : public common::Jsonize
{
public:
    JsonizeConfigMeta()
    {}

    ~JsonizeConfigMeta()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);
        common::FromJson(json["lastModified"], lastModified);
        common::FromJson(json["size"], size);
        common::FromJson(json["epoch"], epoch);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);
        common::ToJson(lastModified, json["lastModified"]);
        common::ToJson(size, json["size"]);
        common::ToJson(epoch, json["epoch"]);
    }

public:
    std::string name;
    std::string lastModified;
    int32_t size;
    int32_t epoch;
};

class ConfigSetParameter : public common::Jsonize
{
public:
    ConfigSetParameter()
    {}

    ~ConfigSetParameter()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);
        common::FromJson(json["content"], content);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);
        common::ToJson(content, json["content"]);
    }

public:
    std::string name;
    std::string content;
};

class ConfigGetParameter : public common::Jsonize
{
public:
    ConfigGetParameter()
    {}

    ~ConfigGetParameter()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);
    }

public:
    std::string name;
};

class ConfigGetData : public common::Jsonize
{
public:
    ConfigGetData()
    {}

    ~ConfigGetData()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["content"], content);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(content, json["content"]);
    }

public:
    std::string content;
};

class ConfigDelParameter : public common::Jsonize
{
public:
    ConfigDelParameter()
    {}

    ~ConfigDelParameter()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);
    }

public:
    std::string name;
};

class ConfigMetaParameter : public common::Jsonize
{
public:
    ConfigMetaParameter()
    {}

    ~ConfigMetaParameter()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["name"], name);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(name, json["name"]);
    }

public:
    std::string name;
};

class ConfigMetaData : public common::Jsonize
{
public:
    ConfigMetaData()
    {}

    ~ConfigMetaData()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["meta"], meta);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(meta, json["meta"]);
    }

public:
    JsonizeConfigMeta meta;
};

}
}
}

#endif//__UT_ROBOT_B2_CONFIG_API_HPP__
