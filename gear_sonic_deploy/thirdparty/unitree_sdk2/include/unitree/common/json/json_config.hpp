#ifndef __UT_JSON_CONFIG_HPP__
#define __UT_JSON_CONFIG_HPP__

#include <unitree/common/json/json.hpp>

#define UT_JSON_CONF_KEY_PARAMETER  "Parameter"

namespace unitree
{
namespace common
{
class JsonConfig
{
public:
    JsonConfig();
    virtual ~JsonConfig();

    JsonConfig(const std::string& configFileName);

    virtual void Parse(const std::string& configFileName);
    virtual void ParseContent(const std::string& content);

    //top-level field
    bool Has(const std::string& name) const;

    //top-level field
    const Any& Get(const std::string& name) const;

    //top-level field
    template<typename T>
    const T& Get(const std::string& name) const
    {
        return AnyCast<T>(Get(name));
    }

    //top-level field
    template<typename T>
    T GetNumber(const std::string& name) const
    {
        return AnyNumberCast<T>(Get(name));
    }

    //top-level field
    template<typename T>
    T Get(const std::string& name, const T& defValue) const
    {
        const Any& a = Get(name);

        if (a.Empty())
        {
            return defValue;
        }
        else
        {
            return AnyCast<T>(a);
        }
    }

    //top-level field
    template<typename T>
    T GetNumber(const std::string& name, const T& defValue) const
    {
        const Any& a = Get(name);

        if (a.Empty())
        {
            return defValue;
        }
        else
        {
            return AnyNumberCast<T>(a);
        }
    }

    //JsonMap field
    bool Has(const JsonMap& jsonMap, const std::string& name) const;

    //JsonMap field
    const Any& Get(const JsonMap& jsonMap, const std::string& name) const;

    //JsonMap field
    template<typename T>
    const T& Get(const JsonMap& jsonMap, const std::string& name) const
    {
        return AnyCast<T>(Get(jsonMap, name));
    }

    //JsonMap field
    template<typename T>
    T GetNumber(const JsonMap& jsonMap, const std::string& name) const
    {
        return AnyNumberCast<T>(Get(jsonMap, name));
    }

    //JsonMap field
    template<typename T>
    T Get(const JsonMap& jsonMap, const std::string& name, const T& defValue) const
    {
        const Any& a = Get(jsonMap, name);

        if (a.Empty())
        {
            return defValue;
        }
        else
        {
            return AnyCast<T>(a);
        }
    }

    //JsonMap field
    template<typename T>
    T GetNumber(const JsonMap& jsonMap, const std::string& name, const T& defValue) const
    {
        const Any& a = Get(jsonMap, name);

        if (a.Empty())
        {
            return defValue;
        }
        else
        {
            return AnyNumberCast<T>(a);
        }
    }

    //top-level field
    const Any& GetGlobalParameter(const std::string& name) const;

    //top-level field: Parameter
    bool HasParameter(const std::string& name) const;

    //top-level field: Parameter
    const JsonMap& GetParameter() const;

    //get field/value from top-level field: Parameter
    const Any& GetParameter(const std::string& name) const;

    //get field/value from top-level field: Parameter
    template<typename T>
    const T& GetParameter(const std::string& name) const
    {
        return AnyCast<T>(GetParameter(name));
    }

    //get field/value from top-level field: Parameter
    template<typename T>
    T GetNumberParameter(const std::string& name) const
    {
        return AnyNumberCast<T>(GetParameter(name));
    }

    //get field/value from top-level field: Parameter
    template<typename T>
    T GetParameter(const std::string& name, const T& defValue) const
    {
        const Any& a = GetParameter(name);

        if (a.Empty())
        {
            return defValue;
        }
        else
        {
            return AnyCast<T>(a);
        }
    }

    template<typename T>
    T GetNumberParameter(const std::string& name, const T& defValue) const
    {
        const Any& a = GetParameter(name);

        if (a.Empty())
        {
            return defValue;
        }
        else
        {
            return AnyNumberCast<T>(a);
        }
    }

protected:
    JsonMap mParameter;
    Any mContent;
};

typedef std::shared_ptr<JsonConfig> JsonConfigPtr;

}
}

#endif//__UT_JSON_CONFIG_HPP__
