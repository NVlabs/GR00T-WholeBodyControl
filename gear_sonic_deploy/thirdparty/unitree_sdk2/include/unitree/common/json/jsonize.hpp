#ifndef __UT_JSONIZE_HPP__
#define __UT_JSONIZE_HPP__

#define JN_FROM_WEAK(m, name, value) \
    if (m.find(name) != m.end()){ JN_FROM(m, name, value); }

#define JN_FROM(m, name, value) \
    unitree::common::FromJson(m[name], value)

#define JN_TO(m, name, value) \
    unitree::common::ToJson(value, m[name])

#include <unitree/common/json/json.hpp>

namespace unitree
{
namespace common
{
template<typename T>
void FromJson(const Any& a, T& t);

template<typename T>
void ToJson(const T& value, Any& a);

template<typename T>
void FromJsonString(const std::string& s, T& t)
{
    Any a = FromJsonString(s);
    FromJson<T>(a, t);
}

template<typename T>
std::string ToJsonString(const T& t, bool pretty = false)
{
    Any a;
    ToJson<T>(t, a);
    return ToJsonString(a, pretty);
}

class Jsonize
{
public:
    virtual void toJson(JsonMap& a) const = 0;
    virtual void fromJson(JsonMap& a) = 0;
};

void FromAny(const Any& a, int8_t& value);

void FromAny(const Any& a, uint8_t& value);

void FromAny(const Any& a, int16_t& value);

void FromAny(const Any& a, uint16_t& value);

void FromAny(const Any& a, int32_t& value);

void FromAny(const Any& a, uint32_t& value);

void FromAny(const Any& a, int64_t& value);

void FromAny(const Any& a, uint64_t& value);

void FromAny(const Any& a, float& value);

void FromAny(const Any& a, double& value);

void FromAny(const Any& a, bool& value);

void FromAny(const Any& a, std::string& value);

void FromAny(const Any& a, JsonMap& value);

void FromAny(const Any& a, JsonArray& value);

void FromAny(const Any& a, Jsonize& value);

template<typename E>
void FromAny(const Any& a, std::vector<E>& value)
{
    if (a.Empty())
    {
        return;
    }

    const JsonArray& arr = AnyCast<JsonArray>(a);
    size_t i, count = arr.size();

    for (i=0; i<count; i++)
    {
        E e;
        const Any& a_in = arr[i];
        FromJson<E>(a_in, e);
        value.push_back(std::move(e));
    }
}

template<typename E>
void FromAny(const Any& a, std::list<E>& value)
{
    if (a.Empty())
    {
        return;
    }

    const JsonArray& arr = AnyCast<JsonArray>(a);
    size_t i, count = arr.size();

    for (i=0; i<count; i++)
    {
        E e;
        const Any& a_in = arr[i];
        FromJson<E>(a_in, e);
        value.push_back(std::move(e));
    }
}

template<typename E>
void FromAny(const Any& a, std::set<E>& value)
{
    if (a.Empty())
    {
        return;
    }

    const JsonArray& arr = AnyCast<JsonArray>(a);
    size_t i, count = arr.size();

    for (i=0; i<count; i++)
    {
        E e;
        const Any& a_in = arr[i];
        FromJson<E>(a_in, e);
        value.insert(std::move(e));
    }
}

template<typename E>
void FromAny(const Any& a, std::map<std::string,E>& value)
{
    if (a.Empty())
    {
        return;
    }

    const JsonMap& m = AnyCast<JsonMap>(a);
    JsonMap::const_iterator iter;

    for (iter = m.begin(); iter != m.end(); ++iter)
    {
        E e;
        const std::string& name = iter->first;
        const Any& a_in = iter->second;
        FromJson<E>(a_in, e);
        value[name] = std::move(e);
    }
}

template<typename T>
void FromJson(const Any& a, T& t)
{
    FromAny(a, t);
}

void ToAny(const int8_t& value, Any& a);

void ToAny(const uint8_t& value, Any& a);

void ToAny(const int16_t& value, Any& a);

void ToAny(const uint16_t& value, Any& a);

void ToAny(const int32_t& value, Any& a);

void ToAny(const uint32_t& value, Any& a);

void ToAny(const int64_t& value, Any& a);

void ToAny(const uint64_t& value, Any& a);

void ToAny(const float& value, Any& a);

void ToAny(const double& value, Any& a);

void ToAny(const bool& value, Any& a);

void ToAny(const std::string& value, Any& a);

void ToAny(const JsonMap& value, Any& a);

void ToAny(const JsonArray& value, Any& a);

void ToAny(const Jsonize& value, Any& a);

template<typename E>
void ToAny(const std::vector<E>& value, Any& a)
{
    JsonArray arr;

    size_t i, count = value.size();
    for (i=0; i<count; i++)
    {
        Any a_in;
        const E& e = value[i];

        ToJson<E>(e, a_in);
        arr.push_back(std::move(a_in));
    }

    a = Any(arr);
}

template<typename E>
void ToAny(const std::list<E>& value, Any& a)
{
    JsonArray arr;

    typename std::list<E>::const_iterator iter;
    for (iter = value.begin(); iter != value.end(); ++iter)
    {
        Any a_in;
        const E& e = *iter;

        ToJson<E>(e, a_in);
        arr.push_back(std::move(a_in));
    }

    a = Any(arr);
}

template<typename E>
void ToAny(const std::set<E>& value, Any& a)
{
    JsonArray arr;

    typename std::set<E>::const_iterator iter;
    for (iter = value.begin(); iter != value.end(); ++iter)
    {
        Any a_in;
        const E& e = *iter;

        ToJson<E>(e, a_in);
        arr.push_back(std::move(a_in));
    }

    a = Any(arr);
}

template<typename E>
void ToAny(const std::map<std::string,E>& value, Any& a)
{
    JsonMap m;
    typename std::map<std::string,E>::const_iterator iter;

    for (iter = value.begin(); iter != value.end(); ++iter)
    {
        Any a_in;
        const std::string& name = iter->first;
        const E& e = iter->second;

        ToJson<E>(e, a_in);
        m[name] = a_in;
    }

    a = Any(m);
}

template<typename T>
void ToJson(const T& value, Any& a)
{
    //std::cout << typeid(value).name() << std::endl;
    ToAny(value, a);
}
}
}
#endif//__UT_JSONIZE_HPP__
