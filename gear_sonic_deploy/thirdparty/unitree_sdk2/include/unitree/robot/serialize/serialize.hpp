#ifndef __UT_ROBOT_SDK_SERIALIZE_HPP__
#define __UT_ROBOT_SDK_SERIALIZE_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
template<typename T>
inline bool Serialize(const T& instance, std::string& serialziedData)
{
    try
    {
        serialziedData = common::ToJsonString(instance);
    }
    catch(const common::Exception& e)
    {
        return false;
    }

    return true;
}

template<typename T>
inline bool Deserialize(const std::string& serialziedData, T& instance)
{
    try
    {
        instance = common::FromJsonString<T>(serialziedData);
    }
    catch(const common::Exception& e)
    {
        return false;
    }

    return true;
}

}
}

#endif//__UT_ROBOT_SDK_SERIALIZE_HPP__
