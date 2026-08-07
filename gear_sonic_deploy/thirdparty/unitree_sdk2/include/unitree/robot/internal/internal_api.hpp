#ifndef __UT_ROBOT_SDK_INERNAL_API_HPP__
#define __UT_ROBOT_SDK_INERNAL_API_HPP__

#include <unitree/common/decl.hpp>
#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
/*
 * @brief  max internal api id
 * @value: 100
 */
const int32_t ROBOT_INTERNAL_API_ID_MAX             = 100;

///////////////////////////////////////////////////////////////
/*
 * @brief  invailed api id
 * @value: -1
 */
const int32_t ROBOT_API_ID_NONE                     = -1;

/*
 * @brief  Get api version from server.
 * @value: 1
 */
const int32_t ROBOT_API_ID_INTERNAL_API_VERSION     = 1;

/*
 * @brief  Noop.
 * @value: 2
 */
const int32_t ROBOT_API_ID_INTERNAL_API_NOOP        = 2;

/*
 * @brief  Apply lease from server.
 * @value: 101
 */
const int32_t ROBOT_API_ID_LEASE_APPLY              = 101;
/*
 * @brief  Renewal lease term from server.
 * @value: 102
 */
const int32_t ROBOT_API_ID_LEASE_RENEWAL            = 102;

///////////////////////////////////////////////////////////////

/*
 * @biref  robot lease term default.
 * @value: default 1000000 us
 */
const int32_t ROBOT_LEASE_TERM                      = 1000000;

/*
 * micro: IS_INTERNAL_API
 */
#define IS_INTERNAL_API(apiId) ((apiId) <= ROBOT_MAX_INTERNAL_API_ID)

///////////////////////////////////////////////////////////////

/*
 * @brief  Input parameter type for ROBOT_API_ID_INTERNAL_APPLY_LEASE
 * @class: ApplyLeaseParameter
 */
class ApplyLeaseParameter : public common::Jsonize
{
public:
    ApplyLeaseParameter()
    {}

    ~ApplyLeaseParameter()
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

/*
 * @brief  Output data type for ROBOT_API_ID_INTERNAL_APPLY_LEASE
 * @class: ApplyLeaseData
 */
class ApplyLeaseData : public common::Jsonize
{
public:
    ApplyLeaseData() : id(0),term(0)
    {}

    ~ApplyLeaseData()
    {}

    void fromJson(common::JsonMap& json)
    {
        common::FromJson(json["id"], id);
        common::FromJson(json["term"], term);
    }

    void toJson(common::JsonMap& json) const
    {
        common::ToJson(id, json["id"]);
        common::ToJson(term, json["term"]);
    }

public:
    int64_t id;
    int64_t term;
};
}
}
#endif//__UT_ROBOT_SDK_INERNAL_API_HPP__
