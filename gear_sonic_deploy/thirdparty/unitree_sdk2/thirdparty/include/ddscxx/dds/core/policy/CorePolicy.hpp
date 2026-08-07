#ifndef OMG_DDS_CORE_POLICY_CORE_POLICY_HPP_
#define OMG_DDS_CORE_POLICY_CORE_POLICY_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Corp.
 * Copyright 2010, Real-Time Innovations, Inc.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dds/core/policy/PolicyKind.hpp>
#include <dds/core/policy/detail/CorePolicy.hpp>
#include <dds/core/policy/detail/QosPolicyCount.hpp>



//==============================================================================
// MACROS
//
#define OMG_DDS_POLICY_TRAITS(POLICY, ID) \
    template <> \
    class OMG_DDS_API policy_id<POLICY> { \
    public: \
        static const dds::core::policy::QosPolicyId value = ID; \
    }; \
    template <> \
    class OMG_DDS_API policy_name<POLICY> { \
    public:\
        static const std::string& name(); \
    };

#define OMG_DDS_DEFINE_POLICY_TRAITS(POLICY, NAME) \
    const std::string& dds::core::policy::policy_name<POLICY>::name() { \
        static std::string name = #NAME; \
        return name; \
    }


namespace dds
{
namespace core
{
namespace policy
{

//==============================================================================
// Policy Trait Classes

template <typename Policy>
class policy_id { };

template <typename Policy>
class policy_name { };



#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
typedef dds::core::policy::detail::DataRepresentation
DataRepresentation;
#endif  //OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

typedef dds::core::policy::detail::Deadline
Deadline;

typedef dds::core::policy::detail::DestinationOrder
DestinationOrder;

typedef dds::core::policy::detail::Durability
Durability;

typedef dds::core::policy::detail::EntityFactory
EntityFactory;

typedef dds::core::policy::detail::GroupData
GroupData;

typedef dds::core::policy::detail::History
History;

typedef dds::core::policy::detail::LatencyBudget
LatencyBudget;

typedef dds::core::policy::detail::Lifespan
Lifespan;

typedef dds::core::policy::detail::Liveliness
Liveliness;

typedef dds::core::policy::detail::Ownership
Ownership;

#ifdef  OMG_DDS_OWNERSHIP_SUPPORT
typedef dds::core::policy::detail::OwnershipStrength
OwnershipStrength;
#endif  // OMG_DDS_OWNERSHIP_SUPPORT

typedef dds::core::policy::detail::Partition
Partition;

template <typename Policy>
class policy_id;

template <typename Policy>
class policy_name;

typedef dds::core::policy::detail::Presentation
Presentation;

typedef ::dds::core::policy::detail::QosPolicyCount QosPolicyCount;

typedef std::vector<QosPolicyCount> QosPolicyCountSeq;

typedef uint32_t QosPolicyId;

typedef dds::core::policy::detail::ReaderDataLifecycle
ReaderDataLifecycle;

typedef dds::core::policy::detail::Reliability
Reliability;

typedef dds::core::policy::detail::ResourceLimits
ResourceLimits;

typedef dds::core::policy::detail::TimeBasedFilter
TimeBasedFilter;

typedef dds::core::policy::detail::TopicData
TopicData;

typedef dds::core::policy::detail::TransportPriority
TransportPriority;

#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
typedef dds::core::policy::detail::TypeConsistencyEnforcement
TypeConsistencyEnforcement;
#endif  //OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

typedef dds::core::policy::detail::UserData
UserData;

typedef dds::core::policy::detail::WriterDataLifecycle
WriterDataLifecycle;

#ifdef  OMG_DDS_PERSISTENCE_SUPPORT
typedef ::dds::core::policy::detail::DurabilityService
DurabilityService;
#endif  // OMG_DDS_PERSISTENCE_SUPPORT


OMG_DDS_POLICY_TRAITS(UserData,             1)
OMG_DDS_POLICY_TRAITS(Durability,           2)
OMG_DDS_POLICY_TRAITS(Presentation,         3)
OMG_DDS_POLICY_TRAITS(Deadline,             4)
OMG_DDS_POLICY_TRAITS(LatencyBudget,        5)
OMG_DDS_POLICY_TRAITS(Ownership,            6)

#ifdef OMG_DDS_OWNERSHIP_SUPPORT
OMG_DDS_POLICY_TRAITS(OwnershipStrength,    7)
#endif  // OMG_DDS_OWNERSHIP_SUPPORT

OMG_DDS_POLICY_TRAITS(Liveliness,           8)
OMG_DDS_POLICY_TRAITS(TimeBasedFilter,      9)
OMG_DDS_POLICY_TRAITS(Partition,            10)
OMG_DDS_POLICY_TRAITS(Reliability,          11)
OMG_DDS_POLICY_TRAITS(DestinationOrder,     12)
OMG_DDS_POLICY_TRAITS(History,              13)
OMG_DDS_POLICY_TRAITS(ResourceLimits,       14)
OMG_DDS_POLICY_TRAITS(EntityFactory,        15)
OMG_DDS_POLICY_TRAITS(WriterDataLifecycle,  16)
OMG_DDS_POLICY_TRAITS(ReaderDataLifecycle,  17)
OMG_DDS_POLICY_TRAITS(TopicData,            18)
OMG_DDS_POLICY_TRAITS(GroupData,            19)
OMG_DDS_POLICY_TRAITS(TransportPriority,    20)
OMG_DDS_POLICY_TRAITS(Lifespan,             21)

#ifdef  OMG_DDS_PERSISTENCE_SUPPORT
OMG_DDS_POLICY_TRAITS(DurabilityService,    22)
#endif  // OMG_DDS_PERSISTENCE_SUPPORT

#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
OMG_DDS_POLICY_TRAITS(DataRepresentation,   23)
OMG_DDS_POLICY_TRAITS(TypeConsistencyEnforcement, 24)
#endif  // OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

}
}
}


#endif /* OMG_DDS_CORE_POLICY_CORE_POLICY_HPP_ */
