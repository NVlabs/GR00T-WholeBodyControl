#ifndef OMG_DDS_CORE_POLICY_DETAIL_CORE_POLICY_HPP_
#define OMG_DDS_CORE_POLICY_DETAIL_CORE_POLICY_HPP_

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


#include <org/eclipse/cyclonedds/core/policy/PolicyDelegate.hpp>
#include <dds/core/policy/detail/TCorePolicyImpl.hpp>
#include <org/eclipse/cyclonedds/core/policy/Policy.hpp>


namespace dds { namespace core { namespace policy { namespace detail {
#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
    typedef dds::core::policy::TDataRepresentation<org::eclipse::cyclonedds::core::policy::DataRepresentationDelegate>
    DataRepresentation;
#endif // OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

    typedef dds::core::policy::TDeadline<org::eclipse::cyclonedds::core::policy::DeadlineDelegate>
    Deadline;

    typedef dds::core::policy::TDestinationOrder<org::eclipse::cyclonedds::core::policy::DestinationOrderDelegate>
    DestinationOrder;

    typedef dds::core::policy::TDurability<org::eclipse::cyclonedds::core::policy::DurabilityDelegate>
    Durability;

#ifdef  OMG_DDS_PERSISTENCE_SUPPORT
    typedef dds::core::policy::TDurabilityService<org::eclipse::cyclonedds::core::policy::DurabilityServiceDelegate>
    DurabilityService;
#endif  // OMG_DDS_PERSISTENCE_SUPPORT

    typedef dds::core::policy::TEntityFactory<org::eclipse::cyclonedds::core::policy::EntityFactoryDelegate>
    EntityFactory;

    typedef dds::core::policy::TGroupData<org::eclipse::cyclonedds::core::policy::GroupDataDelegate>
    GroupData;

    typedef dds::core::policy::THistory<org::eclipse::cyclonedds::core::policy::HistoryDelegate>
    History;

    typedef dds::core::policy::TLatencyBudget<org::eclipse::cyclonedds::core::policy::LatencyBudgetDelegate>
    LatencyBudget;

    typedef dds::core::policy::TLifespan<org::eclipse::cyclonedds::core::policy::LifespanDelegate>
    Lifespan;

    typedef dds::core::policy::TLiveliness<org::eclipse::cyclonedds::core::policy::LivelinessDelegate>
    Liveliness;

    typedef dds::core::policy::TOwnership<org::eclipse::cyclonedds::core::policy::OwnershipDelegate>
    Ownership;

#ifdef  OMG_DDS_OWNERSHIP_SUPPORT
    typedef dds::core::policy::TOwnershipStrength<org::eclipse::cyclonedds::core::policy::OwnershipStrengthDelegate>
    OwnershipStrength;
#endif  // OMG_DDS_OWNERSHIP_SUPPORT

    typedef dds::core::policy::TPartition<org::eclipse::cyclonedds::core::policy::PartitionDelegate>
    Partition;

    typedef dds::core::policy::TPresentation<org::eclipse::cyclonedds::core::policy::PresentationDelegate>
    Presentation;

    typedef dds::core::policy::TReaderDataLifecycle<org::eclipse::cyclonedds::core::policy::ReaderDataLifecycleDelegate>
    ReaderDataLifecycle;

    typedef dds::core::policy::TReliability<org::eclipse::cyclonedds::core::policy::ReliabilityDelegate>
    Reliability;

    typedef dds::core::policy::TResourceLimits<org::eclipse::cyclonedds::core::policy::ResourceLimitsDelegate>
    ResourceLimits;

    typedef dds::core::policy::TTimeBasedFilter<org::eclipse::cyclonedds::core::policy::TimeBasedFilterDelegate>
    TimeBasedFilter;

    typedef dds::core::policy::TTopicData<org::eclipse::cyclonedds::core::policy::TopicDataDelegate>
    TopicData;

    typedef dds::core::policy::TTransportPriority<org::eclipse::cyclonedds::core::policy::TransportPriorityDelegate>
    TransportPriority;

#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
    typedef dds::core::policy::TTypeConsistencyEnforcement<org::eclipse::cyclonedds::core::policy::TypeConsistencyEnforcementDelegate>
    TypeConsistencyEnforcement;
#endif // OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

    typedef dds::core::policy::TUserData<org::eclipse::cyclonedds::core::policy::UserDataDelegate>
    UserData;

    typedef dds::core::policy::TWriterDataLifecycle<org::eclipse::cyclonedds::core::policy::WriterDataLifecycleDelegate>
    WriterDataLifecycle;
} } } } // namespace dds::core::policy::detail


#endif /* OMG_DDS_CORE_POLICY_DETAIL_CORE_POLICY_HPP_ */
