#ifndef OMG_TDDS_TOPIC_BUILT_IN_TOPIC_HPP_
#define OMG_TDDS_TOPIC_BUILT_IN_TOPIC_HPP_

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

#include <dds/core/detail/conformance.hpp>
#include <dds/core/Value.hpp>
#include <dds/core/policy/CorePolicy.hpp>
#include <dds/topic/BuiltinTopicKey.hpp>

namespace dds
{
namespace topic
{
template <typename D>
class TParticipantBuiltinTopicData;

template <typename D>
class TTopicBuiltinTopicData;

template <typename D>
class TPublicationBuiltinTopicData;

template <typename D>
class TSubscriptionBuiltinTopicData;
}
}

/**
 * @brief
 * Class that contains information about available DomainParticipants within
 * the system.
 *
 * The DCPSParticipant topic communicates the existence of DomainParticipants
 * by means of the ParticipantBuiltinTopicData datatype. Each
 * ParticipantBuiltinTopicData sample in a Domain represents a DomainParticipant
 * that participates in that Domain: a new ParticipantBuiltinTopicData instance
 * is created when a newly-added DomainParticipant is enabled, and it is disposed
 * when that DomainParticipant is deleted. An updated ParticipantBuiltinTopicData
 * sample is written each time the DomainParticipant modifies its UserDataQosPolicy.
 *
 * @code{.cpp}
 * // Get builtin subscriber
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::sub::Subscriber builtinSubscriber = dds::sub::builtin_subscriber(participant);
 *
 * // Get DCPSParticipant builtin reader (happy flow)
 * string name = "DCPSParticipant";
 * vector<dds::sub::DataReader<dds::topic::ParticipantBuiltinTopicData> > readersVector;
 * dds::sub::find<dds::sub::DataReader<dds::topic::ParticipantBuiltinTopicData>,
 *                       back_insert_iterator<vector<dds::sub::DataReader<dds::topic::ParticipantBuiltinTopicData> > > >(
 *           builtinSubscriber,
 *           name,
 *           back_inserter<vector<dds::sub::DataReader<dds::topic::ParticipantBuiltinTopicData> > >(readersVector));
 * dds::sub::DataReader<dds::topic::ParticipantBuiltinTopicData> builtinReader = readersVector[0];
 *
 * // The builtinReader can now be used just as a normal dds::sub::DataReader to get
 * // dds::topic::ParticipantBuiltinTopicData samples.
 * @endcode
 *
 * @see for more information: @ref DCPS_Builtin_Topics
 * @see for more information: @ref DCPS_Builtin_Topics_ParticipantData
 */
template <typename D>
class dds::topic::TParticipantBuiltinTopicData : public ::dds::core::Value<D>
{
public:
    /**
     * Globally unique identifier of the participant
     */
    const dds::topic::BuiltinTopicKey& key() const;

    /**
     * User-defined data attached to the participant via a QosPolicy
     */
    const ::dds::core::policy::UserData& user_data() const;
};

/**
 * @brief
 * Class that contains information about available Topics within
 * the system.
 *
 * The DCPSTopic topic communicates the existence of topics by means of the
 * TopicBuiltinTopicData datatype. Each TopicBuiltinTopicData sample in
 * a Domain represents a Topic in that Domain: a new TopicBuiltinTopicData
 * instance is created when a newly-added Topic is enabled. However, the instance is
 * not disposed when a Topic is deleted by its participant because a topic lifecycle
 * is tied to the lifecycle of a Domain, not to the lifecycle of an individual
 * participant. An updated TopicBuiltinTopicData sample is written each time a
 * Topic modifies one or more of its QosPolicy values.
 *
 * @code{.cpp}
 * // Get builtin subscriber
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::sub::Subscriber builtinSubscriber = dds::sub::builtin_subscriber(participant);
 *
 * // Get DCPSTopic builtin reader (happy flow)
 * string name = "DCPSTopic";
 * vector<dds::sub::DataReader<dds::topic::TopicBuiltinTopicData> > readersVector;
 * dds::sub::find<dds::sub::DataReader<dds::topic::TopicBuiltinTopicData>,
 *                       back_insert_iterator<vector<dds::sub::DataReader<dds::topic::TopicBuiltinTopicData> > > >(
 *           builtinSubscriber,
 *           name,
 *           back_inserter<vector<dds::sub::DataReader<dds::topic::TopicBuiltinTopicData> > >(readersVector));
 * dds::sub::DataReader<dds::topic::TopicBuiltinTopicData> builtinReader = readersVector[0];
 *
 * // The builtinReader can now be used just as a normal dds::sub::DataReader to get
 * // dds::topic::TopicBuiltinTopicData samples.
 * @endcode
 *
 * @see for more information: @ref DCPS_Builtin_Topics
 * @see for more information: @ref DCPS_Builtin_Topics_TopicData
 */
template <typename D>
class dds::topic::TTopicBuiltinTopicData : public ::dds::core::Value<D>
{
public:
    /**
     * Global unique identifier of the Topic
     */
    const dds::topic::BuiltinTopicKey& key() const;

    /**
     * Name of the Topic
     */
    const std::string&                  name() const;

    /**
     * Type name of the Topic (i.e. the fully scoped IDL name)
     */
    const std::string&                  type_name() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::Durability&         durability() const;

    #ifdef OMG_DDS_PERSISTENCE_SUPPORT
    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::DurabilityService&  durability_service() const;
    #endif  // OMG_DDS_PERSISTENCE_SUPPORT

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::Deadline&           deadline() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::LatencyBudget&      latency_budget() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::Liveliness&         liveliness() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::Reliability&        reliability() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::TransportPriority&  transport_priority() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::Lifespan&           lifespan() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::DestinationOrder&   destination_order() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::History&            history() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::ResourceLimits&     resource_limits() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::Ownership&          ownership() const;

    /**
     * QosPolicy attached to the Topic
     */
    const ::dds::core::policy::TopicData&          topic_data() const;
};

/**
 * @brief
 * Class that contains information about available DataWriters within
 * the system.
 *
 * The DCPSPublication topic communicates the existence of datawriters by means
 * of the PublicationBuiltinTopicData datatype. Each PublicationBuiltinTopicData
 * sample in a Domain represents a datawriter in that Domain: a new
 * PublicationBuiltinTopicData instance is created when a newly-added DataWriter
 * is enabled, and it is disposed when that DataWriter is deleted. An updated
 * PublicationBuiltinTopicData sample is written each time the DataWriter (or
 * the Publisher to which it belongs) modifies a QosPolicy that applies to the
 * entities connected to it. Also will it be updated when the writer looses or
 * regains its liveliness.
 *
 * @code{.cpp}
 * // Get builtin subscriber
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::sub::Subscriber builtinSubscriber = dds::sub::builtin_subscriber(participant);
 *
 * // Get DCPSPublication builtin reader (happy flow)
 * string name = "DCPSPublication";
 * vector<dds::sub::DataReader<dds::topic::PublicationBuiltinTopicData> > readersVector;
 * dds::sub::find<dds::sub::DataReader<dds::topic::PublicationBuiltinTopicData>,
 *                       back_insert_iterator<vector<dds::sub::DataReader<dds::topic::PublicationBuiltinTopicData> > > >(
 *           builtinSubscriber,
 *           name,
 *           back_inserter<vector<dds::sub::DataReader<dds::topic::PublicationBuiltinTopicData> > >(readersVector));
 * dds::sub::DataReader<dds::topic::PublicationBuiltinTopicData> builtinReader = readersVector[0];
 *
 * // The builtinReader can now be used just as a normal dds::sub::DataReader to get
 * // dds::topic::PublicationBuiltinTopicData samples.
 * @endcode
 *
 * @see for more information: @ref DCPS_Builtin_Topics
 * @see for more information: @ref DCPS_Builtin_Topics_PublicationData
 */
template <typename D>
class dds::topic::TPublicationBuiltinTopicData  : public ::dds::core::Value<D>
{
public:
    /**
     * Global unique identifier of the DataWriter
     */
    const dds::topic::BuiltinTopicKey& key() const;

    /**
     * Global unique identifier of the Participant to which the DataWriter belongs
     */
    const dds::topic::BuiltinTopicKey& participant_key() const;

    /**
     * Name of the Topic used by the DataWriter
     */
    const std::string&                  topic_name() const;

    /**
     * Type name of the Topic used by the DataWriter
     */
    const std::string&                  type_name() const;

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::Durability&         durability() const;

    #ifdef OMG_DDS_PERSISTENCE_SUPPORT
    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::DurabilityService&  durability_service() const;
    #endif  // OMG_DDS_PERSISTENCE_SUPPORT

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::Deadline&           deadline() const;

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::LatencyBudget&      latency_budget() const;

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::Liveliness&         liveliness() const;

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::Reliability&        reliability() const;

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::Lifespan&           lifespan() const;

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::UserData&           user_data() const;

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::Ownership&          ownership() const;

    #ifdef OMG_DDS_OWNERSHIP_SUPPORT
    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::OwnershipStrength&  ownership_strength() const;
    #endif  // OMG_DDS_OWNERSHIP_SUPPORT

    /**
     * QosPolicy attached to the DataWriter
     */
    const ::dds::core::policy::DestinationOrder&   destination_order() const;

    /**
     * QosPolicy attached to the Publisher to which the DataWriter belongs
     */
    const ::dds::core::policy::Presentation&       presentation() const;

    /**
     * QosPolicy attached to the Publisher to which the DataWriter belongs
     */
    const ::dds::core::policy::Partition&          partition() const;

    /**
     * QosPolicy attached to the Publisher to which the DataWriter belongs
     */
    const ::dds::core::policy::TopicData&          topic_data() const;

    /**
     * QosPolicy attached to the Publisher to which the DataWriter belongs
     */
    const ::dds::core::policy::GroupData&          group_data() const;

};

/**
 * @brief
 * Class that contains information about available DataReaders within
 * the system.
 *
 * The DCPSSubscription topic communicates the existence of datareaders by
 * means of the SubscriptionBuiltinTopicData datatype. Each
 * SubscriptionBuiltinTopicData sample in a Domain represents a datareader
 * in that Domain: a new SubscriptionBuiltinTopicData instance is created
 * when a newly-added DataReader is enabled, and it is disposed when that
 * DataReader is deleted. An updated SubscriptionBuiltinTopicData sample is
 * written each time the DataReader (or the Subscriber to which it belongs)
 * modifies a QosPolicy that applies to the entities connected to it.
 *
 * @code{.cpp}
 * // Get builtin subscriber
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::sub::Subscriber builtinSubscriber = dds::sub::builtin_subscriber(participant);
 *
 * // Get DCPSSubscription builtin reader (happy flow)
 * string name = "DCPSSubscription";
 * vector<dds::sub::DataReader<dds::topic::SubscriptionBuiltinTopicData> > readersVector;
 * dds::sub::find<dds::sub::DataReader<dds::topic::SubscriptionBuiltinTopicData>,
 *                       back_insert_iterator<vector<dds::sub::DataReader<dds::topic::SubscriptionBuiltinTopicData> > > >(
 *           builtinSubscriber,
 *           name,
 *           back_inserter<vector<dds::sub::DataReader<dds::topic::SubscriptionBuiltinTopicData> > >(readersVector));
 * dds::sub::DataReader<dds::topic::SubscriptionBuiltinTopicData> builtinReader = readersVector[0];
 *
 * // The builtinReader can now be used just as a normal dds::sub::DataReader to get
 * // dds::topic::SubscriptionBuiltinTopicData samples.
 * @endcode
 *
 * @see for more information: @ref DCPS_Builtin_Topics
 * @see for more information: @ref DCPS_Builtin_Topics_SubscriptionData
 */
template <typename D>
class dds::topic::TSubscriptionBuiltinTopicData  : public ::dds::core::Value<D>
{
public:
    /**
     * Global unique identifier of the DataReader
     */
    const dds::topic::BuiltinTopicKey& key() const;

    /**
     * Global unique identifier of the Participant to which the DataReader belongs
     */
    const dds::topic::BuiltinTopicKey& participant_key() const;

    /**
     * Name of the Topic used by the DataReader
     */
    const std::string&                  topic_name() const;

    /**
     * Type name of the Topic used by the DataReader
     */
    const std::string&                  type_name() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::Durability&         durability() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::Deadline&           deadline() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::LatencyBudget&      latency_budget() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::Liveliness&         liveliness() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::Reliability&        reliability() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::Ownership&          ownership() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::DestinationOrder&   destination_order() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::UserData&           user_data() const;

    /**
     * QosPolicy attached to the DataReader
     */
    const ::dds::core::policy::TimeBasedFilter&    time_based_filter() const;

    /**
     * QosPolicy attached to the Subscriber to which the DataReader belongs
     */
    const ::dds::core::policy::Presentation&       presentation() const;

    /**
     * QosPolicy attached to the Subscriber to which the DataReader belongs
     */
    const ::dds::core::policy::Partition&          partition() const;

    /**
     * QosPolicy attached to the Subscriber to which the DataReader belongs
     */
    const ::dds::core::policy::TopicData&          topic_data() const;

    /**
     * QosPolicy attached to the Subscriber to which the DataReader belongs
     */
    const ::dds::core::policy::GroupData&          group_data() const;

};

#endif /* OMG_TDDS_TOPIC_BUILT_IN_TOPIC_HPP_ */
