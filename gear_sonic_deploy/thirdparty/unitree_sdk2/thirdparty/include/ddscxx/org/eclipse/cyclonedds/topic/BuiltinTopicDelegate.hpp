/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */


/**
 * @file
 */

#ifndef CYCLONEDDS_TOPIC_BUILTIN_TOPIC_DELEGATE_HPP
#define CYCLONEDDS_TOPIC_BUILTIN_TOPIC_DELEGATE_HPP

#include <dds/core/detail/conformance.hpp>
#include <dds/core/policy/CorePolicy.hpp>
#include <dds/topic/BuiltinTopicKey.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace topic
{
class ParticipantBuiltinTopicDataDelegate;
class TopicBuiltinTopicDataDelegate;
class PublicationBuiltinTopicDataDelegate;
class SubscriptionBuiltinTopicDataDelegate;
class CMParticipantBuiltinTopicDataDelegate;
class CMPublisherBuiltinTopicDataDelegate;
class CMSubscriberBuiltinTopicDataDelegate;
class CMDataWriterBuiltinTopicDataDelegate;
class CMDataReaderBuiltinTopicDataDelegate;
}
}
}
}

//==============================================================================
//            ParticipantBuiltinTopicDataDelegate
//==============================================================================

class org::eclipse::cyclonedds::topic::ParticipantBuiltinTopicDataDelegate
{
public:
    const dds::topic::BuiltinTopicKey& key() const
    {
        return key_;
    }

    void key(const int32_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(key));
    }

    const ::dds::core::policy::UserData& user_data() const
    {
        return user_data_;
    }

    void user_data(const dds_qos_t* policy)
    {
        user_data_.delegate().set_iso_policy(policy);
    }

    bool operator ==(const ParticipantBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_ && other.user_data_ == user_data_;
    }

protected:
    dds::topic::BuiltinTopicKey key_;
    ::dds::core::policy::UserData user_data_;
};

class org::eclipse::cyclonedds::topic::TopicBuiltinTopicDataDelegate
{
public:
    const dds::topic::BuiltinTopicKey& key() const
    {
        return key_;
    }

    void key(const int32_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(key));
    }

    const std::string&                  name() const
    {
        return name_;
    }

    void name(const char *name)
    {
        name_ = name;
    }

    const std::string&                  type_name() const
    {
        return type_name_;
    }

    void type_name(const char *name)
    {
        type_name_ = name;
    }

    const ::dds::core::policy::Durability&         durability() const
    {
        return durability_;
    }

    void durability(const dds_qos_t* policy)
    {
        durability_.delegate().set_iso_policy(policy);
    }


#ifdef  OMG_DDS_PERSISTENCE_SUPPORT

    const ::dds::core::policy::DurabilityService&  durability_service() const
    {
        return durability_service_;
    }

    void durability_service(const dds_qos_t* policy)
    {
        durability_service_.delegate().set_iso_policy(policy);
    }

#endif  // OMG_DDS_PERSISTENCE_SUPPORT


    const ::dds::core::policy::Deadline&           deadline() const
    {
        return deadline_;
    }

    void deadline(const dds_qos_t* policy)
    {
        deadline_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::LatencyBudget&      latency_budget() const
    {
        return latency_budget_;
    }

    void latency_budget(const dds_qos_t* policy)
    {
        latency_budget_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Liveliness&         liveliness() const
    {
        return liveliness_;
    }

    void liveliness(const dds_qos_t* policy)
    {
        liveliness_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Reliability&        reliability() const
    {
        return reliability_;
    }

    void reliability(const dds_qos_t* policy)
    {
        reliability_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::TransportPriority&  transport_priority() const
    {
        return transport_priority_;
    }

    void transport_priority(const dds_qos_t* policy)
    {
        transport_priority_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Lifespan&           lifespan() const
    {
        return lifespan_;
    }

    void lifespan(const dds_qos_t* policy)
    {
        lifespan_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::DestinationOrder&   destination_order() const
    {
        return destination_order_;
    }

    void destination_order(const dds_qos_t* policy)
    {
        destination_order_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::History&            history() const
    {
        return history_;
    }

    void history(const dds_qos_t* policy)
    {
        history_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::ResourceLimits&     resource_limits() const
    {
        return resource_limits_;
    }

    void resource_limits(const dds_qos_t* policy)
    {
        resource_limits_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Ownership&          ownership() const
    {
        return ownership_;
    }

    void ownership(const dds_qos_t* policy)
    {
        ownership_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::TopicData&          topic_data() const
    {
        return topic_data_;
    }

    void topic_data(const dds_qos_t* policy)
    {
        topic_data_.delegate().set_iso_policy(policy);
    }

    bool operator ==(const TopicBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_
               && other.name_ == name_
               && other.type_name_ == type_name_
               && other.durability_ == durability_
#ifdef  OMG_DDS_PERSISTENCE_SUPPORT
               && other.durability_service_ == durability_service_
#endif  // OMG_DDS_PERSISTENCE_SUPPORT
               && other.deadline_ == deadline_
               && other.latency_budget_ == latency_budget_
               && other.liveliness_ == liveliness_
               && other.reliability_ == reliability_
               && other.transport_priority_ == transport_priority_
               && other.lifespan_ == lifespan_
               && other.destination_order_ == destination_order_
               && other.history_ == history_
               && other.resource_limits_ == resource_limits_
               && other.ownership_ == ownership_
               && other.topic_data_ == topic_data_;
    }

protected:
    dds::topic::BuiltinTopicKey  key_;
    std::string                  name_;
    std::string                  type_name_;
    ::dds::core::policy::Durability         durability_;

#ifdef  OMG_DDS_PERSISTENCE_SUPPORT
    ::dds::core::policy::DurabilityService  durability_service_;
#endif  // OMG_DDS_PERSISTENCE_SUPPORT

    ::dds::core::policy::Deadline           deadline_;
    ::dds::core::policy::LatencyBudget      latency_budget_;
    ::dds::core::policy::Liveliness         liveliness_;
    ::dds::core::policy::Reliability        reliability_;
    ::dds::core::policy::TransportPriority  transport_priority_;
    ::dds::core::policy::Lifespan           lifespan_;
    ::dds::core::policy::DestinationOrder   destination_order_;
    ::dds::core::policy::History            history_;
    ::dds::core::policy::ResourceLimits     resource_limits_;
    ::dds::core::policy::Ownership          ownership_;
    ::dds::core::policy::TopicData          topic_data_;
};

//==============================================================================
//            PublicationBuiltinTopicDataDelegate
//==============================================================================

class org::eclipse::cyclonedds::topic::PublicationBuiltinTopicDataDelegate
{
public:
    PublicationBuiltinTopicDataDelegate() : ownership_strength_(0) { }

    const dds::topic::BuiltinTopicKey& key() const
    {
        return key_;
    }

    void key(const int32_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(key));
    }

    const dds::topic::BuiltinTopicKey& participant_key() const
    {
        return participant_key_;
    }

    void participant_key(const int32_t* key)
    {
        participant_key_.delegate().value(const_cast<int32_t*>(key));
    }

    const std::string&                  topic_name() const
    {
        return topic_name_;
    }

    void topic_name(const char *name)
    {
        topic_name_ = name;
    }

    const std::string&                  type_name() const
    {
        return type_name_;
    }

    void type_name(const char *name)
    {
        type_name_ = name;
    }

    const ::dds::core::policy::Durability&         durability() const
    {
        return durability_;
    }

    void durability(const dds_qos_t* policy)
    {
        durability_.delegate().set_iso_policy(policy);
    }

#ifdef  OMG_DDS_PERSISTENCE_SUPPORT

    const ::dds::core::policy::DurabilityService&  durability_service() const
    {
        return durability_service_;
    }

#endif  // OMG_DDS_PERSISTENCE_SUPPORT


    const ::dds::core::policy::Deadline&           deadline() const
    {
        return deadline_;
    }

    void deadline(const dds_qos_t* policy)
    {
        deadline_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::LatencyBudget&      latency_budget() const
    {
        return latency_budget_;
    }

    void latency_budget(const dds_qos_t* policy)
    {
        latency_budget_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Liveliness&         liveliness() const
    {
        return liveliness_;
    }

    void liveliness(const dds_qos_t* policy)
    {
        liveliness_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Reliability&        reliability() const
    {
        return reliability_;
    }

    void reliability(const dds_qos_t* policy)
    {
        reliability_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Lifespan&           lifespan() const
    {
        return lifespan_;
    }

    void lifespan(const dds_qos_t* policy)
    {
        lifespan_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::DestinationOrder&   destination_order() const
    {
        return destination_order_;
    }

    void destination_order(const dds_qos_t* policy)
    {
        destination_order_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Ownership&          ownership() const
    {
        return ownership_;
    }

    void ownership(const dds_qos_t* policy)
    {
        ownership_.delegate().set_iso_policy(policy);
    }

#ifdef  OMG_DDS_OWNERSHIP_SUPPORT

    const ::dds::core::policy::OwnershipStrength&  ownership_strength() const
    {
        return ownership_strength_;
    }

    void ownership_strength(const dds_qos_t* policy)
    {
        ownership_strength_.delegate().set_iso_policy(policy);
    }

#endif  // OMG_DDS_OWNERSHIP_SUPPORT


    const ::dds::core::policy::Partition&          partition() const
    {
        return partition_;
    }

    void partition(const dds_qos_t* policy)
    {
        partition_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Presentation&       presentation() const
    {
        return presentation_;
    }

    void presentation(const dds_qos_t* policy)
    {
        presentation_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::TopicData&          topic_data() const
    {
        return topic_data_;
    }

    void topic_data(const dds_qos_t* policy)
    {
        topic_data_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::UserData&           user_data() const
    {
        return user_data_;
    }

    void user_data(const dds_qos_t* policy)
    {
        user_data_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::GroupData&          group_data() const
    {
        return group_data_;
    }

    void group_data(const dds_qos_t* policy)
    {
        group_data_.delegate().set_iso_policy(policy);
    }


    bool operator ==(const PublicationBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_
               && other.participant_key_ == participant_key_
               && other.topic_name_ == topic_name_
               && other.type_name_ == type_name_
               && other.durability_ == durability_
#ifdef  OMG_DDS_PERSISTENCE_SUPPORT
               && other.durability_service_ == durability_service_
#endif  // OMG_DDS_PERSISTENCE_SUPPORT
               && other.deadline_ == deadline_
               && other.latency_budget_ == latency_budget_
               && other.liveliness_ == liveliness_
               && other.reliability_ == reliability_
               && other.lifespan_ == lifespan_
               && other.user_data_ == user_data_
               && other.ownership_ == ownership_
#ifdef  OMG_DDS_OWNERSHIP_SUPPORT
               && other.ownership_strength_ == ownership_strength_
#endif  // OMG_DDS_OWNERSHIP_SUPPORT
               && other.destination_order_ == destination_order_
               && other.topic_data_ == topic_data_
               && other.group_data_ == group_data_;
    }

public:
    dds::topic::BuiltinTopicKey  key_;
    dds::topic::BuiltinTopicKey  participant_key_;
    std::string                  topic_name_;
    std::string                  type_name_;
    ::dds::core::policy::Durability         durability_;

#ifdef  OMG_DDS_PERSISTENCE_SUPPORT
    ::dds::core::policy::DurabilityService  durability_service_;
#endif  // OMG_DDS_PERSISTENCE_SUPPORT

    ::dds::core::policy::Deadline           deadline_;
    ::dds::core::policy::LatencyBudget      latency_budget_;
    ::dds::core::policy::Liveliness         liveliness_;
    ::dds::core::policy::Reliability        reliability_;
    ::dds::core::policy::Lifespan           lifespan_;
    ::dds::core::policy::UserData           user_data_;
    ::dds::core::policy::Ownership          ownership_;

#ifdef  OMG_DDS_OWNERSHIP_SUPPORT
    ::dds::core::policy::OwnershipStrength  ownership_strength_;
#endif  // OMG_DDS_OWNERSHIP_SUPPORT

    ::dds::core::policy::DestinationOrder   destination_order_;
    ::dds::core::policy::Presentation       presentation_;
    ::dds::core::policy::Partition          partition_;
    ::dds::core::policy::TopicData          topic_data_;
    ::dds::core::policy::GroupData          group_data_;
};

//==============================================================================
//            SubscriptionBuiltinTopicDataDelegate
//==============================================================================

class org::eclipse::cyclonedds::topic::SubscriptionBuiltinTopicDataDelegate
{
public:
    const dds::topic::BuiltinTopicKey& key() const
    {
        return key_;
    }

    void key(const int32_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(key));
    }

    const dds::topic::BuiltinTopicKey& participant_key() const
    {
        return participant_key_;
    }

    void participant_key(const int32_t* key)
    {
        participant_key_.delegate().value(const_cast<int32_t*>(key));
    }

    const std::string&                  topic_name() const
    {
        return topic_name_;
    }

    void topic_name(const char *name)
    {
        topic_name_ = name;
    }

    const std::string&                  type_name() const
    {
        return type_name_;
    }

    void type_name(const char *name)
    {
        type_name_ = name;
    }

    const ::dds::core::policy::Durability&         durability() const
    {
        return durability_;
    }

    void durability(const dds_qos_t* policy)
    {
        durability_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Deadline&           deadline() const
    {
        return deadline_;
    }

    void deadline(const dds_qos_t* policy)
    {
        deadline_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::LatencyBudget&      latency_budget() const
    {
        return latency_budget_;
    }

    void latency_budget(const dds_qos_t* policy)
    {
        latency_budget_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Liveliness&         liveliness() const
    {
        return liveliness_;
    }

    void liveliness(const dds_qos_t* policy)
    {
        liveliness_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Reliability&        reliability() const
    {
        return reliability_;
    }

    void reliability(const dds_qos_t* policy)
    {
        reliability_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::DestinationOrder&   destination_order() const
    {
        return destination_order_;
    }

    void destination_order(const dds_qos_t* policy)
    {
        destination_order_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::TimeBasedFilter& time_based_filter() const
    {
        return time_based_filter_;
    }

    void time_based_filter(const dds_qos_t* policy)
    {
        time_based_filter_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Ownership&          ownership() const
    {
        return ownership_;
    }

    void ownership(const dds_qos_t* policy)
    {
        ownership_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::TopicData&          topic_data() const
    {
        return topic_data_;
    }

    void topic_data(const dds_qos_t* policy)
    {
        topic_data_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Partition&          partition() const
    {
        return partition_;
    }

    void partition(const dds_qos_t* policy)
    {
        partition_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Presentation&       presentation() const
    {
        return presentation_;
    }

    void presentation(const dds_qos_t* policy)
    {
        presentation_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::UserData&           user_data() const
    {
        return user_data_;
    }

    void user_data(const dds_qos_t* policy)
    {
        user_data_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::GroupData&          group_data() const
    {
        return group_data_;
    }

    void group_data(const dds_qos_t* policy)
    {
        group_data_.delegate().set_iso_policy(policy);
    }

    bool operator ==(const SubscriptionBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_
               && other.participant_key_ == participant_key_
               && other.topic_name_ == topic_name_
               && other.type_name_ == type_name_
               && other.durability_ == durability_
               && other.deadline_ == deadline_
               && other.latency_budget_ == latency_budget_
               && other.liveliness_ == liveliness_
               && other.reliability_ == reliability_
               && other.ownership_ == ownership_
               && other.destination_order_ == destination_order_
               && other.user_data_ == user_data_
               && other.time_based_filter_ == time_based_filter_
               && other.presentation_ == presentation_
               && other.partition_ == partition_
               && other.topic_data_ == topic_data_
               && other.group_data_ == group_data_;
    }

public:
    dds::topic::BuiltinTopicKey  key_;
    dds::topic::BuiltinTopicKey  participant_key_;
    std::string                  topic_name_;
    std::string                  type_name_;
    ::dds::core::policy::Durability         durability_;
    ::dds::core::policy::Deadline           deadline_;
    ::dds::core::policy::LatencyBudget      latency_budget_;
    ::dds::core::policy::Liveliness         liveliness_;
    ::dds::core::policy::Reliability        reliability_;
    ::dds::core::policy::Ownership          ownership_;
    ::dds::core::policy::DestinationOrder   destination_order_;
    ::dds::core::policy::UserData           user_data_;
    ::dds::core::policy::TimeBasedFilter    time_based_filter_;
    ::dds::core::policy::Presentation       presentation_;
    ::dds::core::policy::Partition          partition_;
    ::dds::core::policy::TopicData          topic_data_;
    ::dds::core::policy::GroupData          group_data_;
};

//==============================================================================
//            CMParticipantBuiltinTopicData
//==============================================================================

class org::eclipse::cyclonedds::topic::CMParticipantBuiltinTopicDataDelegate
{
public:
    const dds::topic::BuiltinTopicKey&        key() const
    {
        return key_;
    }

    void key(const int32_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(key));
    }

    bool operator ==(const CMParticipantBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_;
    }

protected:
    dds::topic::BuiltinTopicKey        key_;
};

//==============================================================================
//            CMPublisherBuiltinTopicDataDelegate
//==============================================================================

class org::eclipse::cyclonedds::topic::CMPublisherBuiltinTopicDataDelegate
{
public:
    const dds::topic::BuiltinTopicKey&        key() const
    {
        return key_;
    }

    void key(const int32_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(key));
    }

    const dds::topic::BuiltinTopicKey&        participant_key() const
    {
        return participant_key_;
    }

    void participant_key(const int32_t* key)
    {
        participant_key_.delegate().value(const_cast<int32_t*>(key));
    }

    const std::string&                        name() const
    {
        return name_;
    }

    void name(const char *name)
    {
        name_ = name;
    }

    const ::dds::core::policy::EntityFactory& entity_factory() const
    {
        return entity_factory_;
    }

    void entity_factory(const dds_qos_t* policy)
    {
        entity_factory_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Partition&     partition() const
    {
        return partition_;
    }

    void partition(const dds_qos_t* policy)
    {
        partition_.delegate().set_iso_policy(policy);
    }

    bool operator ==(const CMPublisherBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_
               && other.participant_key_ == participant_key_
               && other.name_ == name_
               && other.entity_factory_ == entity_factory_
               && other.partition_ == partition_;
    }

protected:
    dds::topic::BuiltinTopicKey        key_;
    dds::topic::BuiltinTopicKey        participant_key_;
    std::string                        name_;
    ::dds::core::policy::EntityFactory entity_factory_;
    ::dds::core::policy::Partition     partition_;
};

//==============================================================================
//            CMSubscriberBuiltinTopicDataDelegate
//==============================================================================

class org::eclipse::cyclonedds::topic::CMSubscriberBuiltinTopicDataDelegate
{
public:
    const dds::topic::BuiltinTopicKey&        key() const
    {
        return key_;
    }

    void key(const int32_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(key));
    }

    const dds::topic::BuiltinTopicKey&        participant_key() const
    {
        return participant_key_;
    }

    void participant_key(const int32_t* key)
    {
        participant_key_.delegate().value(const_cast<int32_t*>(key));
    }

    const std::string&                        name() const
    {
        return name_;
    }

    void name(const char *name)
    {
        name_ = name;
    }

    const ::dds::core::policy::EntityFactory& entity_factory() const
    {
        return entity_factory_;
    }

    void entity_factory(const dds_qos_t* policy)
    {
        entity_factory_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::Partition&     partition() const
    {
        return partition_;
    }

    void partition(const dds_qos_t* policy)
    {
        partition_.delegate().set_iso_policy(policy);
    }

    bool operator ==(const CMSubscriberBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_
               && other.participant_key_ == participant_key_
               && other.name_ == name_
               && other.entity_factory_ == entity_factory_
               && other.partition_ == partition_;
    }

protected:
    dds::topic::BuiltinTopicKey        key_;
    dds::topic::BuiltinTopicKey        participant_key_;
    std::string                        name_;
    ::dds::core::policy::EntityFactory entity_factory_;
    ::dds::core::policy::Partition     partition_;
};

//==============================================================================
//            CMDataWriterBuiltinTopicDataDelegate
//==============================================================================

class org::eclipse::cyclonedds::topic::CMDataWriterBuiltinTopicDataDelegate
{
public:
    const dds::topic::BuiltinTopicKey&              key() const
    {
        return key_;
    }

    void key(const dds_qos_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(reinterpret_cast<const int32_t*>(key)));
    }

    const dds::topic::BuiltinTopicKey&              publisher_key() const
    {
        return publisher_key_;
    }

    void publisher_key(const dds_qos_t* key)
    {
        publisher_key_.delegate().value(const_cast<int32_t*>(reinterpret_cast<const int32_t*>(key)));
    }

    const std::string&                              name() const
    {
        return name_;
    }

    void name(const char *name)
    {
        name_ = name;
    }

    const ::dds::core::policy::History&             history() const
    {
        return history_;
    }

    void history(const dds_qos_t* policy)
    {
        history_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::ResourceLimits&      resource_limits() const
    {
        return resource_limits_;
    }

    void resource_limits(const dds_qos_t* policy)
    {
        resource_limits_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::WriterDataLifecycle& writer_data_lifecycle() const
    {
        return writer_data_lifecycle_;
    }

    void writer_data_lifecycle(const dds_qos_t* policy)
    {
        writer_data_lifecycle_.delegate().set_iso_policy(policy);
    }

    bool operator ==(const CMDataWriterBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_
               && other.publisher_key_ == publisher_key_
               && other.name_ == name_
               && other.history_ == history_
               && other.resource_limits_ == resource_limits_
               && other.writer_data_lifecycle_ == writer_data_lifecycle_;
    }

protected:
    dds::topic::BuiltinTopicKey              key_;
    dds::topic::BuiltinTopicKey              publisher_key_;
    std::string                              name_;
    ::dds::core::policy::History             history_;
    ::dds::core::policy::ResourceLimits      resource_limits_;
    ::dds::core::policy::WriterDataLifecycle writer_data_lifecycle_;
};

//==============================================================================
//            CMDataReaderBuiltinTopicDataDelegate
//==============================================================================

class org::eclipse::cyclonedds::topic::CMDataReaderBuiltinTopicDataDelegate
{
public:
    const dds::topic::BuiltinTopicKey&              key() const
    {
        return key_;
    }

    void key(const int32_t* key)
    {
        key_.delegate().value(const_cast<int32_t*>(key));
    }

    const dds::topic::BuiltinTopicKey&              subscriber_key() const
    {
        return subscriber_key_;
    }

    void subscriber_key(const int32_t* key)
    {
        subscriber_key_.delegate().value(const_cast<int32_t*>(key));
    }

    const std::string&                              name() const
    {
        return name_;
    }

    void name(const char *name)
    {
        name_ = name;
    }

    const ::dds::core::policy::History&             history() const
    {
        return history_;
    }

    void history(const dds_qos_t* policy)
    {
        history_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::ResourceLimits&      resource_limits() const
    {
        return resource_limits_;
    }

    void resource_limits(const dds_qos_t* policy)
    {
        resource_limits_.delegate().set_iso_policy(policy);
    }

    const ::dds::core::policy::ReaderDataLifecycle& reader_data_lifecycle() const
    {
        return reader_data_lifecycle_;
    }

    void reader_data_lifecycle(const dds_qos_t* policy)
    {
        reader_data_lifecycle_.delegate().set_iso_policy(policy);
    }

    bool operator ==(const CMDataReaderBuiltinTopicDataDelegate& other) const
    {
        return other.key_ == key_
               && other.subscriber_key_ == subscriber_key_
               && other.name_ == name_
               && other.history_ == history_
               && other.resource_limits_ == resource_limits_
               && other.reader_data_lifecycle_ == reader_data_lifecycle_;
    }

protected:
    dds::topic::BuiltinTopicKey              key_;
    dds::topic::BuiltinTopicKey              subscriber_key_;
    std::string                              name_;
    ::dds::core::policy::History             history_;
    ::dds::core::policy::ResourceLimits      resource_limits_;
    ::dds::core::policy::ReaderDataLifecycle reader_data_lifecycle_;
};

#endif /* CYCLONEDDS_TOPIC_BUILTIN_TOPIC_DELEGATE_HPP */
