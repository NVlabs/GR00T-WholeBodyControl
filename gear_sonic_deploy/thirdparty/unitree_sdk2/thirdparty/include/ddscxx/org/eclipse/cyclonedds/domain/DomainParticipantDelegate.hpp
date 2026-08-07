/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
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

#ifndef CYCLONEDDS_DOMAIN_PARTICIPANT_DELEGATE_HPP_
#define CYCLONEDDS_DOMAIN_PARTICIPANT_DELEGATE_HPP_


// DDS-PSM-Cxx Includes
#include <dds/core/ref_traits.hpp>
#include <dds/core/Time.hpp>
#include <dds/core/InstanceHandle.hpp>
#include <dds/core/status/State.hpp>
#include <dds/core/detail/WeakReferenceImpl.hpp>
#include <dds/core/Entity.hpp>
#include <dds/domain/qos/DomainParticipantQos.hpp>
#include <dds/topic/qos/TopicQos.hpp>
#include <dds/pub/qos/PublisherQos.hpp>
#include <dds/sub/qos/SubscriberQos.hpp>

// Delegate Includes
#include <org/eclipse/cyclonedds/ForwardDeclarations.hpp>
#include <org/eclipse/cyclonedds/core/Mutex.hpp>
#include <org/eclipse/cyclonedds/core/EntityDelegate.hpp>
#include <org/eclipse/cyclonedds/core/ObjectSet.hpp>
#include <org/eclipse/cyclonedds/core/EntitySet.hpp>
#include "org/eclipse/cyclonedds/domain/Domain.hpp"
#include "org/eclipse/cyclonedds/domain/DomainWrap.hpp"

class OMG_DDS_API org::eclipse::cyclonedds::domain::DomainParticipantDelegate :
    public ::org::eclipse::cyclonedds::core::EntityDelegate
{
public:
    typedef ::dds::core::smart_ptr_traits< DomainParticipantDelegate >::ref_type ref_type;
    typedef ::dds::core::smart_ptr_traits< DomainParticipantDelegate >::weak_ref_type weak_ref_type;

    DomainParticipantDelegate(uint32_t id,
                              const dds::domain::qos::DomainParticipantQos& qos,
                              dds::domain::DomainParticipantListener *listener,
                              const dds::core::status::StatusMask& mask,
                              const std::string& config);

    DomainParticipantDelegate(uint32_t id,
                              const dds::domain::qos::DomainParticipantQos& qos,
                              dds::domain::DomainParticipantListener *listener,
                              const dds::core::status::StatusMask& mask,
                              const ddsi_config& config);

    virtual ~DomainParticipantDelegate();

public:
    void init(ObjectDelegate::weak_ref_type weak_ref);

    void listener(dds::domain::DomainParticipantListener *listener,
                  const ::dds::core::status::StatusMask& mask);
    dds::domain::DomainParticipantListener* listener() const;

    const dds::domain::qos::DomainParticipantQos& qos() const;

    void qos(const dds::domain::qos::DomainParticipantQos& qos);

    uint32_t domain_id();

    void assert_liveliness();

    bool contains_entity(const ::dds::core::InstanceHandle& handle);

    void close();

    dds::core::Time current_time() const;

    dds::topic::qos::TopicQos default_topic_qos() const;
    void default_topic_qos(const dds::topic::qos::TopicQos& qos);

    dds::pub::qos::PublisherQos default_publisher_qos() const;
    void default_publisher_qos(const ::dds::pub::qos::PublisherQos& qos);

    dds::sub::qos::SubscriberQos default_subscriber_qos() const;
    void default_subscriber_qos(const ::dds::sub::qos::SubscriberQos& qos);

    static dds::domain::qos::DomainParticipantQos default_participant_qos();
    static void default_participant_qos(const ::dds::domain::qos::DomainParticipantQos& qos);

    static void add_participant(org::eclipse::cyclonedds::core::EntityDelegate& participant);
    static void remove_participant(org::eclipse::cyclonedds::core::EntityDelegate& participant);

    void add_publisher(org::eclipse::cyclonedds::core::EntityDelegate& publisher);
    void remove_publisher(org::eclipse::cyclonedds::core::EntityDelegate& publisher);

    void add_subscriber(org::eclipse::cyclonedds::core::EntityDelegate& subscriber);
    void remove_subscriber(org::eclipse::cyclonedds::core::EntityDelegate& subscriber);

    void add_topic(org::eclipse::cyclonedds::core::EntityDelegate& topic);
    void remove_topic(org::eclipse::cyclonedds::core::EntityDelegate& topic);

    void add_cfTopic(org::eclipse::cyclonedds::core::ObjectDelegate& cfTopic);
    void remove_cfTopic(org::eclipse::cyclonedds::core::ObjectDelegate& cfTopic);

    org::eclipse::cyclonedds::core::EntityDelegate::ref_type
    find_topic(const std::string& topic_name);

    org::eclipse::cyclonedds::core::ObjectDelegate::ref_type
    find_cfTopic(const std::string& topic_name);

    dds_entity_t
    lookup_topic(const std::string& topic_name,
                 const dds_typeinfo_t *type_info,
                 const dds::core::Duration& timeout);

    void
    lookup_topics(const std::string& type_name,
                  std::vector<dds_entity_t>& topics,
                  uint32_t max_size);

    static org::eclipse::cyclonedds::domain::DomainParticipantDelegate::ref_type
    lookup_participant(uint32_t domain_id);

    dds::domain::TDomainParticipant<DomainParticipantDelegate>
    wrapper();

    bool is_auto_enable() const;

    void ignore_participant(const ::dds::core::InstanceHandle& handle);

    virtual void
    listener_notify(ObjectDelegate::ref_type source,
                    uint32_t       triggerMask,
                    void           *eventData,
                    void           *listener);

    org::eclipse::cyclonedds::core::EntityDelegate::ref_type
    builtin_subscriber();

    void
    builtin_subscriber(const org::eclipse::cyclonedds::core::EntityDelegate::ref_type subscriber);

    // Subscriber events
    virtual void on_data_readers(dds_entity_t subscriber);

    // Reader events
    void on_requested_deadline_missed(dds_entity_t reader,
        org::eclipse::cyclonedds::core::RequestedDeadlineMissedStatusDelegate &sd);
    void on_requested_incompatible_qos(dds_entity_t reader,
        org::eclipse::cyclonedds::core::RequestedIncompatibleQosStatusDelegate &sd);
    void on_sample_rejected(dds_entity_t reader,
        org::eclipse::cyclonedds::core::SampleRejectedStatusDelegate &sd);
    void on_liveliness_changed(dds_entity_t reader,
        org::eclipse::cyclonedds::core::LivelinessChangedStatusDelegate &sd);
    void on_data_available(dds_entity_t reader);
    void on_subscription_matched(dds_entity_t reader,
        org::eclipse::cyclonedds::core::SubscriptionMatchedStatusDelegate &sd);
    void on_sample_lost(dds_entity_t reader,
        org::eclipse::cyclonedds::core::SampleLostStatusDelegate &sd);


private:
    static org::eclipse::cyclonedds::core::EntitySet participants;
    static dds::domain::qos::DomainParticipantQos default_participant_qos_;
    static org::eclipse::cyclonedds::core::Mutex global_participants_lock_;
    static org::eclipse::cyclonedds::domain::DomainWrap::map_ref_type domain_registry_;
    uint32_t domain_id_;
    dds::domain::qos::DomainParticipantQos qos_;
    dds::topic::qos::TopicQos default_topic_qos_;
    dds::pub::qos::PublisherQos default_pub_qos_;
    dds::sub::qos::SubscriberQos default_sub_qos_;
    org::eclipse::cyclonedds::core::EntitySet publishers;
    org::eclipse::cyclonedds::core::EntitySet subscribers;
    org::eclipse::cyclonedds::core::EntitySet topics;
    org::eclipse::cyclonedds::core::ObjectSet cfTopics;
    org::eclipse::cyclonedds::core::EntityDelegate::weak_ref_type builtin_subscriber_;
    org::eclipse::cyclonedds::domain::DomainWrap::ref_type domain_ref_;
};

#endif /* CYCLONEDDS_DOMAIN_PARTICIPANT_DELEGATE_HPP_ */
