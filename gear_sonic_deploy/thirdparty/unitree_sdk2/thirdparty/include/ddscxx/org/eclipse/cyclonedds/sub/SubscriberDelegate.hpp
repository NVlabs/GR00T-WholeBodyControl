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

#ifndef CYCLONEDDS_SUB_SUBSCRIBER_DELEGATE_HPP_
#define CYCLONEDDS_SUB_SUBSCRIBER_DELEGATE_HPP_

#include <dds/core/types.hpp>
#include <dds/core/status/State.hpp>
#include <dds/sub/AnyDataReader.hpp>
#include <dds/sub/qos/SubscriberQos.hpp>
#include <dds/sub/qos/DataReaderQos.hpp>
#include <dds/domain/DomainParticipant.hpp>

#include <org/eclipse/cyclonedds/ForwardDeclarations.hpp>
#include <org/eclipse/cyclonedds/core/EntityDelegate.hpp>
#include <org/eclipse/cyclonedds/core/EntitySet.hpp>
#include <org/eclipse/cyclonedds/sub/AnyDataReaderDelegate.hpp>

#include <vector>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{

class OMG_DDS_API SubscriberDelegate : public org::eclipse::cyclonedds::core::EntityDelegate
{
public:
    typedef ::dds::core::smart_ptr_traits< SubscriberDelegate >::ref_type ref_type;
    typedef ::dds::core::smart_ptr_traits< SubscriberDelegate >::weak_ref_type weak_ref_type;

    SubscriberDelegate(const dds::domain::DomainParticipant& dp,
                       const dds::sub::qos::SubscriberQos& qos,
                       dds::sub::SubscriberListener* listener,
                       const dds::core::status::StatusMask& event_mask);

    virtual ~SubscriberDelegate();

    void init(ObjectDelegate::weak_ref_type weak_ref);

    void close();

    const dds::sub::qos::SubscriberQos& qos() const;
    void qos(const dds::sub::qos::SubscriberQos& sqos);

    void default_datareader_qos(const dds::sub::qos::DataReaderQos& qos);
    dds::sub::qos::DataReaderQos default_datareader_qos() const;

    void begin_coherent_access();
    void end_coherent_access();

    /**
     *  @internal This function initialises the delegate as the built in subscriber
     */
    //void init_builtin(DDS::Subscriber_ptr sub);

    void listener(dds::sub::SubscriberListener* listener,
                  const ::dds::core::status::StatusMask& mask);
    dds::sub::SubscriberListener* listener() const;

    const dds::domain::DomainParticipant& participant() const;

    /** @internal @todo OSPL-1944 Subscriber Listener should return list of affected DataReaders (on_data_on_readers) **/
    //dds::sub::AnyDataReader get_datareaders(); /* TODO: OSPL-1944? */

    bool contains_entity(
            const ::dds::core::InstanceHandle& handle);

    void add_datareader(
            org::eclipse::cyclonedds::core::EntityDelegate& datareader);

    void remove_datareader(
            org::eclipse::cyclonedds::core::EntityDelegate& datareader);

    virtual std::vector<org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::ref_type>
    find_datareaders(
            const std::string& topic_name);

    std::vector<org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::ref_type>
    get_datareaders(
            const dds::sub::status::DataState& mask);

    void notify_datareaders();

    dds::sub::TSubscriber<SubscriberDelegate>
    wrapper();

    bool is_auto_enable() const;

    void reset_data_on_readers();

    // Subscriber events
    void on_data_readers(
            dds_entity_t);

    // Reader events
    void on_requested_deadline_missed(
            dds_entity_t reader,
            org::eclipse::cyclonedds::core::RequestedDeadlineMissedStatusDelegate &sd);
    void on_requested_incompatible_qos(
            dds_entity_t reader,
            org::eclipse::cyclonedds::core::RequestedIncompatibleQosStatusDelegate &sd);
    void on_sample_rejected(
            dds_entity_t reader,
            org::eclipse::cyclonedds::core::SampleRejectedStatusDelegate &sd);
    void on_liveliness_changed(
            dds_entity_t reader,
            org::eclipse::cyclonedds::core::LivelinessChangedStatusDelegate &sd);
    void on_data_available(
            dds_entity_t reader);
    void on_subscription_matched(
            dds_entity_t reader,
            org::eclipse::cyclonedds::core::SubscriptionMatchedStatusDelegate &sd);
    void on_sample_lost(
            dds_entity_t reader,
            org::eclipse::cyclonedds::core::SampleLostStatusDelegate &sd);

private:
    dds::domain::DomainParticipant dp_;
    dds::sub::qos::SubscriberQos qos_;
    dds::sub::qos::DataReaderQos default_dr_qos_;

    org::eclipse::cyclonedds::core::EntitySet readers;
};

}
}
}
}

#endif /* CYCLONEDDS_SUB_SUBSCRIBER_DELEGATE_HPP_ */
