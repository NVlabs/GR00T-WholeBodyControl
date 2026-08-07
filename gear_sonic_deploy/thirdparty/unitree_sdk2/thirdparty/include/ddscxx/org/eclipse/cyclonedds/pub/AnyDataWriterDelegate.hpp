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

#ifndef CYCLONEDDS_PUB_ANYDATAWRITERDELEGATE_HPP_
#define CYCLONEDDS_PUB_ANYDATAWRITERDELEGATE_HPP_

#include <dds/core/types.hpp>
#include <dds/core/Time.hpp>
#include <dds/core/InstanceHandle.hpp>
#include <dds/core/status/Status.hpp>
#include <dds/pub/qos/DataWriterQos.hpp>
#include <org/eclipse/cyclonedds/topic/TopicTraits.hpp>
#include <org/eclipse/cyclonedds/core/EntityDelegate.hpp>
#include <dds/topic/TopicDescription.hpp>
#include <dds/topic/BuiltinTopic.hpp>

#include <org/eclipse/cyclonedds/topic/CDRBlob.hpp>

namespace dds { namespace pub {
template <typename DELEGATE>
class TAnyDataWriter;
} }


namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace pub
{

class OMG_DDS_API AnyDataWriterDelegate : public org::eclipse::cyclonedds::core::EntityDelegate
{
public:
    typedef ::dds::core::smart_ptr_traits< AnyDataWriterDelegate >::ref_type ref_type;
    typedef ::dds::core::smart_ptr_traits< AnyDataWriterDelegate >::weak_ref_type weak_ref_type;

    virtual ~AnyDataWriterDelegate();

    void close();

public:
    /* DDS API mirror. */
    dds::pub::qos::DataWriterQos qos() const;
    void qos(const dds::pub::qos::DataWriterQos& qos);

    /* Let DataWriter<T> implement the publisher handling to circumvent circular dependencies. */
    virtual const dds::pub::TPublisher<org::eclipse::cyclonedds::pub::PublisherDelegate>& publisher() const = 0;

    const dds::topic::TopicDescription& topic_description() const;

    void wait_for_acknowledgments(const dds::core::Duration& timeout);

    const ::dds::core::status::LivelinessLostStatus liveliness_lost_status();

    const ::dds::core::status::OfferedDeadlineMissedStatus offered_deadline_missed_status();

    const ::dds::core::status::OfferedIncompatibleQosStatus offered_incompatible_qos_status();

    const ::dds::core::status::PublicationMatchedStatus publication_matched_status();

    ::dds::core::InstanceHandleSeq
    matched_subscriptions();

    template <typename FwdIterator>
    uint32_t
    matched_subscriptions(FwdIterator begin, uint32_t max_size)
    {
        ::dds::core::InstanceHandleSeq handleSeq = matched_subscriptions();
        uint32_t seq_size = static_cast<uint32_t>(handleSeq.size());
        uint32_t size = (seq_size < max_size ? seq_size : max_size);
        for (uint32_t i = 0; i < size; i++, begin++) {
            *begin = handleSeq[i];
        }
        return size;
    }

    const dds::topic::SubscriptionBuiltinTopicData
    matched_subscription_data(const ::dds::core::InstanceHandle& h);

    void assert_liveliness();

public:
    dds::pub::TAnyDataWriter<AnyDataWriterDelegate> wrapper_to_any();
    void write_flush();
    void set_batch(bool);

private:
    void
    write_cdr(dds_entity_t writer,
          const org::eclipse::cyclonedds::topic::CDRBlob *data,
          const dds::core::InstanceHandle& handle,
          const dds::core::Time& timestamp,
          uint32_t statusinfo);

protected:
    AnyDataWriterDelegate(const dds::pub::qos::DataWriterQos& qos,
                          const dds::topic::TopicDescription& td);

    void
    write_cdr(dds_entity_t writer,
          const org::eclipse::cyclonedds::topic::CDRBlob *data,
          const dds::core::InstanceHandle& handle,
          const dds::core::Time& timestamp);

    void
    dispose_cdr(dds_entity_t writer,
          const org::eclipse::cyclonedds::topic::CDRBlob *data,
          const dds::core::InstanceHandle& handle,
          const dds::core::Time& timestamp);

    void
    unregister_instance_cdr(dds_entity_t writer,
          const org::eclipse::cyclonedds::topic::CDRBlob *data,
          const dds::core::InstanceHandle& handle,
          const dds::core::Time& timestamp);

    bool
    is_loan_supported(const dds_entity_t writer);

    void
    loan_sample(dds_entity_t writer,
          void **sample);

    void
    return_loan(dds_entity_t writer,
          void *sample);

    void
    write(dds_entity_t writer,
          const void *data,
          const dds::core::InstanceHandle& handle,
          const dds::core::Time& timestamp);

    void
    writedispose(dds_entity_t writer,
                 const void *data,
                 const dds::core::InstanceHandle& handle,
                 const dds::core::Time& timestamp);

    dds_instance_handle_t
    register_instance(dds_entity_t writer,
                      const void *data,
                      const dds::core::Time& timestamp);

    void
    unregister_instance(dds_entity_t writer,
                        const dds::core::InstanceHandle& handle,
                        const dds::core::Time& timestamp);

    void
    unregister_instance(dds_entity_t writer,
                        const void *data,
                        const dds::core::Time& timestamp);

    void
    dispose_instance(dds_entity_t writer,
                     const dds::core::InstanceHandle& handle,
                     const dds::core::Time& timestamp);

    void
    dispose_instance(dds_entity_t writer,
                     const void *data,
                     const dds::core::Time& timestamp);

    void
    get_key_value(dds_entity_t writer,
                  void *data,
                  const dds::core::InstanceHandle& handle);

    dds_instance_handle_t
    lookup_instance(dds_entity_t writer,
                    const void *data);

private:
    dds::pub::qos::DataWriterQos qos_;
    dds::topic::TopicDescription td_;

    //@todo static bool copy_data(c_type t, void *data, void *to);
};

}
}
}
}

#endif /* CYCLONEDDS_PUB_ANYDATAWRITERDELEGATE_HPP_ */
