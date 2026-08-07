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

#ifndef CYCLONEDDS_SUB_ANY_DATA_READER_DELEGATE_HPP_
#define CYCLONEDDS_SUB_ANY_DATA_READER_DELEGATE_HPP_

#include <dds/core/types.hpp>
#include <dds/core/Time.hpp>
#include <dds/core/InstanceHandle.hpp>
#include <dds/core/status/Status.hpp>
#include <dds/sub/status/detail/DataStateImpl.hpp>
#include <dds/sub/qos/DataReaderQos.hpp>
#include <dds/sub/Sample.hpp>
#include <dds/sub/SampleInfo.hpp>
#include <org/eclipse/cyclonedds/core/EntityDelegate.hpp>
#include <org/eclipse/cyclonedds/topic/TopicTraits.hpp>
#include <org/eclipse/cyclonedds/core/ObjectSet.hpp>
#include <org/eclipse/cyclonedds/ForwardDeclarations.hpp>
#include <dds/topic/TopicDescription.hpp>
#include <org/eclipse/cyclonedds/topic/CDRBlob.hpp>

#include <dds/topic/BuiltinTopic.hpp>


namespace dds { namespace sub {
template <typename DELEGATE>
class TAnyDataReader;
} }

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{
class QueryContainer;
}
}
}
}

namespace dds
{
namespace sub
{
namespace detail
{

class SamplesHolder
{
public:
    SamplesHolder() {}
    virtual ~SamplesHolder() {}

    virtual void set_length(uint32_t len) = 0;
    virtual uint32_t get_length() const = 0;
    virtual SamplesHolder& operator++(int) = 0;
    virtual void *data() = 0;
    virtual detail::SampleInfo& info() = 0;
    virtual void **cpp_sample_pointers(size_t length) = 0;
    virtual dds_sample_info_t *cpp_info_pointers(size_t length) = 0;
    virtual void set_sample_contents(void** c_sample_pointers, dds_sample_info_t *info) = 0;
    virtual void fini_samples_buffers(void**& c_sample_pointers, dds_sample_info_t*& c_sample_infos) = 0;
};

}
}
}



namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{

class QueryDelegate;

class OMG_DDS_API AnyDataReaderDelegate : public org::eclipse::cyclonedds::core::EntityDelegate
{
public:
    typedef ::dds::core::smart_ptr_traits< AnyDataReaderDelegate >::ref_type ref_type;
    typedef ::dds::core::smart_ptr_traits< AnyDataReaderDelegate >::weak_ref_type weak_ref_type;

    AnyDataReaderDelegate(const dds::sub::qos::DataReaderQos& qos,
                          const dds::topic::TopicDescription& td);
    virtual ~AnyDataReaderDelegate();

    static void copy_sample_infos(
        const dds_sample_info_t &from,
        dds::sub::SampleInfo &to);

public:
    /* DDS API mirror. */
    dds::sub::qos::DataReaderQos qos() const;
    void qos(const dds::sub::qos::DataReaderQos& qos);

    /* Let DataReader<T> implement the subscriber handling to circumvent circular dependencies. */
    virtual const dds::sub::TSubscriber<org::eclipse::cyclonedds::sub::SubscriberDelegate>& subscriber() const = 0;
    const dds::topic::TopicDescription& topic_description() const;

    void wait_for_historical_data(const dds::core::Duration& timeout);

    dds::core::status::LivelinessChangedStatus
    liveliness_changed_status();
    dds::core::status::SampleRejectedStatus
    sample_rejected_status();
    dds::core::status::SampleLostStatus
    sample_lost_status();
    dds::core::status::RequestedDeadlineMissedStatus
    requested_deadline_missed_status();
    dds::core::status::RequestedIncompatibleQosStatus
    requested_incompatible_qos_status();
    dds::core::status::SubscriptionMatchedStatus
    subscription_matched_status();

    ::dds::core::InstanceHandleSeq
     matched_publications();

    template <typename FwdIterator>
    uint32_t
    matched_publications(FwdIterator begin, uint32_t max_size)
    {
        ::dds::core::InstanceHandleSeq handleSeq = matched_publications();
        uint32_t seq_size = static_cast<uint32_t>(handleSeq.size());
        uint32_t size = (seq_size < max_size ? seq_size : max_size);
        for (uint32_t i = 0; i < size; i++, begin++) {
            *begin = handleSeq[i];
        }
        return size;
    }

    const dds::topic::PublicationBuiltinTopicData
    matched_publication_data(const ::dds::core::InstanceHandle& h);

public:
    /* Internal API. */
    dds::sub::TAnyDataReader<AnyDataReaderDelegate> wrapper_to_any();

    static uint32_t get_ddsc_state_mask(const dds::sub::status::DataState& state);

    void reset_data_available();

    void add_query(org::eclipse::cyclonedds::sub::QueryDelegate& query);
    void remove_query(org::eclipse::cyclonedds::sub::QueryDelegate& query);

    void setSample(void* sample);
    void* getSample() const;

    bool is_loan_supported(const dds_entity_t reader) const;

    void read_cdr(
            const dds_entity_t reader,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void take_cdr(
            const dds_entity_t reader,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void loaned_read(
            const dds_entity_t reader,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void loaned_take(
            const dds_entity_t reader,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void loaned_read_instance(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void loaned_take_instance(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void loaned_read_next_instance(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void loaned_take_next_instance(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void read(
            const dds_entity_t reader,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void take(
            const dds_entity_t reader,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void read_instance(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void take_instance(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void read_next_instance(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void take_next_instance(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            const dds::sub::status::DataState& mask,
            dds::sub::detail::SamplesHolder& samples,
            uint32_t max_samples);

    void get_key_value(
            const dds_entity_t reader,
            const dds::core::InstanceHandle& handle,
            void *key);

    dds_instance_handle_t lookup_instance(
            const dds_entity_t reader,
            const void *key) const;

    void close();

private:
    bool init_samples_buffers(
            const uint32_t                    requested_max_samples,
            uint32_t&                         samples_to_read_cnt,
            size_t&                           c_sample_pointers_size,
            dds::sub::detail::SamplesHolder&  samples,
            void**&                           c_sample_pointers,
            dds_sample_info_t*&               c_sample_infos);

    void fini_samples_buffers(
            void**& c_sample_pointers,
            dds_sample_info_t*& c_sample_infos);

protected:
    org::eclipse::cyclonedds::core::ObjectSet queries;
    dds::sub::qos::DataReaderQos qos_;
    dds::topic::TopicDescription td_;

    void *sample_;

};


}
}
}
}

#endif /* CYCLONEDDS_SUB_ANY_DATA_READER_DELEGATE_HPP_ */
