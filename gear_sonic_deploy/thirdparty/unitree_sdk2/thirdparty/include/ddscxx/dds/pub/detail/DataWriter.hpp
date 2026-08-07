#ifndef OMG_DDS_PUB_DETAIL_DATA_WRITER_HPP_
#define OMG_DDS_PUB_DETAIL_DATA_WRITER_HPP_

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

#include <dds/topic/Topic.hpp>
#include <dds/pub/AnyDataWriter.hpp>
#include <dds/topic/detail/Topic.hpp>
#include <org/eclipse/cyclonedds/core/EntityDelegate.hpp>
#include <org/eclipse/cyclonedds/topic/TopicTraits.hpp>
#include <org/eclipse/cyclonedds/core/ScopedLock.hpp>
#include <org/eclipse/cyclonedds/pub/AnyDataWriterDelegate.hpp>
#include <dds/dds.h>

namespace dds {
    namespace pub {

        template <typename T>
        class DataWriterListener;

        namespace detail {
            template <typename T>
            class DataWriter;
        }

        template <typename T, template <typename Q> class DELEGATE>
        class DataWriter;
    }
}



/***************************************************************************
 *
 * dds/pub/detail/DataWriter<> DELEGATE declaration.
 * Implementation can be found in dds/pub/detail/DataWriterImpl.hpp
 *
 ***************************************************************************/
template <typename T>
class dds::pub::detail::DataWriter : public ::org::eclipse::cyclonedds::pub::AnyDataWriterDelegate  {
public:

    typedef typename ::dds::core::smart_ptr_traits< DataWriter<T> >::ref_type ref_type;
    typedef typename ::dds::core::smart_ptr_traits< DataWriter<T> >::weak_ref_type weak_ref_type;

    DataWriter(const dds::pub::Publisher& pub,
               const ::dds::topic::Topic<T>& topic,
               const dds::pub::qos::DataWriterQos& qos,
               dds::pub::DataWriterListener<T>* listener,
               const dds::core::status::StatusMask& mask);

    virtual ~DataWriter();

    void init(ObjectDelegate::weak_ref_type weak_ref);

    bool is_loan_supported();

    T& loan_sample();

    void return_loan(T& sample);

    void write_cdr(const org::eclipse::cyclonedds::topic::CDRBlob& sample);

    void write_cdr(const org::eclipse::cyclonedds::topic::CDRBlob& sample, const dds::core::Time& timestamp);

    void dispose_cdr(const org::eclipse::cyclonedds::topic::CDRBlob& sample);

    void dispose_cdr(const org::eclipse::cyclonedds::topic::CDRBlob& sample, const dds::core::Time& timestamp);

    void unregister_instance_cdr(const org::eclipse::cyclonedds::topic::CDRBlob& sample);

    void unregister_instance_cdr(const org::eclipse::cyclonedds::topic::CDRBlob& sample, const dds::core::Time& timestamp);

    void write(const T& sample);

    void write(const T& sample, const dds::core::Time& timestamp);

    void write(const T& sample, const ::dds::core::InstanceHandle& instance);

    void write(const T& sample,
               const ::dds::core::InstanceHandle& instance,
               const dds::core::Time& timestamp);

    void write(const dds::topic::TopicInstance<T>& i);

    void write(const dds::topic::TopicInstance<T>& i,
               const dds::core::Time& timestamp);

    void writedispose(const T& sample);

    void writedispose(const T& sample, const dds::core::Time& timestamp);

    void writedispose(const T& sample, const ::dds::core::InstanceHandle& instance);

    void writedispose(const T& sample,
            const ::dds::core::InstanceHandle& instance,
            const dds::core::Time& timestamp);

    void writedispose(const dds::topic::TopicInstance<T>& i);

    void writedispose(const dds::topic::TopicInstance<T>& i,
                      const dds::core::Time& timestamp);

    template <typename FWIterator>
    void writedispose(const FWIterator& begin, const FWIterator& end);

    template <typename FWIterator>
    void writedispose(const FWIterator& begin, const FWIterator& end,
                      const dds::core::Time& timestamp);

    template <typename SamplesFWIterator, typename HandlesFWIterator>
    void writedispose(const SamplesFWIterator& data_begin,
                      const SamplesFWIterator& data_end,
                      const HandlesFWIterator& handle_begin,
                      const HandlesFWIterator& handle_end);

    template <typename SamplesFWIterator, typename HandlesFWIterator>
    void writedispose(const SamplesFWIterator& data_begin,
                      const SamplesFWIterator& data_end,
                      const HandlesFWIterator& handle_begin,
                      const HandlesFWIterator& handle_end,
                      const dds::core::Time& timestamp);

    const ::dds::core::InstanceHandle register_instance(const T& key,
                                                        const dds::core::Time& timestamp);

    void unregister_instance(const ::dds::core::InstanceHandle& handle,
                             const dds::core::Time& timestamp);

    void unregister_instance(const T& sample,
                             const dds::core::Time& timestamp);

    void dispose_instance(const ::dds::core::InstanceHandle& handle,
                          const dds::core::Time& timestamp);

    void dispose_instance(const T& sample,
                          const dds::core::Time& timestamp);

    dds::topic::TopicInstance<T>& key_value(dds::topic::TopicInstance<T>& i,
                                            const ::dds::core::InstanceHandle& h);

    T& key_value(T& sample, const ::dds::core::InstanceHandle& h);

    dds::core::InstanceHandle lookup_instance(const T& key);

    const dds::topic::Topic<T>& topic() const;

    virtual const dds::pub::Publisher& publisher() const;

    void listener(DataWriterListener<T>* listener,
                  const ::dds::core::status::StatusMask& mask);

    DataWriterListener<T>* listener() const;

    virtual void close();

    dds::pub::DataWriter<T, dds::pub::detail::DataWriter> wrapper();

    void on_offered_deadline_missed(dds_entity_t,
          org::eclipse::cyclonedds::core::OfferedDeadlineMissedStatusDelegate &sd);

    void on_offered_incompatible_qos(dds_entity_t,
          org::eclipse::cyclonedds::core::OfferedIncompatibleQosStatusDelegate &sd);

    void on_liveliness_lost(dds_entity_t,
          org::eclipse::cyclonedds::core::LivelinessLostStatusDelegate &sd);

    void on_publication_matched(dds_entity_t,
          org::eclipse::cyclonedds::core::PublicationMatchedStatusDelegate &sd);

private:
   dds::pub::Publisher                    pub_;
   dds::topic::Topic<T>                   topic_;
};




#endif /* OMG_DDS_PUB_DETAIL_DATA_WRITER_HPP_ */
