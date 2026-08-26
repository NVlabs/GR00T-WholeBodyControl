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
#ifndef OMG_DDS_SUB_DETAIL_DATA_READER_HPP_
#define OMG_DDS_SUB_DETAIL_DATA_READER_HPP_

#include <dds/topic/Topic.hpp>
#include <dds/topic/TopicInstance.hpp>

#include <dds/core/status/Status.hpp>
#include <dds/sub/status/detail/DataStateImpl.hpp>
#include <dds/sub/detail/Manipulators.hpp>
#include <dds/sub/LoanedSamples.hpp>
#include <dds/sub/Subscriber.hpp>
#include <dds/sub/Query.hpp>

#include <org/eclipse/cyclonedds/core/EntityDelegate.hpp>
#include <org/eclipse/cyclonedds/sub/AnyDataReaderDelegate.hpp>

#include <org/eclipse/cyclonedds/core/ScopedLock.hpp>
#include <org/eclipse/cyclonedds/ForwardDeclarations.hpp>

#include <dds/dds.h>

/***************************************************************************
 *
 * dds/sub/detail/DataReader<> DELEGATE declaration.
 * Implementation can be found in dds/sub/detail/TDataReaderImpl.hpp
 *
 ***************************************************************************/
template <typename T>
class dds::sub::detail::DataReader : public ::org::eclipse::cyclonedds::sub::AnyDataReaderDelegate
{
public:

    typedef typename ::dds::core::smart_ptr_traits< DataReader<T> >::ref_type ref_type;
    typedef typename ::dds::core::smart_ptr_traits< DataReader<T> >::weak_ref_type weak_ref_type;

    DataReader(const dds::sub::Subscriber& sub,
               const dds::topic::Topic<T>& topic,
               const dds::sub::qos::DataReaderQos& qos,
               dds::sub::DataReaderListener<T>* listener = NULL,
               const dds::core::status::StatusMask& mask = ::dds::core::status::StatusMask::none());

    DataReader(const dds::sub::Subscriber& sub,
               const dds::topic::ContentFilteredTopic<T, dds::topic::detail::ContentFilteredTopic>& topic,
               const dds::sub::qos::DataReaderQos& qos,
               dds::sub::DataReaderListener<T>* listener = NULL,
               const dds::core::status::StatusMask& mask = ::dds::core::status::StatusMask::none());

    void common_constructor(dds::sub::DataReaderListener<T>* listener,
                            const dds::core::status::StatusMask& mask);

    virtual ~DataReader();

    void copy_samples(
      dds::sub::detail::SamplesHolder& samples,
      void**& c_sample_pointers,
      dds_sample_info_t*& c_sample_infos,
      int num_read);

    void init(ObjectDelegate::weak_ref_type weak_ref);

    dds::sub::status::DataState default_filter_state();
    void default_filter_state(const dds::sub::status::DataState& state);

    bool is_loan_supported();

    dds::sub::LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob> read_cdr();
    dds::sub::LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob> take_cdr();

    dds::sub::LoanedSamples<T> read();
    dds::sub::LoanedSamples<T> take();

    template<typename SamplesFWIterator>
    uint32_t read(SamplesFWIterator samples, uint32_t max_samples);
    template<typename SamplesFWIterator>
    uint32_t take(SamplesFWIterator samples, uint32_t max_samples);

    template<typename SamplesBIIterator>
    uint32_t read(SamplesBIIterator samples);
    template<typename SamplesBIIterator>
    uint32_t take(SamplesBIIterator samples);

    dds::topic::TopicInstance<T> key_value(const dds::core::InstanceHandle& h);
    T& key_value(T& key, const dds::core::InstanceHandle& h);

    const dds::core::InstanceHandle lookup_instance(const T& key) const;

    virtual const dds::sub::Subscriber& subscriber() const;

    void close();

    dds::sub::DataReaderListener<T>* listener();
    void listener(dds::sub::DataReaderListener<T>* l,
                  const dds::core::status::StatusMask& event_mask);

    dds::sub::DataReader<T, dds::sub::detail::DataReader> wrapper();

    void on_requested_deadline_missed(dds_entity_t,
          org::eclipse::cyclonedds::core::RequestedDeadlineMissedStatusDelegate &);

    void on_requested_incompatible_qos(dds_entity_t,
          org::eclipse::cyclonedds::core::RequestedIncompatibleQosStatusDelegate &);

    void on_sample_rejected(dds_entity_t,
          org::eclipse::cyclonedds::core::SampleRejectedStatusDelegate &);

    void on_liveliness_changed(dds_entity_t,
          org::eclipse::cyclonedds::core::LivelinessChangedStatusDelegate &);

    void on_data_available(dds_entity_t);

    void on_subscription_matched(dds_entity_t,
          org::eclipse::cyclonedds::core::SubscriptionMatchedStatusDelegate &);

    void on_sample_lost(dds_entity_t,
          org::eclipse::cyclonedds::core::SampleLostStatusDelegate &);

private:
    dds::sub::Subscriber sub_;
    dds::sub::status::DataState status_filter_;


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
private:
    enum SelectMode {
        SELECT_MODE_READ,
        SELECT_MODE_READ_INSTANCE,
        SELECT_MODE_READ_NEXT_INSTANCE,
        SELECT_MODE_READ_WITH_CONDITION,
        SELECT_MODE_READ_INSTANCE_WITH_CONDITION,
        SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION
    };


public:

    class Selector
    {
    public:
        Selector(typename DataReader<T>::ref_type dr);

        Selector& instance(const dds::core::InstanceHandle& h);
        Selector& next_instance(const dds::core::InstanceHandle& h);
        Selector& filter_state(const dds::sub::status::DataState& s);
        Selector& max_samples(uint32_t n);
        Selector& filter_content(const dds::sub::Query& query);

        dds::sub::LoanedSamples<T> read();
        dds::sub::LoanedSamples<T> take();

        // --- Forward Iterators: --- //
        template<typename SamplesFWIterator>
        uint32_t read(SamplesFWIterator sfit, uint32_t max_samples);
        template<typename SamplesFWIterator>
        uint32_t take(SamplesFWIterator sfit, uint32_t max_samples);

        // --- Back-Inserting Iterators: --- //
        template<typename SamplesBIIterator>
        uint32_t read(SamplesBIIterator sbit);
        template<typename SamplesBIIterator>
        uint32_t take(SamplesBIIterator sbit);

        SelectMode get_mode() const;

    private:
        friend class DataReader;
        SelectMode mode;
        typename DataReader<T>::ref_type reader;
        dds::sub::status::DataState state_filter_;
        bool state_filter_is_set_;
        dds::core::InstanceHandle handle;
        uint32_t max_samples_;
        dds::sub::Query query_;
    };


    class ManipulatorSelector: public Selector
    {
    public:
        //ManipulatorSelector(DataReader<T>* dr);
        ManipulatorSelector(typename DataReader<T>::ref_type dr);

        bool read_mode();
        void read_mode(bool b);

        ManipulatorSelector&
        operator >>(dds::sub::LoanedSamples<T>& samples);

    private:
        bool read_mode_;
    };


private:
    // ==============================================================
    // == Selector Read/Take API

    dds::sub::LoanedSamples<T> read(const Selector& selector);

    dds::sub::LoanedSamples<T> take(const Selector& selector);

    // --- Forward Iterators: --- //
    template<typename SamplesFWIterator>
    uint32_t read(SamplesFWIterator samples,
                  uint32_t max_samples, const Selector& selector);

    template<typename SamplesFWIterator>
    uint32_t take(SamplesFWIterator samples,
                  uint32_t max_samples, const Selector& selector);

    // --- Back-Inserting Iterators: --- //
    template<typename SamplesBIIterator>
    uint32_t read(SamplesBIIterator samples, const Selector& selector);

    template<typename SamplesBIIterator>
    uint32_t take(SamplesBIIterator samples, const Selector& selector);

 private:
    T typed_sample_;

};


#endif /* OMG_TDDS_SUB_DETAIL_DATA_READER_HPP_ */
