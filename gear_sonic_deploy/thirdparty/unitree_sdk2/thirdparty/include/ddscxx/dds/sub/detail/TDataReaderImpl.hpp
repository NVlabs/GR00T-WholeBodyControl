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
#ifndef CYCLONEDDS_DDS_SUB_TDATAREADER_IMPL_HPP_
#define CYCLONEDDS_DDS_SUB_TDATAREADER_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/detail/DataReader.hpp>
#include <dds/sub/Query.hpp>
#include <dds/sub/detail/SamplesHolder.hpp>
#include <dds/domain/DomainParticipantListener.hpp>
#include "dds/core/macros.hpp"




/***************************************************************************
 *
 * dds/sub/DataReader<> WRAPPER implementation.
 * Declaration can be found in dds/sub/TDataReader.hpp
 *
 ***************************************************************************/

// Implementation

namespace dds
{
namespace sub
{

//--------------------------------------------------------------------------------
//  DATAREADER
//--------------------------------------------------------------------------------

template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>::Selector::Selector(DataReader& dr) : impl_(dr.delegate())
{
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::Selector&
DataReader<T, DELEGATE>::Selector::instance(const dds::core::InstanceHandle& h)
{
    impl_.instance(h);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::Selector&
DataReader<T, DELEGATE>::Selector::next_instance(const dds::core::InstanceHandle& h)
{
    impl_.next_instance(h);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::Selector&
DataReader<T, DELEGATE>::Selector::state(const dds::sub::status::DataState& s)
{
    impl_.filter_state(s);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::Selector&
DataReader<T, DELEGATE>::Selector::content(const dds::sub::Query& query)
{
    impl_.filter_content(query);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::Selector&
DataReader<T, DELEGATE>::Selector::max_samples(uint32_t n)
{
    impl_.max_samples(n);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
dds::sub::LoanedSamples<T>
DataReader<T, DELEGATE>::Selector::read()
{
    return impl_.read();
}

template <typename T, template <typename Q> class DELEGATE>
dds::sub::LoanedSamples<T>
DataReader<T, DELEGATE>::Selector::take()
{
    return impl_.take();
}

template <typename T, template <typename Q> class DELEGATE>
template <typename SamplesFWIterator>
uint32_t
DataReader<T, DELEGATE>::Selector::read(SamplesFWIterator sfit, uint32_t max_samples)
{
    return impl_.read(sfit, max_samples);
}

template <typename T, template <typename Q> class DELEGATE>
template <typename SamplesFWIterator>
uint32_t
DataReader<T, DELEGATE>::Selector::take(SamplesFWIterator sfit,    uint32_t max_samples)
{
    return impl_.take(sfit, max_samples);
}

template <typename T, template <typename Q> class DELEGATE>
template <typename SamplesBIIterator>
uint32_t
DataReader<T, DELEGATE>::Selector::read(SamplesBIIterator sbit)
{
    return impl_.read(sbit);
}

template <typename T, template <typename Q> class DELEGATE>
template <typename SamplesBIIterator>
uint32_t
DataReader<T, DELEGATE>::Selector::take(SamplesBIIterator sbit)
{
    return impl_.take(sbit);
}

//--------------------------------------------------------------------------------
//  DATAREADER::MANIPULATORSELECTOR
//--------------------------------------------------------------------------------
template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>::ManipulatorSelector::
ManipulatorSelector(DataReader& dr) : impl_(dr.delegate()) {}

template <typename T, template <typename Q> class DELEGATE>
bool
DataReader<T, DELEGATE>::ManipulatorSelector::read_mode()
{
    return impl_.read_mode();
}

template <typename T, template <typename Q> class DELEGATE>
void
DataReader<T, DELEGATE>::ManipulatorSelector::read_mode(bool b)
{
    impl_.read_mode(b);
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::ManipulatorSelector&
DataReader<T, DELEGATE>::ManipulatorSelector::instance(const dds::core::InstanceHandle& h)
{
    impl_.instance(h);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::ManipulatorSelector&
DataReader<T, DELEGATE>::ManipulatorSelector::next_instance(const dds::core::InstanceHandle& h)
{
    impl_.next_instance(h);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::ManipulatorSelector&
DataReader<T, DELEGATE>::ManipulatorSelector::operator >>(dds::sub::LoanedSamples<T>& samples)
{
    impl_ >> samples;
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::ManipulatorSelector&
DataReader<T, DELEGATE>::ManipulatorSelector::operator >> (ManipulatorSelector & (manipulator)(ManipulatorSelector&))
{
    manipulator(*this);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
template <typename Functor>
typename DataReader<T, DELEGATE>::ManipulatorSelector
DataReader<T, DELEGATE>::ManipulatorSelector::operator >> (Functor f)
{
    f(*this);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::ManipulatorSelector&
DataReader<T, DELEGATE>::ManipulatorSelector::state(const dds::sub::status::DataState& s)
{
    impl_.filter_state(s);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::ManipulatorSelector&
DataReader<T, DELEGATE>::ManipulatorSelector::content(const dds::sub::Query& query)
{
    impl_.filter_content(query);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::ManipulatorSelector&
DataReader<T, DELEGATE>::ManipulatorSelector::max_samples(uint32_t n)
{
    impl_.max_samples(n);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>::DataReader(
    const dds::sub::Subscriber& sub,
    const dds::topic::Topic<T>& topic):
        ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(sub, topic, sub->default_datareader_qos()))
{
    this->delegate()->init(this->impl_);
}

template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>::DataReader(
    const dds::sub::Subscriber& sub,
    const ::dds::topic::Topic<T>& topic,
    const dds::sub::qos::DataReaderQos& qos,
    dds::sub::DataReaderListener<T>* listener,
    const dds::core::status::StatusMask& mask) :
        ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(sub, topic, qos, listener, mask))
{
    this->delegate()->init(this->impl_);
}

#ifdef OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT
template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>::DataReader(
    const dds::sub::Subscriber& sub,
    const dds::topic::ContentFilteredTopic<T>& topic) :
        ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(sub, topic, sub.default_datareader_qos()))
{
    this->delegate()->init(this->impl_);
}

template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>::DataReader(
    const dds::sub::Subscriber& sub,
    const ::dds::topic::ContentFilteredTopic<T>& topic,
    const dds::sub::qos::DataReaderQos& qos,
    dds::sub::DataReaderListener<T>* listener,
    const dds::core::status::StatusMask& mask) :
    ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(sub, topic, qos, listener, mask))
{
    this->delegate()->init(this->impl_);
}
#endif /* OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT */

#ifdef OMG_DDS_MULTI_TOPIC_SUPPORT
template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>::DataReader(
    const dds::sub::Subscriber& sub,
    const dds::topic::MultiTopic<T>& topic) :
        ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(sub, topic))
{
    this->delegate()->init(this->impl_);
}

template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>::DataReader(
    const dds::sub::Subscriber& sub,
    const ::dds::topic::MultiTopic<T>& topic,
    const dds::sub::qos::DataReaderQos& qos,
    dds::sub::DataReaderListener<T>* listener,
    const dds::core::status::StatusMask& mask) :
       ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(sub, topic, qos, listener, mask))
{
    this->delegate()->init(this->impl_);
}
#endif /* OMG_DDS_MULTI_TOPIC_SUPPORT */

template <typename T, template <typename Q> class DELEGATE>
dds::sub::status::DataState
DataReader<T, DELEGATE>::default_filter_state()
{
    return this->delegate()->default_filter_state();
}

template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>& DataReader<T, DELEGATE>::default_filter_state(const dds::sub::status::DataState& status)
{
    this->delegate()->default_filter_state(status);
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
DataReader<T, DELEGATE>& DataReader<T, DELEGATE>::operator >>(dds::sub::LoanedSamples<T>& ls)
{
    ls = this->read();
    return *this;
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::ManipulatorSelector
DataReader<T, DELEGATE>::operator >> (ManipulatorSelector& (manipulator)(ManipulatorSelector&))
{
    ManipulatorSelector selector(*this);
    manipulator(selector);
    return selector;
}

template <typename T, template <typename Q> class DELEGATE>
template <typename Functor>
typename DataReader<T, DELEGATE>::ManipulatorSelector
DataReader<T, DELEGATE>::operator >> (Functor f)
{
    ManipulatorSelector selector(*this);
    f(selector);
    return selector;
}

template <typename T, template <typename Q> class DELEGATE>
LoanedSamples<T>
DataReader<T, DELEGATE>::read()
{
    return this->delegate()->read();
}

template <typename T, template <typename Q> class DELEGATE>
LoanedSamples<T>
DataReader<T, DELEGATE>::take()
{
    return this->delegate()->take();
}


template <typename T, template <typename Q> class DELEGATE>
template <typename SamplesFWIterator>
uint32_t
DataReader<T, DELEGATE>::read(SamplesFWIterator sfit, uint32_t max_samples)
{
    return this->delegate()->read(sfit, max_samples);
}

template <typename T, template <typename Q> class DELEGATE>
template <typename SamplesFWIterator>
uint32_t
DataReader<T, DELEGATE>::take(SamplesFWIterator sfit, uint32_t max_samples)
{
    return this->delegate()->take(sfit, max_samples);
}

template <typename T, template <typename Q> class DELEGATE>
template <typename SamplesBIIterator>
uint32_t
DataReader<T, DELEGATE>::read(SamplesBIIterator sbit)
{
    return this->delegate()->read(sbit);
}

template <typename T, template <typename Q> class DELEGATE>
template <typename SamplesBIIterator>
uint32_t
DataReader<T, DELEGATE>::take(SamplesBIIterator sbit)
{
    return this->delegate()->take(sbit);
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::Selector
DataReader<T, DELEGATE>::select()
{
    Selector selector(*this);
    return selector;
}

template <typename T, template <typename Q> class DELEGATE>
dds::topic::TopicInstance<T>
DataReader<T, DELEGATE>::key_value(const dds::core::InstanceHandle& h)
{
    return this->delegate()->key_value(h);
}

template <typename T, template <typename Q> class DELEGATE>
T&
DataReader<T, DELEGATE>::key_value(T& sample, const dds::core::InstanceHandle& h)
{
    return this->delegate()->key_value(sample, h);
}

template <typename T, template <typename Q> class DELEGATE>
const dds::core::InstanceHandle
DataReader<T, DELEGATE>::lookup_instance(const T& key) const
{
    return this->delegate()->lookup_instance(key);
}

template <typename T, template <typename Q> class DELEGATE>
void
DataReader<T, DELEGATE>::listener(
    Listener* listener,
    const dds::core::status::StatusMask& event_mask)
{
    this->delegate()->listener(listener, event_mask);
}

template <typename T, template <typename Q> class DELEGATE>
typename DataReader<T, DELEGATE>::Listener*
DataReader<T, DELEGATE>::listener() const
{
    return this->delegate()->listener();
}

}
}




/***************************************************************************
 *
 * dds/sub/detail/DataReader<> DELEGATE implementation.
 * Declaration can be found in dds/sub/detail/DataReader.hpp
 *
 * Implementation and declaration have been separated because some circular
 * dependencies, like with DataReaderListener and AnyDataReader.
 *
 ***************************************************************************/

#include <dds/sub/AnyDataReader.hpp>
#include <dds/sub/DataReaderListener.hpp>
#include <dds/topic/Topic.hpp>
#include <dds/topic/ContentFilteredTopic.hpp>
#include <org/eclipse/cyclonedds/sub/AnyDataReaderDelegate.hpp>
#include <org/eclipse/cyclonedds/core/ListenerDispatcher.hpp>


template <typename T>
dds::sub::detail::DataReader<T>::DataReader(const dds::sub::Subscriber& sub,
           const dds::topic::Topic<T>& topic,
           const dds::sub::qos::DataReaderQos& qos,
           dds::sub::DataReaderListener<T>* listener,
           const dds::core::status::StatusMask& mask)
    : ::org::eclipse::cyclonedds::sub::AnyDataReaderDelegate(qos, topic), sub_(sub),
      typed_sample_()
{
    common_constructor(listener, mask);
}

template <typename T>
dds::sub::detail::DataReader<T>::DataReader(const dds::sub::Subscriber& sub,
           const dds::topic::ContentFilteredTopic<T, dds::topic::detail::ContentFilteredTopic>& topic,
           const dds::sub::qos::DataReaderQos& qos,
           dds::sub::DataReaderListener<T>* listener,
           const dds::core::status::StatusMask& mask)
  : ::org::eclipse::cyclonedds::sub::AnyDataReaderDelegate(qos, topic), sub_(sub),
    typed_sample_()

{
    common_constructor(listener, mask);
}

template <typename T>
void
dds::sub::detail::DataReader<T>::common_constructor(
            dds::sub::DataReaderListener<T>* listener,
            const dds::core::status::StatusMask& mask)
{
    DDSCXX_WARNING_MSVC_OFF(4127)
    DDSCXX_WARNING_MSVC_OFF(6326)
    if (dds::topic::is_topic_type<T>::value == 0) {
        ISOCPP_THROW_EXCEPTION(ISOCPP_PRECONDITION_NOT_MET_ERROR, "DataReader cannot be created, topic information not found");
    }
    DDSCXX_WARNING_MSVC_ON(6326)
    DDSCXX_WARNING_MSVC_ON(4127)

    org::eclipse::cyclonedds::sub::qos::DataReaderQosDelegate drQos = qos_.delegate();

    dds_entity_t ddsc_sub = sub_.delegate()->get_ddsc_entity();
    dds_entity_t ddsc_top = this->AnyDataReaderDelegate::td_.delegate()->get_ddsc_entity();

    // get and validate the ddsc qos
    drQos.check();
    dds_qos_t* ddsc_qos = drQos.ddsc_qos();

#if 0
    std::string expression = this->AnyDataReaderDelegate::td_.delegate()->reader_expression();
    c_value *params = this->AnyDataReaderDelegate::td_.delegate()->reader_parameters();
#endif

    dds_entity_t ddsc_reader = dds_create_reader(ddsc_sub, ddsc_top, ddsc_qos, NULL);
    dds_delete_qos(ddsc_qos);
    ISOCPP_DDSC_RESULT_CHECK_AND_THROW(ddsc_reader, "Could not create DataReader.");

    this->AnyDataReaderDelegate::td_.delegate()->incrNrDependents();

    this->AnyDataReaderDelegate::setSample(&this->typed_sample_);
    this->set_ddsc_entity(ddsc_reader);
    this->listener(listener, mask);
}

template <typename T>
dds::sub::detail::DataReader<T>::~DataReader/*<T>*/()
{
    if (!this->closed) {
        try {
            close();
        } catch (...) {

        }
    }
}

template <typename T>
void
dds::sub::detail::DataReader<T>::init(ObjectDelegate::weak_ref_type weak_ref)
{
    /* Set weak_ref before passing ourselves to other isocpp objects. */
    this->set_weak_ref(weak_ref);
    /* Add weak_ref to the map of entities */
    this->add_to_entity_map(weak_ref);
    /* Add the datareader to the datareader set of the subscriber */
    this->sub_.delegate()->add_datareader(*this);

    // Because listeners are added after reader is created (which is in enabled state, because
    // disabled state is not yet supported), events could have occured before listeners were
    // registered. Therefore the event handlers for those events are called here.
    if (this->listener_get()) {
        dds::core::status::StatusMask readerStatus = status_changes();

        if (listener_mask.to_ulong() & dds::core::status::StatusMask::data_available().to_ulong()
                && readerStatus.test(DDS_DATA_AVAILABLE_STATUS_ID))
        {
            on_data_available(this->ddsc_entity);
        }
        if (listener_mask.to_ulong() & dds::core::status::StatusMask::liveliness_changed().to_ulong()
                && readerStatus.test(DDS_LIVELINESS_CHANGED_STATUS_ID))
        {
            dds::core::status::LivelinessChangedStatus status = liveliness_changed_status();
            on_liveliness_changed(this->ddsc_entity, status);
        }
        if (listener_mask.to_ulong() & dds::core::status::StatusMask::requested_deadline_missed().to_ulong()
                && readerStatus.test(DDS_REQUESTED_DEADLINE_MISSED_STATUS_ID))
        {
            dds::core::status::RequestedDeadlineMissedStatus status = requested_deadline_missed_status();
            on_requested_deadline_missed(this->ddsc_entity, status);
        }
        if (listener_mask.to_ulong() & dds::core::status::StatusMask::requested_incompatible_qos().to_ulong()
                && readerStatus.test(DDS_REQUESTED_INCOMPATIBLE_QOS_STATUS_ID))
        {
            dds::core::status::RequestedIncompatibleQosStatus status = requested_incompatible_qos_status();
            on_requested_incompatible_qos(this->ddsc_entity, status);
        }
        if (listener_mask.to_ulong() & dds::core::status::StatusMask::sample_lost().to_ulong()
                && readerStatus.test(DDS_SAMPLE_LOST_STATUS_ID))
        {
            dds::core::status::SampleLostStatus status = sample_lost_status();
            on_sample_lost(this->ddsc_entity, status);
        }
        if (listener_mask.to_ulong() & dds::core::status::StatusMask::sample_rejected().to_ulong()
                && readerStatus.test(DDS_SAMPLE_REJECTED_STATUS_ID))
        {
            dds::core::status::SampleRejectedStatus status = sample_rejected_status();
            on_sample_rejected(this->ddsc_entity, status);
        }
        if (listener_mask.to_ulong() & dds::core::status::StatusMask::subscription_matched().to_ulong()
                && readerStatus.test(DDS_SUBSCRIPTION_MATCHED_STATUS_ID))
        {
            dds::core::status::SubscriptionMatchedStatus status = subscription_matched_status();
            on_subscription_matched(this->ddsc_entity, status);
        }
    }

    this->enable();
}

template <typename T>
dds::sub::status::DataState
dds::sub::detail::DataReader<T>::default_filter_state()
{
    org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);

    dds::sub::status::DataState state = this->status_filter_;

    scopedLock.unlock();

    return state;
}

template <typename T>
void
dds::sub::detail::DataReader<T>::default_filter_state(const dds::sub::status::DataState& state)
{
    org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);

    this->status_filter_ = state;

    scopedLock.unlock();
}

template <typename T>
bool
dds::sub::detail::DataReader<T>::is_loan_supported()
{
  this->check();
  return this->AnyDataReaderDelegate::is_loan_supported(static_cast<dds_entity_t>(this->ddsc_entity));
}

template <typename T>
dds::sub::LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob>
dds::sub::detail::DataReader<T>::read_cdr()
{
    dds::sub::LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob> samples;
    dds::sub::detail::CDRSamplesHolder holder(samples);

    this->AnyDataReaderDelegate::read_cdr(static_cast<dds_entity_t>(this->ddsc_entity), this->status_filter_, holder, static_cast<uint32_t>(dds::core::LENGTH_UNLIMITED));

    return samples;
}

template <typename T>
dds::sub::LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob>
dds::sub::detail::DataReader<T>::take_cdr()
{
    dds::sub::LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob> samples;
    dds::sub::detail::CDRSamplesHolder holder(samples);

    this->AnyDataReaderDelegate::take_cdr(static_cast<dds_entity_t>(this->ddsc_entity), this->status_filter_, holder, static_cast<uint32_t>(dds::core::LENGTH_UNLIMITED));

    return samples;
}

template <typename T>
dds::sub::LoanedSamples<T>
dds::sub::detail::DataReader<T>::read()
{
    dds::sub::LoanedSamples<T> samples;
    dds::sub::detail::LoanedSamplesHolder<T> holder(samples);

    this->AnyDataReaderDelegate::loaned_read(static_cast<dds_entity_t>(this->ddsc_entity), this->status_filter_, holder, static_cast<uint32_t>(dds::core::LENGTH_UNLIMITED));

    return samples;
}

template <typename T>
dds::sub::LoanedSamples<T>
dds::sub::detail::DataReader<T>::take()
{
    dds::sub::LoanedSamples<T> samples;
    dds::sub::detail::LoanedSamplesHolder<T> holder(samples);

    this->AnyDataReaderDelegate::loaned_take(static_cast<dds_entity_t>(this->ddsc_entity), this->status_filter_, holder, static_cast<uint32_t>(dds::core::LENGTH_UNLIMITED));

    return samples;
}

template <typename T>
template<typename SamplesFWIterator>
uint32_t
dds::sub::detail::DataReader<T>::read(SamplesFWIterator samples, uint32_t max_samples)
{
    dds::sub::detail::SamplesFWInteratorHolder<T, SamplesFWIterator> holder(samples);

    this->AnyDataReaderDelegate::read(static_cast<dds_entity_t>(this->ddsc_entity), this->status_filter_, holder, max_samples);

    return holder.get_length();
}

template <typename T>
template<typename SamplesFWIterator>
uint32_t
dds::sub::detail::DataReader<T>::take(SamplesFWIterator samples, uint32_t max_samples)
{
    dds::sub::detail::SamplesFWInteratorHolder<T, SamplesFWIterator> holder(samples);

    this->AnyDataReaderDelegate::take(static_cast<dds_entity_t>(this->ddsc_entity), this->status_filter_, holder, max_samples);

    return holder.get_length();
}

template <typename T>
template<typename SamplesBIIterator>
uint32_t
dds::sub::detail::DataReader<T>::read(SamplesBIIterator samples)
{
    dds::sub::detail::SamplesBIIteratorHolder<T, SamplesBIIterator> holder(samples);

    this->AnyDataReaderDelegate::read(static_cast<dds_entity_t>(this->ddsc_entity), this->status_filter_, holder, static_cast<uint32_t>(dds::core::LENGTH_UNLIMITED));

    return holder.get_length();
}

template <typename T>
template<typename SamplesBIIterator>
uint32_t
dds::sub::detail::DataReader<T>::take(SamplesBIIterator samples)
{
    dds::sub::detail::SamplesBIIteratorHolder<T, SamplesBIIterator> holder(samples);

    this->AnyDataReaderDelegate::take(static_cast<dds_entity_t>(this->ddsc_entity), this->status_filter_, holder, static_cast<uint32_t>(dds::core::LENGTH_UNLIMITED));

    return holder.get_length();
}

template <typename T>
dds::topic::TopicInstance<T>
dds::sub::detail::DataReader<T>::key_value(const dds::core::InstanceHandle& h)
{
    T key_holder;

    this->AnyDataReaderDelegate::get_key_value(static_cast<dds_entity_t>(this->ddsc_entity), h, &key_holder);

    return dds::topic::TopicInstance<T>(h, key_holder);
}

template <typename T>
T&
dds::sub::detail::DataReader<T>::key_value(T& key, const dds::core::InstanceHandle& h)
{
    this->AnyDataReaderDelegate::get_key_value(static_cast<dds_entity_t>(this->ddsc_entity), h, &key);

    return key;
}

template <typename T>
const dds::core::InstanceHandle
dds::sub::detail::DataReader<T>::lookup_instance(const T& key) const
{
    dds::core::InstanceHandle handle(this->AnyDataReaderDelegate::lookup_instance(static_cast<dds_entity_t>(this->ddsc_entity), &key));

    return handle;
}

template <typename T>
const dds::sub::Subscriber&
dds::sub::detail::DataReader<T>::subscriber() const
{
    this->check();

    return sub_;
}

template <typename T>
void
dds::sub::detail::DataReader<T>::close()
{
    this->prevent_callbacks();
    org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);

    this->listener_set(NULL, dds::core::status::StatusMask::none());

    this->sub_.delegate()->remove_datareader(*this);

    // Remove our dependency on the topicdescription, and drop our reference to it,
    // so that it can become garbage collected.
    // It is important that we also drop our reference to the topicdescription, since
    // subsequent dependencies between for example ContentFilteredTopic to Topic can
    // only be dropped by the destructor of the ContentFilteredTopic.
    this->AnyDataReaderDelegate::td_.delegate()->decrNrDependents();
    this->AnyDataReaderDelegate::td_ = dds::topic::TopicDescription(dds::core::null);

    org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::close();

    scopedLock.unlock();
}


template <typename T>
dds::sub::DataReaderListener<T>*
dds::sub::detail::DataReader<T>::listener()
{
    this->check();
    return reinterpret_cast<dds::sub::DataReaderListener<T>*>(this->listener_get());
}

template <typename T>
void
dds::sub::detail::DataReader<T>::listener(
        dds::sub::DataReaderListener<T>* l,
        const dds::core::status::StatusMask& event_mask)
{
    org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);
    this->listener_set( l, event_mask ) ;
    scopedLock.unlock();
}

template <typename T>
dds::sub::DataReader<T, dds::sub::detail::DataReader>
dds::sub::detail::DataReader<T>::wrapper()
{
    typename DataReader::ref_type ref =
            ::std::dynamic_pointer_cast<DataReader<T> >(this->get_strong_ref());
    dds::sub::DataReader<T, dds::sub::detail::DataReader> reader(ref);

    return reader;
}

template <typename T>
dds::sub::detail::DataReader<T>::Selector::Selector(typename DataReader<T>::ref_type dr)
    : mode(SELECT_MODE_READ), reader(dr), state_filter_is_set_(false),
      max_samples_(static_cast<uint32_t>(dds::core::LENGTH_UNLIMITED)), query_(dds::core::null)
{
}

template <typename T>
typename dds::sub::detail::DataReader<T>::Selector&
dds::sub::detail::DataReader<T>::Selector::instance(const dds::core::InstanceHandle& h)
{
    this->handle = h;
    switch (this->mode) {
    case SELECT_MODE_READ:
    case SELECT_MODE_READ_INSTANCE:
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->mode = SELECT_MODE_READ_INSTANCE;
        break;
    case SELECT_MODE_READ_WITH_CONDITION:
    case SELECT_MODE_READ_INSTANCE_WITH_CONDITION:
    case SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION:
        this->mode = SELECT_MODE_READ_INSTANCE_WITH_CONDITION;
        break;
    }

    return *this;
}

template <typename T>
typename dds::sub::detail::DataReader<T>::Selector&
dds::sub::detail::DataReader<T>::Selector::next_instance(const dds::core::InstanceHandle& h)
{
    this->handle = h;
    switch (this->mode) {
    case SELECT_MODE_READ:
    case SELECT_MODE_READ_INSTANCE:
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->mode = SELECT_MODE_READ_NEXT_INSTANCE;
        break;
    case SELECT_MODE_READ_WITH_CONDITION:
    case SELECT_MODE_READ_INSTANCE_WITH_CONDITION:
    case SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION:
        this->mode = SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION;
        break;
    }

    return *this;
}

template <typename T>
typename dds::sub::detail::DataReader<T>::Selector&
dds::sub::detail::DataReader<T>::Selector::filter_state(const dds::sub::status::DataState& s)
{
    this->state_filter_ = s;
    this->state_filter_is_set_ = true;

    if ((this->mode == SELECT_MODE_READ_WITH_CONDITION) ||
        (this->mode == SELECT_MODE_READ_INSTANCE_WITH_CONDITION) ||
        (this->mode == SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION)) {
        if (!this->query_.delegate()->modify_state_filter(this->state_filter_)) {
            dds::sub::Query q(this->query_.data_reader(), this->query_.expression(), this->query_.delegate()->parameters());
            q.delegate()->state_filter(this->state_filter_);
            this->query_ = q;
        }
    }

    return *this;
}

template <typename T>
typename dds::sub::detail::DataReader<T>::Selector&
dds::sub::detail::DataReader<T>::Selector::max_samples(uint32_t n)
{
    this->max_samples_ = n;
    return *this;
}

template <typename T>
typename dds::sub::detail::DataReader<T>::Selector&
dds::sub::detail::DataReader<T>::Selector::filter_content(
    const dds::sub::Query& query)
{
    ISOCPP_THROW_EXCEPTION(ISOCPP_UNSUPPORTED_ERROR, "Read with queries not currently supported");
    this->query_ = query;
    switch (this->mode) {
    case SELECT_MODE_READ:
    case SELECT_MODE_READ_INSTANCE:
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->mode = SELECT_MODE_READ_WITH_CONDITION;
        break;
    default:
        break;
    }
    return *this;
}

template <typename T>
dds::sub::LoanedSamples<T>
dds::sub::detail::DataReader<T>::Selector::read()
{
    return this->reader->read(*this);
}

template <typename T>
dds::sub::LoanedSamples<T>
dds::sub::detail::DataReader<T>::Selector::take()
{
    return this->reader->take(*this);
}

// --- Forward Iterators: --- //

template <typename T>
template<typename SamplesFWIterator>
uint32_t
dds::sub::detail::DataReader<T>::Selector::read(SamplesFWIterator sfit, uint32_t max_samples)
{
    return this->reader->read(sfit, max_samples, *this);
}

template <typename T>
template<typename SamplesFWIterator>
uint32_t
dds::sub::detail::DataReader<T>::Selector::take(SamplesFWIterator sfit, uint32_t max_samples)
{
    return this->reader->take(sfit, max_samples, *this);
}

// --- Back-Inserting Iterators: --- //

template <typename T>
template<typename SamplesBIIterator>
uint32_t
dds::sub::detail::DataReader<T>::Selector::read(SamplesBIIterator sbit)
{
    return this->reader->read(sbit, *this);
}

template <typename T>
template<typename SamplesBIIterator>
uint32_t
dds::sub::detail::DataReader<T>::Selector::take(SamplesBIIterator sbit)
{
    return this->reader->take(sbit, *this);
}

template <typename T>
typename dds::sub::detail::DataReader<T>::SelectMode
dds::sub::detail::DataReader<T>::Selector::get_mode() const
{
    return this->mode;
}

template <typename T>
dds::sub::detail::DataReader<T>::ManipulatorSelector::ManipulatorSelector(typename DataReader<T>::ref_type dr) :
      Selector(dr), read_mode_(true)
{
}

template <typename T>
bool
dds::sub::detail::DataReader<T>::ManipulatorSelector::read_mode()
{
    return read_mode_;
}

template <typename T>
void
dds::sub::detail::DataReader<T>::ManipulatorSelector::read_mode(bool b)
{
    read_mode_ = b;
}

template <typename T>
typename dds::sub::detail::DataReader<T>::ManipulatorSelector&
dds::sub::detail::DataReader<T>::ManipulatorSelector::operator >>(dds::sub::LoanedSamples<T>& samples)
{
    if(read_mode_)
    {
        samples = this->Selector::read();
    }
    else
    {
        samples = this->Selector::take();
    }
    return *this;
}


template <typename T>
dds::sub::LoanedSamples<T>
dds::sub::detail::DataReader<T>::read(const Selector& selector)
{
    dds::sub::LoanedSamples<T> samples;
    dds::sub::detail::LoanedSamplesHolder<T> holder(samples);

    switch(selector.mode) {
    case SELECT_MODE_READ:
        this->AnyDataReaderDelegate::loaned_read(static_cast<dds_entity_t>(this->ddsc_entity),
                                          selector.state_filter_,
                                          holder,
                                          selector.max_samples_);
        break;
    case SELECT_MODE_READ_INSTANCE:
        this->AnyDataReaderDelegate::loaned_read_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                   selector.handle,
                                                   selector.state_filter_,
                                                   holder,
                                                   selector.max_samples_);
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->AnyDataReaderDelegate::loaned_read_next_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                        selector.handle,
                                                        selector.state_filter_,
                                                        holder,
                                                        selector.max_samples_);
        break;
    case SELECT_MODE_READ_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    }

    return samples;
}

template <typename T>
dds::sub::LoanedSamples<T>
dds::sub::detail::DataReader<T>::take(const Selector& selector)
{
    dds::sub::LoanedSamples<T> samples;
    dds::sub::detail::LoanedSamplesHolder<T> holder(samples);

    switch(selector.mode) {
    case SELECT_MODE_READ:
        this->AnyDataReaderDelegate::loaned_take(static_cast<dds_entity_t>(this->ddsc_entity),
                                          selector.state_filter_,
                                          holder,
                                          selector.max_samples_);
        break;
    case SELECT_MODE_READ_INSTANCE:
        this->AnyDataReaderDelegate::loaned_take_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                   selector.handle,
                                                   selector.state_filter_,
                                                   holder,
                                                   selector.max_samples_);
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->AnyDataReaderDelegate::loaned_take_next_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                        selector.handle,
                                                        selector.state_filter_,
                                                        holder,
                                                        selector.max_samples_);
        break;
    case SELECT_MODE_READ_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    }

    return samples;
}

// --- Forward Iterators: --- //

template <typename T>
template<typename SamplesFWIterator>
uint32_t
dds::sub::detail::DataReader<T>::read(SamplesFWIterator samples,
              uint32_t max_samples, const Selector& selector)
{
    dds::sub::detail::SamplesFWInteratorHolder<T, SamplesFWIterator> holder(samples);
    max_samples = std::min(max_samples, selector.max_samples_);

    switch(selector.mode) {
    case SELECT_MODE_READ:
        this->AnyDataReaderDelegate::read(static_cast<dds_entity_t>(this->ddsc_entity),
                                          selector.state_filter_,
                                          holder,
                                          max_samples);
        break;
    case SELECT_MODE_READ_INSTANCE:
        this->AnyDataReaderDelegate::read_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                   selector.handle,
                                                   selector.state_filter_,
                                                   holder,
                                                   max_samples);
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->AnyDataReaderDelegate::read_next_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                        selector.handle,
                                                        selector.state_filter_,
                                                        holder,
                                                        max_samples);
        break;
    case SELECT_MODE_READ_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    }

    return holder.get_length();
}

template <typename T>
template<typename SamplesFWIterator>
uint32_t
dds::sub::detail::DataReader<T>::take(SamplesFWIterator samples,
              uint32_t max_samples, const Selector& selector)
{
    dds::sub::detail::SamplesFWInteratorHolder<T, SamplesFWIterator> holder(samples);
    max_samples = std::min(max_samples, selector.max_samples_);

    switch(selector.mode) {
    case SELECT_MODE_READ:
        this->AnyDataReaderDelegate::take(static_cast<dds_entity_t>(this->ddsc_entity),
                                          selector.state_filter_,
                                          holder,
                                          max_samples);
        break;
    case SELECT_MODE_READ_INSTANCE:
        this->AnyDataReaderDelegate::take_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                   selector.handle,
                                                   selector.state_filter_,
                                                   holder,
                                                   max_samples);
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->AnyDataReaderDelegate::take_next_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                        selector.handle,
                                                        selector.state_filter_,
                                                        holder,
                                                        max_samples);
        break;
    case SELECT_MODE_READ_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    }

    return holder.get_length();
}

// --- Back-Inserting Iterators: --- //

template <typename T>
template<typename SamplesBIIterator>
uint32_t
dds::sub::detail::DataReader<T>::read(SamplesBIIterator samples, const Selector& selector)
{
    dds::sub::detail::SamplesBIIteratorHolder<T, SamplesBIIterator> holder(samples);

    switch(selector.mode) {
    case SELECT_MODE_READ:
        this->AnyDataReaderDelegate::read(static_cast<dds_entity_t>(this->ddsc_entity),
                                          selector.state_filter_,
                                          holder,
                                          selector.max_samples_);
        break;
    case SELECT_MODE_READ_INSTANCE:
        this->AnyDataReaderDelegate::read_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                   selector.handle,
                                                   selector.state_filter_,
                                                   holder,
                                                   selector.max_samples_);
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->AnyDataReaderDelegate::read_next_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                        selector.handle,
                                                        selector.state_filter_,
                                                        holder,
                                                        selector.max_samples_);
        break;
    case SELECT_MODE_READ_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    }

    return holder.get_length();
}

template <typename T>
template<typename SamplesBIIterator>
uint32_t
dds::sub::detail::DataReader<T>::take(SamplesBIIterator samples, const Selector& selector)
{
    dds::sub::detail::SamplesBIIteratorHolder<T, SamplesBIIterator> holder(samples);

    switch(selector.mode) {
    case SELECT_MODE_READ:
        this->AnyDataReaderDelegate::take(static_cast<dds_entity_t>(this->ddsc_entity),
                                          selector.state_filter_,
                                          holder,
                                          selector.max_samples_);
        break;
    case SELECT_MODE_READ_INSTANCE:
        this->AnyDataReaderDelegate::take_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                   selector.handle,
                                                   selector.state_filter_,
                                                   holder,
                                                   selector.max_samples_);
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE:
        this->AnyDataReaderDelegate::take_next_instance(static_cast<dds_entity_t>(this->ddsc_entity),
                                                        selector.handle,
                                                        selector.state_filter_,
                                                        holder,
                                                        selector.max_samples_);
        break;
    case SELECT_MODE_READ_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    case SELECT_MODE_READ_NEXT_INSTANCE_WITH_CONDITION:
        /* When SQL queries and QueryContitions are supported, then
         * create a QueryContitions from the Query and Reader status_filter_.
         * Use the resulting condition entity to read. */
        break;
    }

    return holder.get_length();
}


namespace dds
{
namespace sub
{

template <typename SELECTOR>
SELECTOR& read(SELECTOR& selector)
{
    selector.read_mode(true);
    return selector;
}

template <typename SELECTOR>
SELECTOR& take(SELECTOR& selector)
{
    selector.read_mode(false);
    return selector;
}

inline dds::sub::functors::MaxSamplesManipulatorFunctor
max_samples(uint32_t n)
{
    return dds::sub::functors::MaxSamplesManipulatorFunctor(n);
}

inline dds::sub::functors::ContentFilterManipulatorFunctor
content(const dds::sub::Query& query)
{
    return dds::sub::functors::ContentFilterManipulatorFunctor(query);
}

inline dds::sub::functors::StateFilterManipulatorFunctor
state(const dds::sub::status::DataState& s)
{
    return dds::sub::functors::StateFilterManipulatorFunctor(s);
}

inline dds::sub::functors::InstanceManipulatorFunctor
instance(const dds::core::InstanceHandle& h)
{
    return dds::sub::functors::InstanceManipulatorFunctor(h);
}

inline dds::sub::functors::NextInstanceManipulatorFunctor
next_instance(const dds::core::InstanceHandle& h)
{
    return dds::sub::functors::NextInstanceManipulatorFunctor(h);
}

}
}

template <typename T>
void dds::sub::detail::DataReader<T>::on_requested_deadline_missed(dds_entity_t,
        org::eclipse::cyclonedds::core::RequestedDeadlineMissedStatusDelegate &sd)
{
    dds::core::status::RequestedDeadlineMissedStatus s;
    s.delegate() = sd;

    dds::sub::DataReader<T, dds::sub::detail::DataReader> dr = wrapper();

    dds::sub::DataReaderListener<T> *l =
        reinterpret_cast<dds::sub::DataReaderListener<T> *>(this->listener_get());
    l->on_requested_deadline_missed(dr, s);
}

template <typename T>
void dds::sub::detail::DataReader<T>::on_requested_incompatible_qos(dds_entity_t,
        org::eclipse::cyclonedds::core::RequestedIncompatibleQosStatusDelegate &sd)
{
    dds::core::status::RequestedIncompatibleQosStatus s;
    s.delegate() = sd;

    dds::sub::DataReader<T, dds::sub::detail::DataReader> dr = wrapper();

    dds::sub::DataReaderListener<T> *l =
        reinterpret_cast<dds::sub::DataReaderListener<T> *>(this->listener_get());
    l->on_requested_incompatible_qos(dr, s);
}

template <typename T>
void dds::sub::detail::DataReader<T>::on_sample_rejected(dds_entity_t,
             org::eclipse::cyclonedds::core::SampleRejectedStatusDelegate &sd)
{
    dds::core::status::SampleRejectedStatus s;
    s.delegate() = sd;

    dds::sub::DataReader<T, dds::sub::detail::DataReader> dr = wrapper();

    dds::sub::DataReaderListener<T> *l =
            reinterpret_cast<dds::sub::DataReaderListener<T> *>(this->listener_get());
    l->on_sample_rejected(dr, s);
}


template <typename T>
void dds::sub::detail::DataReader<T>::on_liveliness_changed(dds_entity_t,
             org::eclipse::cyclonedds::core::LivelinessChangedStatusDelegate &sd)
{
    dds::core::status::LivelinessChangedStatus s;
    s.delegate() = sd;

    dds::sub::DataReader<T, dds::sub::detail::DataReader> dr = wrapper();

    dds::sub::DataReaderListener<T> *l =
            reinterpret_cast<dds::sub::DataReaderListener<T> *>(this->listener_get());
    l->on_liveliness_changed(dr, s);
}

template <typename T>
void dds::sub::detail::DataReader<T>::on_data_available(dds_entity_t)
{
    dds::sub::DataReader<T, dds::sub::detail::DataReader> dr = wrapper();

    dds::sub::DataReaderListener<T> *l =
        reinterpret_cast<dds::sub::DataReaderListener<T> *>(this->listener_get());
    l->on_data_available(dr);
}

template <typename T>
void dds::sub::detail::DataReader<T>::on_subscription_matched(dds_entity_t,
        org::eclipse::cyclonedds::core::SubscriptionMatchedStatusDelegate &sd)
{
    dds::core::status::SubscriptionMatchedStatus s;
    s.delegate() = sd;

    dds::sub::DataReader<T, dds::sub::detail::DataReader> dr = wrapper();

    dds::sub::DataReaderListener<T> *l =
        reinterpret_cast<dds::sub::DataReaderListener<T> *>(this->listener_get());
    l->on_subscription_matched(dr, s);
}

template <typename T>
void dds::sub::detail::DataReader<T>::on_sample_lost(dds_entity_t,
        org::eclipse::cyclonedds::core::SampleLostStatusDelegate &sd)
{
    dds::core::status::SampleLostStatus s;
    s.delegate() = sd;

    dds::sub::DataReader<T, dds::sub::detail::DataReader> dr = wrapper();

    dds::sub::DataReaderListener<T> *l =
        reinterpret_cast<dds::sub::DataReaderListener<T> *>(this->listener_get());
    l->on_sample_lost(dr, s);
}

// End of implementation

#endif /* CYCLONEDDS_DDS_SUB_TDATAREADER_IMPL_HPP_ */
