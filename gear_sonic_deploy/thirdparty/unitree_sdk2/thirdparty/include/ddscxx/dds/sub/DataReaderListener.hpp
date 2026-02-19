#ifndef OMG_DDS_SUB_DATA_READER_LISTENER_HPP_
#define OMG_DDS_SUB_DATA_READER_LISTENER_HPP_

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

#include <dds/core/status/Status.hpp>

namespace dds
{
namespace sub
{
template <typename T>
class DataReaderListener;

template <typename T>
class NoOpDataReaderListener;
}
}

/**
 * @brief
 * DataReader events Listener
 *
 * Since a DataReader is an Entity, it has the ability to have a Listener
 * associated with it. In this case, the associated Listener should be of type
 * DataReaderListener. This interface must be implemented by the
 * application. A user-defined class must be provided by the application which must
 * extend from the DataReaderListener class.
 *
 * <b><i>
 * All operations for this interface must be implemented in the user-defined class, it is
 * up to the application whether an operation is empty or contains some functionality.
 * </i></b>
 *
 * The DataReaderListener provides a generic mechanism (actually a
 * callback function) for the Data Distribution Service to notify the application of
 * relevant asynchronous status change events, such as a missed deadline, violation of
 * a QosPolicy setting, etc. The DataReaderListener is related to
 * changes in communication status StatusConditions.
 *
 * @code{.cpp}
 * // Application example listener
 * class ExampleListener :
 *                public virtual dds::sub::DataReaderListener<Foo::Bar>
 * {
 * public:
 *     virtual void on_requested_deadline_missed (
 *         dds::sub::DataReader<Foo::Bar>& reader,
 *         const dds::core::status::RequestedDeadlineMissedStatus & status)
 *     {
 *         std::cout << "on_requested_deadline_missed" << std::endl;
 *     }
 *
 *     virtual void on_requested_incompatible_qos (
 *         dds::sub::DataReader<Foo::Bar>& reader,
 *         const dds::core::status::RequestedIncompatibleQosStatus & status)
 *     {
 *         std::cout << "on_requested_incompatible_qos" << std::endl;
 *     }
 *
 *     virtual void on_sample_rejected (
 *         dds::sub::DataReader<Foo::Bar>& reader,
 *         const dds::core::status::SampleRejectedStatus & status)
 *     {
 *         std::cout << "on_sample_rejected" << std::endl;
 *     }
 *
 *     virtual void on_liveliness_changed (
 *         dds::sub::DataReader<Foo::Bar>& reader,
 *         const dds::core::status::LivelinessChangedStatus & status)
 *     {
 *         std::cout << "on_liveliness_changed" << std::endl;
 *     }
 *
 *     virtual void on_data_available (
 *         dds::sub::DataReader<Foo::Bar>& reader)
 *     {
 *         std::cout << "on_data_available" << std::endl;
 *     }
 *
 *     virtual void on_subscription_matched (
 *         dds::sub::DataReader<Foo::Bar>& reader,
 *         const dds::core::status::SubscriptionMatchedStatus & status)
 *     {
 *         std::cout << "on_subscription_matched" << std::endl;
 *     }
 *
 *     virtual void on_sample_lost (
 *         dds::sub::DataReader<Foo::Bar>& reader,
 *         const dds::core::status::SampleLostStatus & status)
 *     {
 *         std::cout << "on_sample_lost" << std::endl;
 *     }
 * };
 *
 * // Create DataReader with the listener
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
 * dds::sub::Subscriber subscriber(participant);
 * dds::sub::DataReader<Foo::Bar> reader(subscriber,
 *                                       topic,
 *                                       subscriber.default_datareader_qos(),
 *                                       new ExampleListener(),
 *                                       dds::core::status::StatusMask::all());
 *
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Subscription_DataReader "Data Reader"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Listener "Listener information"
 */
template <typename T>
class dds::sub::DataReaderListener
{
public:
    /** @cond */
    typedef typename ::dds::core::smart_ptr_traits<DataReaderListener>::ref_type ref_type;
    /** @endcond */

public:
    /** @cond */
    virtual ~DataReaderListener() { }
    /** @endcond */

public:
    /**
     * This operation called by the Data Distribution Service when the deadline
     * that the DataReader was expecting through its DeadlineQosPolicy was not
     * respected for a specific instance.
     *
     * The implementation may be left empty when this
     * functionality is not needed. This operation will only be called when the relevant
     * DataReaderListener is installed and enabled for the
     * StatusMask::requested_deadline_missed().
     *
     * @param reader contain a pointer to the DataReader for which
     *               the deadline was missed (this is an input to the application provided by the Data
     *               Distribution Service).
     * @param status contain the
     *               RequestedDeadlineMissedStatus object (this is an input to the application
     *               provided by the Data Distribution Service).
     */
    virtual void on_requested_deadline_missed(
        DataReader<T>& reader,
        const dds::core::status::RequestedDeadlineMissedStatus& status) = 0;

    /**
     * This operation is called by the Data Distribution Service when the
     * RequestedIncompatibleQosStatus changes.
     *
     * The implementation may be
     * left empty when this functionality is not needed. This operation will only be called
     * when the relevant DataReaderListener is installed and enabled for the
     * StatusMask::requested_incompatible_qos().
     *
     * The Data Distribution Service will provide a reference to the DataReader in the
     * parameter reader and the RequestedIncompatibleQosStatus object in the
     * parameter status, for use by the application.
     *
     * When the DataReaderListener on the
     * DataReader is not enabled with the StatusMask::requested_incompatible_qos(),
     * the RequestedIncompatibleQosStatus change will propagate to the SubscriberListener
     * of the Subscriber (if enabled) or to the DomainParticipantListener of the
     * DomainParticipant (if enabled).
     *
     * @param reader the DataReader provided by the Data Distribution Service.
     * @param status the RequestedIncompatibleQosStatus object provided by the
     *               Data Distribution Service.
     */
    virtual void on_requested_incompatible_qos(
        DataReader<T>& reader,
        const dds::core::status::RequestedIncompatibleQosStatus& status) = 0;

    /**
     * This operation called by the Data Distribution Service when a (received)
     * sample has been rejected.
     *
     * Samples may be rejected by the DataReader when it
     * runs out of resource_limits to store incoming samples. Usually this means that
     * old samples need to be ‘consumed’ (for example by ‘taking’ them instead of
     * ‘reading’ them) to make room for newly incoming samples.
     *
     * The implementation may be left empty when this functionality is not needed. This
     * operation will only be called when the relevant DataReaderListener is installed
     * and enabled with the StatusMask::sample_lost().
     *
     * @param reader contains a pointer to the DataReader for which
     *               a sample has been rejected (this is an input to the application provided by the
     *               Data Distribution Service).
     * @param status contains the
     *               SampleRejectedStatus object (this is an input to the application provided by
     *               the Data Distribution Service).
     */
    virtual void on_sample_rejected(
        DataReader<T>& reader,
        const dds::core::status::SampleRejectedStatus& status) = 0;

    /**
     * This operation is called by the Data Distribution Service when the liveliness of
     * one or more DataWriter objects that were writing instances read through this
     * DataReader has changed.
     *
     * In other words, some DataWriter have become
     * “alive” or “not alive”. The implementation may be left empty when this
     * functionality is not needed. This operation will only be called when the relevant
     * DataReaderListener is installed and enabled for the
     * StatusMask::liveliness_changed().
     *
     * @param reader contain a pointer to the DataReader for which
     *               the liveliness of one or more DataWriter objects has changed (this is an input
     *               to the application provided by the Data Distribution Service).
     * @param status contain the
     *               LivelinessChangedStatus object (this is an input to the application
     *               provided by the Data Distribution Service).
     */
    virtual void on_liveliness_changed(
        DataReader<T>& reader,
        const dds::core::status::LivelinessChangedStatus& status) = 0;

    /**
     * This operation is called by the Data Distribution Service when new data is
     * available for this DataReader.
     *
     * The implementation may be left empty when this
     * functionality is not needed. This operation will only be called when the relevant
     * DataReaderListener is installed and enabled for the
     * StatusMask::data_available().
     *
     * The Data Distribution Service will provide a reference to the DataReader in the
     * parameter reader for use by the application.
     *
     * The statuses StatusMask::data_on_readers() and StatusMask::data_available() will
     * occur together. In case these status changes occur, the Data Distribution Service will
     * look for an attached and activated SubscriberListener or
     * DomainParticipantListener (in that order) for the enabled
     * StatusMask::data_on_readers(). In case the StatusMask::data_on_readers() can not be
     * handled, the Data Distribution Service will look for an attached and activated
     * DataReaderListener, SubscriberListener or DomainParticipantListener for the enabled
     * StatusMask::data_available() (in that order).
     *
     * Note that if on_data_on_readers is called, then the Data Distribution Service
     * will not try to call on_data_available, however, the application can force a call
     * to the DataReader objects that have data by means of the Subscriber::notify_datareaders()
     * operation.
     *
     * @param reader contain a pointer to the DataReader for which
     *               data is available (this is an input to the application provided by the Data
     *               Distribution Service).
     */
    virtual void on_data_available(DataReader<T>& reader) = 0;

    /**
     * This operation  is called by the Data
     * Distribution Service when a new match has been discovered for the current
     * subscription, or when an existing match has ceased to exist.
     *
     * Usually this means that
     * a new DataWriter that matches the Topic and that has compatible Qos as the current
     * DataReader has either been discovered, or that a previously discovered DataWriter
     * has ceased to be matched to the current DataReader. A DataWriter may cease to
     * match when it gets deleted, when it changes its Qos to a value that is incompatible
     * with the current DataReader or when either the DataReader or the DataWriter
     * has chosen to put its matching counterpart on its ignore-list using the
     * dds::sub::ignore or dds::pub::ignore operations.
     *
     * The implementation of this Listener operation may be left empty when this
     * functionality is not needed: it will only be called when the relevant
     * DataReaderListener is installed and enabled for the
     * StatusMask::subscription_matched().
     *
     * @param reader contains a pointer to the DataReader for which
     *               a match has been discovered (this is an input to the application provided by the
     *               Data Distribution Service).
     * @param status contains the
     *               SubscriptionMatchedStatus object (this is an input to the application
     *               provided by the Data Distribution Service).
     */
    virtual void on_subscription_matched(
        DataReader<T>& reader,
        const dds::core::status::SubscriptionMatchedStatus& status) = 0;

    /**
     * <b><i>
     * NOTE: This operation is not yet implemented. It is scheduled for a future release.
     * </i></b>
     *
     * @param reader the DataReader the Listener is applied to
     * @param status the SampleLostStatus status
     */
    virtual void on_sample_lost(
        DataReader<T>& reader,
        const dds::core::status::SampleLostStatus& status) = 0;
};


/**
 * @brief
 * DataReader events Listener
 *
 * This listener is just like DataReaderListener, except
 * that the application doesn't have to implement all operations.
 *
 * @code{.cpp}
 * class ExampleListener : public virtual dds::sub::NoOpDataReaderListener<Foo::Bar>
 * {
 *    // Not necessary to implement any Listener operations.
 * };
 * @endcode
 *
 * @see dds::sub::DataReaderListener
 */
template <typename T>
class dds::sub::NoOpDataReaderListener : public virtual DataReaderListener<T>
{
/** @cond
 * All these functions have already been documented in the non-NoOp listener.
 * Ignore these functions for the doxygen API documentation for clarity.
 */
public:
    typedef typename ::dds::core::smart_ptr_traits<NoOpDataReaderListener>::ref_type ref_type;

public:
    virtual ~NoOpDataReaderListener() { }

public:
    virtual void on_requested_deadline_missed(
        DataReader<T>&,
        const dds::core::status::RequestedDeadlineMissedStatus&) { }

    virtual void on_requested_incompatible_qos(
        DataReader<T>&,
        const dds::core::status::RequestedIncompatibleQosStatus&) { }

    virtual void on_sample_rejected(
        DataReader<T>&,
        const dds::core::status::SampleRejectedStatus&) { }

    virtual void on_liveliness_changed(
        DataReader<T>&,
        const dds::core::status::LivelinessChangedStatus&) { }

    virtual void on_data_available(
        DataReader<T>&) { }

    virtual void on_subscription_matched(
        DataReader<T>&,
        const dds::core::status::SubscriptionMatchedStatus&) { }

    virtual void on_sample_lost(
        DataReader<T>&,
        const dds::core::status::SampleLostStatus&) { }
/** @endcond */
};

#endif /* OMG_DDS_SUB_DATA_READER_LISTENER_HPP_ */
