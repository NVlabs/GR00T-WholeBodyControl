#ifndef OMG_DDS_SUB_SUBSCRIBER_LISTENER_HPP_
#define OMG_DDS_SUB_SUBSCRIBER_LISTENER_HPP_

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

#include <dds/sub/AnyDataReaderListener.hpp>
#include <dds/sub/Subscriber.hpp>

namespace dds
{
namespace sub
{

DDSCXX_WARNING_MSVC_OFF(4250)

/**
 * @brief
 * Subscriber events Listener
 *
 * Since a Subscriber is an Entity, it has the ability to have a Listener
 * associated with it. In this case, the associated Listener should be of type
 * SubscriberListener. This interface must be implemented by the
 * application. A user-defined class must be provided by the application which must
 * extend from the SubscriberListener class.
 *
 * <b><i>
 * All operations for this interface must be implemented in the user-defined class, it is
 * up to the application whether an operation is empty or contains some functionality.
 * </i></b>
 *
 * The SubscriberListener provides a generic mechanism (actually a
 * callback function) for the Data Distribution Service to notify the application of
 * relevant asynchronous status change events, such as a missed deadline, violation of
 * a QosPolicy setting, etc. The SubscriberListener is related to
 * changes in communication status StatusConditions.
 *
 * @code{.cpp}
 * // Application example listener
 * class ExampleListener :
 *                public virtual dds::sub::SubscriberListener
 * {
 * public:
 *     virtual void on_requested_deadline_missed (
 *         dds::sub::AnyDataReader& reader,
 *         const dds::core::status::RequestedDeadlineMissedStatus & status)
 *     {
 *         std::cout << "on_requested_deadline_missed" << std::endl;
 *     }
 *
 *     virtual void on_requested_incompatible_qos (
 *         dds::sub::AnyDataReader& reader,
 *         const dds::core::status::RequestedIncompatibleQosStatus & status)
 *     {
 *         std::cout << "on_requested_incompatible_qos" << std::endl;
 *     }
 *
 *     virtual void on_sample_rejected (
 *         dds::sub::AnyDataReader& reader,
 *         const dds::core::status::SampleRejectedStatus & status)
 *     {
 *         std::cout << "on_sample_rejected" << std::endl;
 *     }
 *
 *     virtual void on_liveliness_changed (
 *         dds::sub::AnyDataReader& reader,
 *         const dds::core::status::LivelinessChangedStatus & status)
 *     {
 *         std::cout << "on_liveliness_changed" << std::endl;
 *     }
 *
 *     virtual void on_data_available (
 *         dds::sub::AnyDataReader& reader)
 *     {
 *         std::cout << "on_data_available" << std::endl;
 *     }
 *
 *     virtual void on_subscription_matched (
 *         dds::sub::AnyDataReader& reader,
 *         const dds::core::status::SubscriptionMatchedStatus & status)
 *     {
 *         std::cout << "on_subscription_matched" << std::endl;
 *     }
 *
 *     virtual void on_sample_lost (
 *         dds::sub::AnyDataReader& reader,
 *         const dds::core::status::SampleLostStatus & status)
 *     {
 *         std::cout << "on_sample_lost" << std::endl;
 *     }
 *
 *     virtual void on_data_on_readers (
 *         dds::sub::Subscriber& subs)
 *     {
 *         std::cout << "on_data_on_readers" << std::endl;
 *     }
 * };
 *
 * // Create Subscriber with the listener
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::sub::Subscriber subscriber(participant,
 *                                 participant.default_subscriber_qos(),
 *                                 new ExampleListener(),
 *                                 dds::core::status::StatusMask::all());
 *
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Subscriber "Subscriber"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Listener "Listener information"
 */
class OMG_DDS_API SubscriberListener : public virtual AnyDataReaderListener
{
public:
    /** @cond */
    typedef ::dds::core::smart_ptr_traits<SubscriberListener>::ref_type ref_type;
    /** @endcond */

public:
    /** @cond */
    virtual ~SubscriberListener() { }
    /** @endcond */

public:
    /**
     * This operation called by the Data Distribution Service when new data is
     * available for this Subscriber.
     *
     * The implementation may be left empty when this
     * functionality is not needed. This operation will only be called when the relevant
     * SubscriberListener is installed and enabled with the
     * StatusMask::data_on_readers().
     *
     * The statuses on_data_on_readers() and on_data_available() will
     * occur together. In case these status changes occur, the Data Distribution Service will
     * look for an attached and activated SubscriberListener or
     * DomainParticipantListener (in that order) for the enabled
     * StatusMask::data_on_readers(). In case the StatusMask::data_on_readers() can not be
     * handled, the Data Distribution Service will look for an attached and activated
     * DataReaderListener, SubscriberListener or
     * DomainParticipantListener for the enabled StatusMask::data_available() (in that
     * order).
     *
     * Note that if on_data_on_readers() is called, then the Data Distribution Service
     * will not try to call on_data_available(), however, the application can force a call
     * to the callback function on_data_available of DataReaderListener objects
     * that have data by means of the Subscriber::notify_datareaders() operation.
     *
     * @param sub contain a pointer to the Subscriber for which data is available (this is
     *            an input to the application provided by the Data Distribution Service).
     */
    virtual void on_data_on_readers(Subscriber& sub) = 0;
};


/**
 * @brief
 * Subscriber events Listener
 *
 * This listener is just like SubscriberListener, except
 * that the application doesn't have to implement all operations.
 *
 * @code{.cpp}
 * class ExampleListener : public virtual dds::sub::NoOpSubscriberListener
 * {
 *    // Not necessary to implement any Listener operations.
 * };
 * @endcode
 *
 * @see dds::sub::SubscriberListener
 */
class OMG_DDS_API NoOpSubscriberListener :
    public virtual SubscriberListener,
    public virtual NoOpAnyDataReaderListener
{
/** @cond
 * All these functions have already been documented in the non-NoOp listener.
 * Ignore these functions for the doxygen API documentation for clarity.
 */
public:
    virtual ~NoOpSubscriberListener() { }

public:
    virtual void on_data_on_readers(Subscriber&) { }
/** @endcond */
};

DDSCXX_WARNING_MSVC_ON(4250)

}
}

#endif /* OMG_DDS_SUB_SUBSCRIBER_LISTENER_HPP_ */
