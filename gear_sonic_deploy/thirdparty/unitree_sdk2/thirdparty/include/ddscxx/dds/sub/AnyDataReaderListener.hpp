#ifndef OMG_DDS_SUB_ANY_DATA_READER_LISTENER_HPP_
#define OMG_DDS_SUB_ANY_DATA_READER_LISTENER_HPP_

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

#include <dds/core/refmacros.hpp>
#include <dds/core/status/Status.hpp>
#include <dds/sub/AnyDataReader.hpp>

namespace dds
{
namespace sub
{


/**
 * @brief
 * AnyDataReader events Listener
 *
 * Because Subscriber and DomainParticipant do not have knowledge of data types,
 * they have to use non-data-type-listeners. In other words Any* listeners.
 *
 * <b><i>
 * All operations for this interface must be implemented in the user-defined class, it is
 * up to the application whether an operation is empty or contains some functionality.
 *
 * This class is used as a base for other listeners and is not used on its own.
 * </i></b>
 *
 * @see dds::sub::SubscriberListener
 * @see dds::domain::DomainParticipantListener
 */
class OMG_DDS_API AnyDataReaderListener
{
public:
    /** @cond */
    typedef ::dds::core::smart_ptr_traits<AnyDataReaderListener>::ref_type ref_type;
    /** @endcond */

public:
    /** @cond */
    virtual ~AnyDataReaderListener() { }
    /** @endcond */

public:
    /** @copydoc dds::sub::DataReaderListener::on_requested_deadline_missed() */
    virtual void on_requested_deadline_missed(
        AnyDataReader& reader,
        const dds::core::status::RequestedDeadlineMissedStatus& status) = 0;

    /** @copydoc dds::sub::DataReaderListener::on_requested_incompatible_qos() */
    virtual void on_requested_incompatible_qos(
        AnyDataReader& reader,
        const dds::core::status::RequestedIncompatibleQosStatus& status) = 0;

    /** @copydoc dds::sub::DataReaderListener::on_sample_rejected() */
    virtual void on_sample_rejected(
        AnyDataReader& reader,
        const dds::core::status::SampleRejectedStatus& status) = 0;

    /** @copydoc dds::sub::DataReaderListener::on_liveliness_changed() */
    virtual void on_liveliness_changed(
        AnyDataReader& reader,
        const dds::core::status::LivelinessChangedStatus& status)  = 0;

    /** @copydoc dds::sub::DataReaderListener::on_data_available() */
    virtual void on_data_available(AnyDataReader& reader) = 0;

    /** @copydoc dds::sub::DataReaderListener::on_subscription_matched() */
    virtual void on_subscription_matched(
        AnyDataReader& reader,
        const dds::core::status::SubscriptionMatchedStatus& status) = 0;

    /** @copydoc dds::sub::DataReaderListener::on_sample_lost() */
    virtual void on_sample_lost(
        AnyDataReader& reader,
        const dds::core::status::SampleLostStatus& status) = 0;
};


/**
 * @brief
 * AnyDataReader events Listener
 *
 * This listener is just like AnyDataReaderListener, except
 * that the application doesn't have to implement all operations.
 *
 * This class is used as a base for other listeners and is not used on its own.
 *
 * @see dds::sub::AnyDataReaderListener
 * @see dds::sub::NoOpSubscriberListener
 * @see dds::domain::NoOpDomainParticipantListener
 */
class OMG_DDS_API NoOpAnyDataReaderListener : public virtual AnyDataReaderListener
{
/** @cond
 * All these functions have already been documented in the non-NoOp listener.
 * Ignore these functions for the doxygen API documentation for clarity.
 */
public:
    virtual ~NoOpAnyDataReaderListener() { }

public:
    virtual void on_requested_deadline_missed(
        AnyDataReader&,
        const dds::core::status::RequestedDeadlineMissedStatus&) { }

    virtual void on_requested_incompatible_qos(
        AnyDataReader&,
        const dds::core::status::RequestedIncompatibleQosStatus&) { }

    virtual void on_sample_rejected(
        AnyDataReader&,
        const dds::core::status::SampleRejectedStatus&) { }

    virtual void on_liveliness_changed(
        AnyDataReader&,
        const dds::core::status::LivelinessChangedStatus&) { }

    virtual void on_data_available(AnyDataReader&) { }

    virtual void on_subscription_matched(
        AnyDataReader&,
        const dds::core::status::SubscriptionMatchedStatus&) { }

    virtual void on_sample_lost(
        AnyDataReader&,
        const dds::core::status::SampleLostStatus&) { }
/** @endcond */
};
}
}

#endif /* OMG_DDS_SUB_ANY_DATA_READER_LISTENER_HPP_ */
