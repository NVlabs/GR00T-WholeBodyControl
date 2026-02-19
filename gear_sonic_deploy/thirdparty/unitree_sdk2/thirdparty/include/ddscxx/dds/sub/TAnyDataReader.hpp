#ifndef OMG_TDDS_SUB_ANY_DATA_READER_HPP_
#define OMG_TDDS_SUB_ANY_DATA_READER_HPP_
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

#include <dds/core/TEntity.hpp>
#include <dds/sub/Subscriber.hpp>
#include <dds/sub/qos/DataReaderQos.hpp>
#include <dds/topic/TopicDescription.hpp>
#include <dds/core/macros.hpp>

namespace dds
{
namespace sub
{
template <typename DELEGATE>
class TAnyDataReader;

}
}

/**
 * @brief
 * Typeless base class for the typed DataReader.
 *
 * DataReaders are created type specific (fi DataReader<Foo::Bar> reader). However, there
 * are many places in the API (and possibly application) where the type can not be known
 * while still some DataReader has to be passed around, stored or even typeless functionality
 * called.<br>
 * Main examples in the API that need typeless DataReader are: Subscriber, SubscriberListener
 * and DomainParticipantListener.
 *
 * @see dds::sub::DataReader
 */
template <typename DELEGATE>
class dds::sub::TAnyDataReader : public dds::core::TEntity<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TAnyDataReader, dds::core::TEntity, DELEGATE)
#if 0
    OMG_DDS_IMPLICIT_REF_BASE(TAnyDataReader)
#else
public:
template <typename H__>
TAnyDataReader(const H__& h)
{
    if (h.is_nil()) {
        /* We got a null object and are not really able to do a typecheck here. */
        /* So, just set a null object. */
        *this = dds::core::null;
    } else {
        this->::dds::core::Reference< DELEGATE_T >::impl_ = ::std::dynamic_pointer_cast< DELEGATE_T >(h.delegate());
        if (h.delegate() != this->::dds::core::Reference< DELEGATE_T >::impl_) {
            throw dds::core::IllegalOperationError(std::string("Attempted invalid cast: ") + typeid(h).name() + " to " + typeid(*this).name());
        }
    }
}

template <typename T__>
TAnyDataReader&
operator=(const T__& rhs) {
    const TAnyDataReader &t = rhs;
    if (this != &t) {
        if (rhs.is_nil()) {
            /* We got a null object and are not really able to do a typecheck here. */
            /* So, just set a null object. */
            *this = dds::core::null;
        } else {
            TAnyDataReader other(rhs);
            /* Dont have to copy when the delegate impl is the same. */
            if (other.delegate() != this->::dds::core::Reference< DELEGATE_T >::impl_) {
                *this = other;
            }
        }
    }
    return *this;
}
#endif
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_DEFAULT(TAnyDataReader)

    //==========================================================================
    // -- Entity Navigation API

    /**
     * Get the Subscriber that owns this DataReader.
     *
     * @return the Subscriber
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     */
    const dds::sub::Subscriber& subscriber() const;

    /**
     * Get the TopicDescription associated with this DataReader.
     *
     * @return the TopicDescription
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     */
    const dds::topic::TopicDescription& topic_description() const;



    /**
     * This operation will block the application thread until all "historical" data is
     * received.
     *
     * This operation behaves differently for DataReader objects which have a
     * non-VOLATILE dds::core::policy::Durability QosPolicy and for DataReader
     * objects which have a VOLATILE dds::core::policy::Durability QosPolicy.
     *
     * As soon as an application enables a non-VOLATILE
     * DataReader it will start receiving both "historical" data, i.e. the data that was
     * written prior to the time the DataReader joined the domain, as well as any new
     * data written by the DataWriter objects. There are situations where the application
     * logic may require the application to wait until all "historical" data is received. This
     * is the purpose of the wait_for_historical_data operation.
     *
     * As soon as an application enables a VOLATILE DataReader it
     * will not start receiving "historical" data but only new data written by the
     * DataWriter objects. By calling wait_for_historical_data the DataReader
     * explicitly requests the Data Distribution Service to start receiving also the
     * "historical" data and to wait until either all "historical" data is received, or the
     * duration specified by the max_wait parameter has elapsed, whichever happens
     * first.
     *
     * <i>Thread Blocking</i><br>
     * The operation wait_for_historical_data blocks the calling thread until either
     * all "historical" data is received, or the duration specified by the max_wait
     * parameter elapses, whichever happens first. When the function returns normally,
     * indicates that all the "historical" data was received. If the function throws
     * TimeoutError, it indicates that max_wait elapsed before all the data was
     * received.
     *
     * @param timeout the time to wait for historical data (can be dds::core::Duration::infinite())
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The entity has not yet been enabled.
     * @throws dds::core::TimeoutError
     *                  Not all data is received before timeout elapsed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    void wait_for_historical_data(const dds::core::Duration& timeout);


    //==========================================================================
    // -- QoS Management

    /**
     * Gets the DataReaderQos setting for this instance.
     *
     * @return the qos
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::sub::qos::DataReaderQos
    qos() const;

    /**
     * This operation replaces the existing set of QosPolicy settings for a DataReader.
     *
     * The parameter qos contains the object with the QosPolicy settings which is
     * checked for self-consistency and mutability.
     *
     * When the application tries to change a
     * QosPolicy setting for an enabled DataReader, which can only be set before the
     * DataReader is enabled, the operation will fail and a
     * ImmutablePolicyError is thrown. In other words, the application must
     * provide the presently set QosPolicy settings in case of the immutable QosPolicy
     * settings. Only the mutable QosPolicy settings can be changed.
     *
     * When the qos contains conflicting QosPolicy settings (not self-consistent),
     * the operation will fail and an InconsistentPolicyError is thrown.
     *
     * @param qos the qos
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::ImmutablePolicyError
     *                  The parameter qos contains an immutable QosPolicy setting with a
     *                  different value than set during enabling of the DataReader.
     * @throws dds::core::InconsistentPolicyError
     *                  The parameter qos contains conflicting QosPolicy settings,
     */
    void qos(const dds::sub::qos::DataReaderQos& qos);

    /** @copydoc dds::sub::TAnyDataReader::qos(const dds::sub::qos::DataReaderQos& qos) */
    TAnyDataReader& operator << (const dds::sub::qos::DataReaderQos& qos);

    /** @copydoc dds::sub::TAnyDataReader::qos() */
    const TAnyDataReader& operator >> (dds::sub::qos::DataReaderQos& qos) const;


    //========================================================================
    // -- Status API
    /**
     * This operation obtains the LivelinessChangedStatus object of the DataReader.
     *
     * This object contains the information whether the liveliness of one or more
     * DataWriter objects that were writing instances read by the DataReader has
     * changed. In other words, some DataWriter have become "alive" or "not alive".
     *
     * The LivelinessChangedStatus can also be monitored using a
     * DataReaderListener or by using the associated StatusCondition.
     *
     * @return the LivelinessChangedStatus
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::core::status::LivelinessChangedStatus
    liveliness_changed_status();

    /**
     * This operation obtains the SampleRejectedStatus object of the DataReader.
     *
     * This object contains the information whether a received sample has been rejected.
     * Samples may be rejected by the DataReader when it runs out of
     * resource_limits to store incoming samples. Usually this means that old
     * samples need to be "onsumed" (for example by "taking" them instead of "reading"
     * them) to make room for newly incoming samples.
     *
     * The SampleRejectedStatus can also be monitored using a
     * DataReaderListener or by using the associated StatusCondition.
     *
     * @return the SampleRejectedStatus
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::core::status::SampleRejectedStatus
    sample_rejected_status();

    /**
     * This operation obtains the SampleLostStatus object of the DataReader.
     *
     * This object contains information whether samples have been lost. This
     * only applies when the dds::core::policy::Reliability QosPolicy is set
     * to RELIABLE. If the ReliabilityQos Policy is set to BEST_EFFORT, the
     * Data Distribution Service will not report the loss of samples.
     *
     * The SampleLostStatus can also be monitored using a
     * DataReaderListener or by using the associated StatusCondition.
     *
     * @return the SampleLostStatus
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::core::status::SampleLostStatus
    sample_lost_status();

    /**
     * This operation obtains the RequestedDeadlineMissedStatus object of the DataReader.
     *
     * This object contains the information whether the deadline that the DataReader
     * was expecting through its dds::core::policy::Deadline QosPolicy was not respected
     * for a specific instance.
     *
     * The RequestedDeadlineMissedStatus can also be monitored using a
     * DataReaderListener or by using the associated StatusCondition.
     *
     * @return the RequestedDeadlineMissedStatus
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::core::status::RequestedDeadlineMissedStatus
    requested_deadline_missed_status();

    /**
     * This operation obtains the RequestedIncompatibleQosStatus object of the DataReader.
     *
     * This object contains the information whether a QosPolicy setting
     * was incompatible with the offered QosPolicy setting.
     *
     * The Request/Offering mechanism is applicable between the DataWriter and the
     * DataReader. If the QosPolicy settings between DataWriter and DataReader
     * are inconsistent, no communication between them is established. In addition the
     * DataWriter will be informed via a REQUESTED_INCOMPATIBLE_QOS status
     * change and the DataReader will be informed via an
     * OFFERED_INCOMPATIBLE_QOS status change.
     *
     * The RequestedIncompatibleQosStatus can also be monitored using a
     * DataReaderListener or by using the associated StatusCondition.
     *
     * @return the RequestedIncompatibleQosStatus
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::core::status::RequestedIncompatibleQosStatus
    requested_incompatible_qos_status();

    /**
     * This operation obtains the SubscriptionMatchedStatus object of the DataReader.
     *
     * This object contains the information whether a new match has been
     * discovered for the current subscription, or whether an existing match has ceased to
     * exist.
     *
     * This means that the status represents that either a DataWriter object has been
     * discovered by the DataReader with the same Topic and a compatible Qos, or that a
     * previously discovered DataWriter has ceased to be matched to the current
     * DataReader. A DataWriter may cease to match when it gets deleted, when it
     * changes its Qos to a value that is incompatible with the current DataReader or
     * when either the DataReader or the DataWriter has chosen to put its matching
     * counterpart on its ignore-list using the dds::sub::ignore or dds::pub::ignore
     * operations on the DomainParticipant.
     *
     * The operation may fail if the infrastructure does not hold the information necessary
     * to fill in the SubscriptionMatchedStatus. This is the case when OpenSplice is
     * configured not to maintain discovery information in the Networking Service. (See
     * the description for the NetworkingService/Discovery/enabled property in
     * the Deployment Manual for more information about this subject.) In this case the
     * operation will throw UnsupportedError.
     *
     * The SubscriptionMatchedStatus can also be monitored using a
     * DataReaderListener or by using the associated StatusCondition.
     *
     * @return the SubscriptionMatchedStatus
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::UnsupportedError
     *                  OpenSplice is configured not to maintain the information
     *                  about "associated" publications.
     */
    dds::core::status::SubscriptionMatchedStatus
    subscription_matched_status();

};


#endif /* OMG_TDDS_SUB_ANY_DATA_READER_HPP_ */
