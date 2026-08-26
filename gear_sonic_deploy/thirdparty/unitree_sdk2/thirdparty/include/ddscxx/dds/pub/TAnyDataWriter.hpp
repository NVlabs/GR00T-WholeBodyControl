#ifndef OMG_TDDS_PUB_ANY_DATA_WRITER_HPP_
#define OMG_TDDS_PUB_ANY_DATA_WRITER_HPP_
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
#include <dds/pub/Publisher.hpp>
#include <dds/pub/qos/DataWriterQos.hpp>
#include <dds/topic/TopicDescription.hpp>

namespace dds
{
namespace pub
{
template <typename DELEGATE>
class TAnyDataWriter;
}
}

/**
 * @brief
 * Typeless base class for the typed DataWriter.
 *
 * DataWriters are created type specific (fi DataWriter<Foo::Bar> writer). However, there
 * are many places in the API (and possibly application) where the type can not be known
 * while still some DataWriter has to be passed around, stored or even typeless functionality
 * called.<br>
 * Main examples in the API that need typeless DataWriter are: Publisher, PublisherListener
 * and DomainParticipantListener.
 *
 */
template <typename DELEGATE>
class dds::pub::TAnyDataWriter : public dds::core::TEntity<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TAnyDataWriter, dds::core::TEntity, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TAnyDataWriter)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_DEFAULT(TAnyDataWriter)

    //==========================================================================
    //== Entity Navigation

    /**
     * Get the Publisher that owns this DataWriter.
     *
     * @return the Publisher
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     */
    const dds::pub::Publisher& publisher() const;

    /**
     * Get the TopicDescription associated with this DataWriter.
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



    //==========================================================================
    //== QoS Management

    /**
     * Gets the DataWriterQos setting for this instance.
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
    dds::pub::qos::DataWriterQos
    qos() const;

    /**
     * This operation replaces the existing set of QosPolicy settings for a DataWriter.
     *
     * The parameter qos contains the object with the QosPolicy settings which is
     * checked for self-consistency and mutability.
     *
     * When the application tries to change a
     * QosPolicy setting for an enabled DataWriter, which can only be set before the
     * DataWriter is enabled, the operation will fail and a
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
     *                  different value than set during enabling of the DataWriter.
     * @throws dds::core::InconsistentPolicyError
     *                  The parameter qos contains conflicting QosPolicy settings,
     */
    void qos(const dds::pub::qos::DataWriterQos& qos);

    /** @copydoc dds::pub::TAnyDataWriter::qos(const dds::pub::qos::DataWriterQos& qos) */
    TAnyDataWriter& operator << (const dds::pub::qos::DataWriterQos& qos);

    /** @copydoc dds::pub::TAnyDataWriter::qos() */
    const TAnyDataWriter& operator >> (dds::pub::qos::DataWriterQos& qos) const;


    //==========================================================================
    //== ACKs

    /**
     * This operation blocks the calling thread until either all data written
     * by the DataWriter is acknowledged by the local infrastructure, or until
     * the duration specified by the timeout parameter elapses,
     * whichever happens first.
     *
     * Data is acknowledged by the local infrastructure when it does not need to be stored
     * in its DataWriter’s local history. When a locally-connected subscription (including
     * the networking service) has no more resources to store incoming samples it will start
     * to reject these samples, resulting in their source DataWriters to store them
     * temporarily in their own local history to be retransmitted at a later moment in time.<br>
     * In such scenarios, the wait_for_acknowledgments operation will block until the
     * DataWriter has retransmitted its entire history, which is therefore effectively
     * empty, or until the timeout expires, whichever happens first. In the latter
     * case, this operation will throw a TimeoutError.
     *
     * <i>
     * Be aware that in case the operation returns normally, the data has only been
     * acknowledged by the local infrastructure: it does not mean all remote subscriptions
     * have already received the data. However, delivering the data to remote nodes is then
     * the sole responsibility of the networking service: even when the publishing
     * application would terminate, all data that has not yet been received may be
     * considered ‘on-route’ and will therefore eventually arrive (unless the networking
     * service itself will crash). In contrast, if a DataWriter would still have data in it’s
     * local history buffer when it terminates, this data is considered ‘lost’.
     * </i>
     *
     * This operation is intended to be used only if one or more of the contained
     * DataWriters has its ReliabilityQosPolicyKind set to RELIABLE.
     * Otherwise the operation will return immediately, since best-effort DataWriters will
     * never store rejected samples in their local history:
     * they will just drop them and continue business as usual.
     *
     * @param timeout the time out duration
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::TimeoutError
     *                  Not all data is acknowledged before timeout elapsed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    void wait_for_acknowledgments(const dds::core::Duration& timeout);

    //==========================================================================
    //== Status Management

    /**
     * This operation obtains the LivelinessLostStatus object of the DataWriter.
     *
     * The LivelinessLostStatus contains the information whether the liveliness (that the
     * DataWriter has committed through its Liveliness QosPolicy) was respected.
     * This means that the status represents whether the DataWriter failed to actively
     * signal its liveliness within the offered liveliness period. If the liveliness is lost, the
     * DataReader objects will consider the DataWriter as no longer “alive”.
     *
     * The LivelinessLostStatus can also be monitored using a
     * DataWriterListener or by using the associated StatusCondition.
     *
     * @return the LivelinessLostStatus
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
    const dds::core::status::LivelinessLostStatus liveliness_lost_status();

    /**
     * This operation obtains the OfferedDeadlineMissedStatus object of the DataWriter.
     *
     * The OfferedDeadlineMissedStatus contains the information whether the deadline (that the
     * DataWriter has committed through its Deadline QosPolicy) was respected for
     * each instance.
     *
     * The OfferedDeadlineMissedStatus can also be monitored using a
     * DataWriterListener or by using the associated StatusCondition.
     *
     * @return the OfferedDeadlineMissedStatus
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
    const dds::core::status::OfferedDeadlineMissedStatus offered_deadline_missed_status();

    /**
     * This operation obtains the OfferedIncompatibleQosStatus object of the DataWriter.
     *
     * The OfferedIncompatibleQosStatus contains the information whether a QosPolicy setting
     * was incompatible with the requested QosPolicy setting.
     *
     * This means that the status represents whether a DataReader object has been
     * discovered by the DataWriter with the same Topic and a requested
     * DataReaderQos that was incompatible with the one offered by the DataWriter.
     *
     * The OfferedIncompatibleQosStatus can also be monitored using a
     * DataWriterListener or by using the associated StatusCondition.
     *
     * @return the OfferedIncompatibleQosStatus
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
    const dds::core::status::OfferedIncompatibleQosStatus offered_incompatible_qos_status();

    /**
     * This operation obtains the PublicationMatchedStatus object of the DataWriter.
     *
     * The PublicationMatchedStatus contains the information whether a new match has been
     * discovered for the current publication, or whether an existing match has ceased to
     * exist.
     *
     * This means that the status represents that either a DataReader object has been
     * discovered by the DataWriter with the same Topic and a compatible Qos, or that a
     * previously discovered DataReader has ceased to be matched to the current
     * DataWriter. A DataReader may cease to match when it gets deleted, when it
     * changes its Qos to a value that is incompatible with the current DataWriter or
     * when either the DataWriter or the DataReader has chosen to put its matching
     * counterpart on its ignore-list using the dds::sub::ignore or
     * dds::pub::ignore operations.
     *
     * The operation may fail if the infrastructure does not hold the information necessary
     * to fill in the PublicationMatchedStatus. This is the case when OpenSplice is
     * configured not to maintain discovery information in the Networking Service. (See
     * the description for the NetworkingService/Discovery/enabled property in
     * the Deployment Manual for more information about this subject.) In this case the
     * operation will throw UnsupportedError.
     *
     * The PublicationMatchedStatus can also be monitored using a
     * DataWriterListener or by using the associated StatusCondition.
     *
     * @return the PublicationMatchedStatus
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::UnsupportedError
     *                  OpenSplice is configured not to maintain the information
     *                  about “associated” subscriptions.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    const dds::core::status::PublicationMatchedStatus publication_matched_status();


    //==========================================================================
    //== Liveliness Management

    /**
     * This operation asserts the liveliness for the DataWriter.
     *
     * This operation will manually assert the liveliness for the DataWriter. This way,
     * the Data Distribution Service is informed that the corresponding DataWriter is
     * still alive. This operation is used in combination with the Liveliness QosPolicy
     * set to Liveliness::ManualByParticipant or Liveliness::ManualByTopic.
     *
     * Writing data via the write operation of a DataWriter will assert the liveliness on
     * the DataWriter itself and its containing DomainParticipant. Therefore,
     * assert_liveliness is only needed when not writing regularly.
     *
     * The liveliness should be asserted by the application, depending on the
     * LivelinessQosPolicy. Asserting the liveliness for this DataWriter can also
     * be achieved by asserting the liveliness to the DomainParticipant.
     *
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The entity has not yet been enabled.
     */
    void assert_liveliness();

};


#endif /* OMG_TDDS_PUB_ANY_DATA_WRITER_HPP_ */
