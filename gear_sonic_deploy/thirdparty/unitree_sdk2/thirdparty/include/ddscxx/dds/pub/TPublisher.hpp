#ifndef OMG_TDDS_PUB_PUBLISHER_HPP_
#define OMG_TDDS_PUB_PUBLISHER_HPP_

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

#include <dds/core/types.hpp>
#include <dds/core/TEntity.hpp>
#include <dds/pub/qos/DataWriterQos.hpp>
#include <dds/pub/qos/PublisherQos.hpp>
#include <dds/domain/DomainParticipant.hpp>



namespace dds
{
namespace pub
{
template <typename DELEGATE>
class TPublisher;

class PublisherListener;
}
}

/**
 * @brief
 * The Publisher acts on the behalf of one or several DataWriter objects
 * that belong to it.
 *
 * When it is informed of a change to the data associated
 * with one of its DataWriter objects, it decides when it is appropriate
 * to actually send the data-update message. In making this decision, it
 * considers any extra information that goes with the data (timestamp,
 * writer, etc.) as well as the QoS of the Publisher and the DataWriter.
 *
 * @see for more information: @ref DCPS_Modules_Publisher "Publisher"
 */
template <typename DELEGATE>
class dds::pub::TPublisher : public dds::core::TEntity<DELEGATE>
{
public:
    /**
     * Local convenience typedef for dds::pub::PublisherListener.
     */
    typedef dds::pub::PublisherListener                 Listener;

public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TPublisher, dds::core::TEntity, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TPublisher)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_DEFAULT(TPublisher)

    /**
     * Create a new Publisher.
     *
     * The Publisher will be created with the QoS values specified on the last
     * successful call to @link dds::domain::DomainParticipant::default_publisher_qos(const ::dds::pub::qos::PublisherQos& qos)
     * dp.default_publisher_qos(qos) @endlink or, if the call was never made, the
     * @ref anchor_dds_pub_publisher_qos_defaults "default" values.
     *
     * @param dp the domain participant
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    TPublisher(const dds::domain::DomainParticipant& dp);

    /**
     * Create a new Publisher.
     *
     * The Publisher will be created with the given QosPolicy settings and if
     * applicable, attaches the optionally specified PublisherListener to it.
     *
     * See @ref DCPS_Modules_Infrastructure_Listener "listener" for more information
     * about listeners and possible status propagation to other entities.
     *
     * @param dp the domain participant to create the Publisher with.
     * @param qos a collection of QosPolicy settings for the new Publisher. In case
     *            these settings are not self consistent, no Publisher is created.
     * @param listener the publisher listener
     * @param mask the mask of events notified to the listener
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::InconsistentPolicyError
     *                  The parameter qos contains conflicting QosPolicy settings.
     */
    TPublisher(const dds::domain::DomainParticipant& dp,
               const dds::pub::qos::PublisherQos& qos,
               dds::pub::PublisherListener* listener = NULL,
               const dds::core::status::StatusMask& mask = dds::core::status::StatusMask::none());

    //==========================================================================

    /**
     * Gets the PublisherQos setting for this instance.
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
    const dds::pub::qos::PublisherQos& qos() const;


    /**
     * Sets the PublisherQos setting for this instance.
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
     */
    void qos(const dds::pub::qos::PublisherQos& qos);

    /** @copydoc dds::pub::Publisher::qos(const dds::pub::qos::PublisherQos& qos) */
    TPublisher& operator <<(const dds::pub::qos::PublisherQos& qos);

    /** @copydoc dds::pub::Publisher::qos() */
    TPublisher& operator >> (dds::pub::qos::PublisherQos& qos);

    /**
     * Sets the default DataWriterQos of the Publisher.
     *
     * This operation sets the default SubscriberQos of the Publisher which
     * is used for newly created Subscriber objects, when no QoS is provided.
     *
     * This operation checks if the DataWriterQos is self consistent. If it is not, the
     * operation has no effect and throws dds::core::InconsistentPolicyError.
     *
     * The values set by this operation are returned by dds::pub::Publisher::default_datawriter_qos().
     *
     * @param qos the default DataWriterQos
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::UnsupportedError
     *                  One or more of the selected QosPolicy values are
     *                  currently not supported by OpenSplice.
     * @throws dds::core::InconsistentPolicyError
     *                  The parameter qos contains conflicting QosPolicy settings,
     *                  e.g. a history depth that is higher than the specified resource limits.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    TPublisher& default_datawriter_qos(const dds::pub::qos::DataWriterQos& qos);

    /**
     * Gets the default DataWriterQos of the Publisher.
     *
     * This operation gets an object with the default DataWriter QosPolicy settings of
     * the Publisher (that is the DataWriterQos) which is used for newly
     * created DataWriter objects, in case no QoS was provided during the creation.
     *
     * The values retrieved by this operation match the set of values specified on the last
     * successful call to
     * dds::pub::Publisher::default_datawriter_qos(const dds::pub::qos::DataWriterQos& qos),
     * or, if the call was never made, the @ref anchor_dds_pub_datawriter_qos_defaults "default" values.
     *
     * @return the default DataWriterQos
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
    dds::pub::qos::DataWriterQos default_datawriter_qos() const;

    //==========================================================================

    /**
     * Register a listener with the Publisher.
     *
     * The notifications received by the listener depend on the
     * status mask with which it was registered.
     *
     * Listener un-registration is performed by setting the listener to NULL.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @param plistener the listener
     * @param mask      the mask defining the events for which the listener
     *                  will be notified.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::UnsupportedError
     *                  A status was selected that cannot be supported because
     *                  the infrastructure does not maintain the required connectivity information.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    void listener(Listener* plistener,
                  const dds::core::status::StatusMask& mask);

    /**
     * Get the listener of this Publisher.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @return the listener
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     */
    Listener* listener() const;

    //==========================================================================

    /**
     * This operation blocks the calling thread until either all data written
     * by the reliable DataWriter entities is acknowledged by all matched
     * reliable DataReader entities, or else the duration specified by the
     * timeout parameter elapses, whichever happens first.
     *
     * Data is acknowledged by the local infrastructure when it does not need to be stored
     * in its DataWriter’s local history. When a locally-connected subscription (including
     * the networking service) has no more resources to store incoming samples it will start
     * to reject these samples, resulting in their source DataWriters to store them
     * temporarily in their own local history to be retransmitted at a later moment in time.<br>
     * In such scenarios, the wait_for_acknowledgments operation will block until all
     * contained DataWriters have retransmitted their entire history, which is therefore
     * effectively empty, or until the max_wait timeout expires, whichever happens first.
     * In the latter case it will throw dds::core::TimeoutError.
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

    /**
     * Return the DomainParticipant that owns this Publisher.
     *
     * @return the DomainParticipant
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     */
    const dds::domain::DomainParticipant& participant() const;

};


#endif /* OMG_TDDS_PUB_PUBLISHER_HPP_ */
