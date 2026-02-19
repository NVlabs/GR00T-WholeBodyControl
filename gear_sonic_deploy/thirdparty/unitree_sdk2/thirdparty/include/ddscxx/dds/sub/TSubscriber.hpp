#ifndef OMG_TDDS_SUB_SUBSCRIBER_HPP_
#define OMG_TDDS_SUB_SUBSCRIBER_HPP_

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
#include <dds/domain/DomainParticipant.hpp>
#include <dds/sub/qos/DataReaderQos.hpp>

namespace dds
{
namespace sub
{
template <typename DELEGATE>
class TSubscriber;

class SubscriberListener;
}
}

/**
 * @brief
 * A Subscriber is the object responsible for the actual reception of the
 * data resulting from its subscriptions.
 *
 * A Subscriber acts on the behalf of one or several DataReader objects
 * that are related to it. When it receives data (from the other parts of
 * the system), it builds the list of concerned DataReader objects, and then
 * indicates to the application that data is available, through its listener
 * or by enabling related conditions. The application can access the list of
 * concerned DataReader objects through the operation get_datareaders and
 * then access the data available through operations on the DataReader.
 *
 * @see for more information: @ref DCPS_Modules_Subscriber "Subscriber"
 */
template <typename DELEGATE>
class dds::sub::TSubscriber : public dds::core::TEntity<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TSubscriber, dds::core::TEntity, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TSubscriber)

public:
    /**
     * Local convenience typedef for dds::sub::SubscriberListener.
     */
    typedef dds::sub::SubscriberListener                 Listener;

public:
    /**
     * Create a new Subscriber.
     *
     * The Subscriber will be created with the QoS values specified on the last
     * successful call to @link dds::domain::DomainParticipant::default_subscriber_qos(const ::dds::sub::qos::SubscriberQos& qos)
     * dp.default_subscriber_qos(qos) @endlink or, if the call was never made, the
     * @ref anchor_dds_sub_subscriber_qos_defaults "default" values.
     *
     * @param dp the domain participant
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    TSubscriber(const ::dds::domain::DomainParticipant& dp);

    /**
     * Create a new Subscriber.
     *
     * The Subscriber will be created with the given QosPolicy settings and if
     * applicable, attaches the optionally specified SubscriberListener to it.
     *
     * See @ref DCPS_Modules_Infrastructure_Listener "listener" for more information
     * about listeners and possible status propagation to other entities.
     *
     * @param dp the domain participant to create the Subscriber with.
     * @param qos a collection of QosPolicy settings for the new Subscriber. In case
     *            these settings are not self consistent, no Subscriber is created.
     * @param listener the subscriber listener
     * @param mask the mask of events notified to the listener
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::InconsistentPolicyError
     *                  The parameter qos contains conflicting QosPolicy settings.
     */
    TSubscriber(const ::dds::domain::DomainParticipant& dp,
                const dds::sub::qos::SubscriberQos& qos,
                dds::sub::SubscriberListener* listener = NULL,
                const dds::core::status::StatusMask& mask = dds::core::status::StatusMask::none());

public:
    /**
     * This operation invokes the on_data_available operation on
     * DataReaderListener objects which are attached to the contained DataReader
     * entities having new, available data.
     *
     * The notify_datareaders operation ignores the bit mask value of individual
     * DataReaderListener objects, even when the dds::core::status::StatusMask::data_available()
     * has not been set on a DataReader that has new, available data. The
     * on_data_available operation will still be invoked, when the
     * dds::core::status::StatusMask::data_available() bit has not been set on a DataReader,
     * but will not propagate to the DomainParticipantListener.
     *
     * When the DataReader has attached a NULL listener, the event will be consumed
     * and will not propagate to the DomainParticipantListener. (Remember that a
     * NULL listener is regarded as a listener that handles all its events as a NOOP).
     *
     * Look @ref DCPS_Modules_Infrastructure_Status "here" for Status change information.<br>
     * Look @ref DCPS_Modules_Infrastructure_Listener "here" for Listener information.<br>
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
     */
    void notify_datareaders();

    /**
     * Register a listener with the Subscriber.
     *
     * The notifications received by the listener depend on the
     * status mask with which it was registered.
     *
     * Listener un-registration is performed by setting the listener to NULL.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @param listener   the listener
     * @param event_mask the mask defining the events for which the listener
     *                   will be notified.
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
    void listener(Listener* listener,
                  const dds::core::status::StatusMask& event_mask);

    /**
     * Get the listener of this Subscriber.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @return the listener
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     */
    Listener* listener() const;

    /**
     * Gets the SubscriberQos setting for this instance.
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
    const dds::sub::qos::SubscriberQos& qos() const;

    /**
     * Sets the SubscriberQos setting for this instance.
     *
     * @param sqos the qos
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
    void qos(const dds::sub::qos::SubscriberQos& sqos);

    /**
     * Gets the default DataReaderQos of the Subscriber.
     *
     * This operation gets an object with the default DataReader QosPolicy settings of
     * the Subscriber (that is the DataReaderQos) which is used for newly
     * created DataReader objects, in case no QoS was provided during the creation.
     *
     * The values retrieved by this operation match the set of values specified on the last
     * successful call to
     * dds::sub::Subscriber::default_datareader_qos(const dds::sub::qos::DataReaderQos& qos),
     * or, if the call was never made, the @ref anchor_dds_sub_datareader_qos_defaults "default" values.
     *
     * @return the default DataReaderQos
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
    dds::sub::qos::DataReaderQos default_datareader_qos() const;

    /**
     * Sets the default DataReaderQos of the Subscriber.
     *
     * This operation sets the default SubscriberQos of the Subscriber which
     * is used for newly created Subscriber objects, when no QoS is provided.
     *
     * This operation checks if the DataReaderQos is self consistent. If it is not, the
     * operation has no effect and throws dds::core::InconsistentPolicyError.
     *
     * The values set by this operation are returned by dds::sub::Subscriber::default_datareader_qos().
     *
     * @param qos the default DataReaderQos
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
    TSubscriber& default_datareader_qos(const dds::sub::qos::DataReaderQos& qos);

    /**
     * Return the DomainParticipant that owns this Subscriber.
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

    /** @copydoc dds::sub::Subscriber::qos(const dds::sub::qos::SubscriberQos& qos) */
    TSubscriber& operator << (const dds::sub::qos::SubscriberQos& qos);

    /** @copydoc dds::sub::Subscriber::qos() */
    const TSubscriber& operator >> (dds::sub::qos::SubscriberQos& qos) const;
};


#endif /* OMG_TDDS_SUB_SUBSCRIBER_HPP_ */
