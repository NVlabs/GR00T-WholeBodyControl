#ifndef OMG_DDS_TTOPIC_TOPIC_HPP_
#define OMG_DDS_TTOPIC_TOPIC_HPP_

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

#include <dds/core/conformance.hpp>
#include <dds/core/types.hpp>
#include <dds/core/ref_traits.hpp>
#include <dds/domain/DomainParticipant.hpp>
#include <dds/topic/qos/TopicQos.hpp>
#include <dds/topic/TopicDescription.hpp>
#include <dds/topic/AnyTopic.hpp>


namespace dds
{
namespace topic
{
template <typename T, template <typename Q> class DELEGATE>
class Topic;

template <typename T>
class TopicListener;
}
}


/**
 * @brief
 * Topic is the most basic description of the data to be published and
 * subscribed.
 *
 * A Topic is identified by its name, which must be unique in the whole Domain.
 * In addition (by virtue of extending TopicDescription) it fully specifies the
 * type of the data that can be communicated when publishing or subscribing to
 * the Topic.
 *
 * Topic is the only TopicDescription that can be used for publications and
 * therefore associated with a DataWriter.
 *
 * <b><i>Example</i></b>
 * @code{.cpp}
 * // Default creation of a Topic
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
 *
 * // The Topic can be used to create readers and writers
 * // DataReader
 * dds::sub::Subscriber subscriber(participant);
 * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
 * // DataWriter
 * dds::pub::Publisher publisher(participant);
 * dds::pub::DataWriter<Foo::Bar> writer(publisher, topic);
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_TopicDefinition "Topic Definition"
 */
template <typename T, template <typename Q> class DELEGATE>
class dds::topic::Topic : public dds::topic::TAnyTopic< DELEGATE<T> >
{
public:
    /**
     * Convenience typedef for the type of the data sample.
     */
    typedef T                               DataType;

    /**
     * Local convenience typedef for dds::topic::TopicListener.
     */
    typedef TopicListener<T>                Listener;

public:
    OMG_DDS_REF_TYPE_PROTECTED_DC_T(Topic, ::dds::topic::TAnyTopic, T, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(Topic)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_DEFAULT(Topic)

public:

    /**
     * Create a new Topic.
     *
     * This operation creates a reference to a new or existing Topic under the given name,
     * for a specific data type.
     *
     * <i>QoS</i><br>
     * The Topic will be created with the QoS values specified on the last
     * successful call to @link dds::domain::DomainParticipant::default_topic_qos(const ::dds::topic::qos::TopicQos& qos)
     * dp.default_topic_qos(qos) @endlink or, if the call was never made, the
     * @ref anchor_dds_topic_qos_defaults "default" values.
     *
     * <i>Existing Topic Name</i><br>
     * Before creating a new Topic, this operation performs a
     * lookup_topicdescription for the specified topic_name. When a Topic is
     * found with the same name in the current domain, the QoS and type_name of the
     * found Topic are matched against the parameters qos and type_name. When they
     * are the same, no Topic is created but a new proxy of the existing Topic is returned.
     * When they are not exactly the same, no Topic is created and dds::core::Error is thrown.
     *
     * <i>Local Proxy</i><br>
     * Since a Topic is a global concept in the system, access is provided through a local
     * proxy. In other words, the reference returned is actually not a reference to a Topic
     * but to a locally created proxy. The Data Distribution Service propagates Topics
     * and makes remotely created Topics locally available through this proxy. The deletion
     * of a Topic object will not delete the Topic from the domain, just the local proxy is
     * deleted.
     *
     * @param dp        the domain participant on which the topic will be defined
     * @param topic_name the name of the Topic to be created
     * @throws dds::core::Error
     *                  A other Topic with the same name but different type or QoS was
     *                  detected in the current domain or another internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    Topic(const dds::domain::DomainParticipant& dp,
          const std::string& topic_name);

    /**
     * Create a new Topic.
     *
     * This operation creates a reference to a new or existing Topic under the given name,
     * for a specific data type and type_name.
     *
     * <i>QoS</i><br>
     * The Topic will be created with the QoS values specified on the last
     * successful call to @link dds::domain::DomainParticipant::default_topic_qos(const ::dds::topic::qos::TopicQos& qos)
     * dp.default_topic_qos(qos) @endlink or, if the call was never made, the
     * @ref anchor_dds_topic_qos_defaults "default" values.
     *
     * <i>Existing Topic Name</i><br>
     * Before creating a new Topic, this operation performs a
     * lookup_topicdescription for the specified topic_name. When a Topic is
     * found with the same name in the current domain, the QoS and type_name of the
     * found Topic are matched against the parameters qos and type_name. When they
     * are the same, no Topic is created but a new proxy of the existing Topic is returned.
     * When they are not exactly the same, no Topic is created and dds::core::Error is thrown.
     *
     * <i>Local Proxy</i><br>
     * Since a Topic is a global concept in the system, access is provided through a local
     * proxy. In other words, the reference returned is actually not a reference to a Topic
     * but to a locally created proxy. The Data Distribution Service propagates Topics
     * and makes remotely created Topics locally available through this proxy. The deletion
     * of a Topic object will not delete the Topic from the domain, just the local proxy is
     * deleted.
     *
     * @param dp        the domain participant on which the topic will be defined
     * @param topic_name the topic's name
     * @param type_name a local alias of the data type
     * @throws dds::core::Error
     *                  A other Topic with the same name but different type or QoS was
     *                  detected in the current domain or another internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    Topic(const dds::domain::DomainParticipant& dp,
          const std::string& topic_name,
          const std::string& type_name);

    /**
     * Create a new Topic.
     *
     * This operation creates a reference to a new or existing Topic under the given name,
     * for a specific data type.
     *
     * <i>QoS</i><br>
     * A possible application pattern to construct the TopicQos for the
     * Topic is to:
     * @code{.cpp}
     * // 1) Retrieve the QosPolicy settings on the associated DomainParticipant
     * dds::topic::qos::TopicQos topicQos = participant.default_datareader_qos();
     * // 2) Selectively modify QosPolicy settings as desired.
     * topicQos << dds::core::policy::Durability::Transient();
     * // 3) Use the resulting QoS to construct the DataReader.
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName", topicQos);
     * @endcode
     *
     * <i>Existing Topic Name</i><br>
     * Before creating a new Topic, this operation performs a
     * lookup_topicdescription for the specified topic_name. When a Topic is
     * found with the same name in the current domain, the QoS and type_name of the
     * found Topic are matched against the parameters qos and type_name. When they
     * are the same, no Topic is created but a new proxy of the existing Topic is returned.
     * When they are not exactly the same, no Topic is created and dds::core::Error is thrown.
     *
     * <i>Local Proxy</i><br>
     * Since a Topic is a global concept in the system, access is provided through a local
     * proxy. In other words, the reference returned is actually not a reference to a Topic
     * but to a locally created proxy. The Data Distribution Service propagates Topics
     * and makes remotely created Topics locally available through this proxy. The deletion
     * of a Topic object will not delete the Topic from the domain, just the local proxy is
     * deleted.
     *
     * <i>Listener</i><br>
     * The following statuses are applicable to the TopicListener:
     *  - dds::core::status::StatusMask::inconsistent_topic()
     *
     * See @ref DCPS_Modules_Infrastructure_Listener "listener concept",
     * @ref anchor_dds_topic_listener_commstatus "communication status" and
     * @ref anchor_dds_topic_listener_commpropagation "communication propagation"
     * for more information.
     *
     * @param dp the domain participant on which the topic will be defined
     * @param topic_name the topic's name
     * @param qos the topic listener
     * @param listener the topic listener
     * @param mask the listener event mask
     * @throws dds::core::Error
     *                  A other Topic with the same name but different type or QoS was
     *                  detected in the current domain or another internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    Topic(const dds::domain::DomainParticipant& dp,
          const std::string& topic_name,
          const dds::topic::qos::TopicQos& qos,
          dds::topic::TopicListener<T>* listener = NULL,
          const dds::core::status::StatusMask& mask = dds::core::status::StatusMask::none());

    /**
     * Create a new Topic.
     *
     * This operation creates a reference to a new or existing Topic under the given name,
     * for a specific data type and type_name.
     *
     * <i>QoS</i><br>
     * A possible application pattern to construct the TopicQos for the
     * Topic is to:
     * @code{.cpp}
     * // 1) Retrieve the QosPolicy settings on the associated DomainParticipant
     * dds::topic::qos::TopicQos topicQos = participant.default_datareader_qos();
     * // 2) Selectively modify QosPolicy settings as desired.
     * topicQos << dds::core::policy::Durability::Transient();
     * // 3) Use the resulting QoS to construct the DataReader.
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName", "TypeName", topicQos);
     * @endcode
     *
     * <i>Existing Topic Name</i><br>
     * Before creating a new Topic, this operation performs a
     * lookup_topicdescription for the specified topic_name. When a Topic is
     * found with the same name in the current domain, the QoS and type_name of the
     * found Topic are matched against the parameters qos and type_name. When they
     * are the same, no Topic is created but a new proxy of the existing Topic is returned.
     * When they are not exactly the same, no Topic is created and dds::core::Error is thrown.
     *
     * <i>Local Proxy</i><br>
     * Since a Topic is a global concept in the system, access is provided through a local
     * proxy. In other words, the reference returned is actually not a reference to a Topic
     * but to a locally created proxy. The Data Distribution Service propagates Topics
     * and makes remotely created Topics locally available through this proxy. The deletion
     * of a Topic object will not delete the Topic from the domain, just the local proxy is
     * deleted.
     *
     * <i>Listener</i><br>
     * The following statuses are applicable to the TopicListener:
     *  - dds::core::status::StatusMask::inconsistent_topic()
     *
     * See @ref DCPS_Modules_Infrastructure_Listener "listener concept",
     * @ref anchor_dds_topic_listener_commstatus "communication status" and
     * @ref anchor_dds_topic_listener_commpropagation "communication propagation"
     * for more information.
     *
     * @param dp the domain participant on which the topic will be defined
     * @param topic_name the topic's name
     * @param type_name a local alias of the data type
     * @param qos the topic listener
     * @param listener the topic listener
     * @param mask the listener event mask
     * @throws dds::core::Error
     *                  A other Topic with the same name but different type or QoS was
     *                  detected in the current domain or another internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    Topic(const dds::domain::DomainParticipant& dp,
          const std::string& topic_name,
          const std::string& type_name,
          const dds::topic::qos::TopicQos& qos,
          dds::topic::TopicListener<T>* listener = NULL,
          const dds::core::status::StatusMask& mask = dds::core::status::StatusMask::none());

    #if defined (OMG_DDS_X_TYPE_DYNAMIC_TYPES_SUPPORT)
    /**
     * Create a new topic with a dynamic type description. Notice that in this
     * case the data type has to be DynamicData, so the Topic type will be
     * Topic<DynamicData>.
     *
     * @param dp the domain participant on which the topic will be defined
     * @param topic_name the topic's name. The QoS will be set to
     *        dp.default_topic_qos().
     * @param type the topic type
     * @throws dds::core::Error
     *                  A other Topic with the same name but different type or QoS was
     *                  detected in the current domain or another internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    Topic(const dds::domain::DomainParticipant& dp,
          const std::string& topic_name,
          const dds::core::xtypes::DynamicType type);

    /**
     * Create a new topic with a dynamic type description. Notice that in this
     * case the data type has to be DynamicData, so the Topic type will be
     * Topic<DynamicData>.
     *
     * @param dp the domain participant on which the topic will be defined
     * @param topic_name the topic's name
     * @param type the topic type
     * @param qos the topic listener
     * @param listener the topic listener
     * @param mask the listener event mask
     * @throws dds::core::Error
     *                  A other Topic with the same name but different type or QoS was
     *                  detected in the current domain or another internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    Topic(const dds::domain::DomainParticipant& dp,
          const std::string& topic_name,
          const dds::core::xtypes::DynamicType type
          const dds::topic::qos::TopicQos& qos,
          dds::topic::TopicListener<T>* listener = NULL,
          const dds::core::status::StatusMask& mask = dds::core::status::StatusMask::none());

    #endif /* OMG_DDS_X_TYPE_DYNAMIC_TYPES_SUPPORT */

public:
    /**
     * Register a listener with the Topic.
     *
     * This operation attaches a TopicListener to the Topic. Only one
     * TopicListener can be attached to each Topic. If a
     * TopicListener was already attached, the operation will replace it with the
     * new one. When the listener is the NULL pointer, it represents a listener that is
     * treated as a NOOP for all statuses activated in the bit mask.
     *
     * Listener un-registration is performed by setting the listener to NULL and mask none().
     *
     * @anchor anchor_dds_topic_listener_commstatus
     * <i>Communication Status</i><br>
     * For each communication status, the StatusChangedFlag flag is initially set to
     * FALSE. It becomes TRUE whenever that communication status changes. For each
     * communication status activated in the mask, the associated TopicListener
     * operation is invoked and the communication status is reset to FALSE, as the listener
     * implicitly accesses the status which is passed as a parameter to that operation. The
     * status is reset prior to calling the listener, so if the application calls the
     * get_<status_name>_status from inside the listener it will see the status
     * already reset. An exception to this rule is the NULL listener, which does not reset the
     * communication statuses for which it is invoked.
     *
     * The following statuses are applicable to the TopicListener:
     *  - dds::core::status::StatusMask::inconsistent_topic()
     *
     * Status bits are declared as a constant and can be used by the application in an OR
     * operation to create a tailored mask. The special constant dds::core::status::StatusMask::none()
     * can be used to indicate that the created entity should not respond to any of its available
     * statuses. The DDS will therefore attempt to propagate these statuses to its factory.
     * The special constant dds::core::status::StatusMask::all() can be used to select all applicable
     * statuses specified in the “Data Distribution Service for Real-time Systems Version
     * 1.2” specification which are applicable to the PublisherListener.
     *
     * @anchor anchor_dds_topic_listener_commpropagation
     * <i>Status Propagation</i><br>
     * In case a communication status is not activated in the mask of the
     * TopicListener, the DomainParticipantListener of the containing DomainParticipant
     * is invoked (if attached and activated for the status that occurred). This allows the
     * application to set a default behaviour in the DomainParticipantListener of the containing
     * DomainParticipant and a Topic specific behaviour when needed. In case the
     * communication status is not activated in the mask of the DomainParticipantListener as
     * well, the application is not notified of the change.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @param listener  the listener
     * @param event_mask the mask defining the events for which the listener
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
    void listener(Listener* listener,
                  const ::dds::core::status::StatusMask& event_mask);

    /**
     * Get the listener of this Topic.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @return the listener
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     */
    Listener* listener() const;

};


#endif /* OMG_DDS_TTOPIC_TOPIC_HPP_ */
