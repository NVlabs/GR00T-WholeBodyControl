#ifndef OMG_DDS_DOMAIN_DOMAINPARTICIPANT_LISTENER_HPP_
#define OMG_DDS_DOMAIN_DOMAINPARTICIPANT_LISTENER_HPP_

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

#include <dds/pub/PublisherListener.hpp>
#include <dds/sub/SubscriberListener.hpp>
#include <dds/topic/AnyTopicListener.hpp>


namespace dds
{
namespace domain
{

DDSCXX_WARNING_MSVC_OFF(4250)

/**
 * @brief
 * DomainParticipant events Listener
 *
 * Since a DomainParticipant is an Entity, it has the ability to have a Listener
 * associated with it. In this case, the associated Listener should be of type
 * DomainParticipantListener. This interface must be implemented by the
 * application. A user-defined class must be provided by the application which must
 * extend from the DomainParticipantListener class.
 *
 * <b><i>
 * All operations for this interface must be implemented in the user-defined class, it is
 * up to the application whether an operation is empty or contains some functionality.
 * </i></b>
 *
 * The DomainParticipantListener provides a generic mechanism (actually a
 * callback function) for the Data Distribution Service to notify the application of
 * relevant asynchronous status change events, such as a missed deadline, violation of
 * a QosPolicy setting, etc. The DomainParticipantListener is related to
 * changes in communication status StatusConditions.
 *
 * @code{.cpp}
 * // Application example listener
 * class ExampleListener :
 *                public virtual dds::domain::DomainParticipantListener
 * {
 * public:
 *     virtual void on_inconsistent_topic (
 *         dds::topic::AnyTopic& topic,
 *         const dds::core::status::InconsistentTopicStatus& status)
 *     {
 *         std::cout << "on_inconsistent_topic" << std::endl;
 *     }
 *
 *     virtual void on_offered_deadline_missed (
 *         dds::pub::AnyDataWriter& writer,
 *         const dds::core::status::OfferedDeadlineMissedStatus& status)
 *     {
 *         std::cout << "on_offered_deadline_missed" << std::endl;
 *     }
 *
 *     virtual void on_offered_incompatible_qos (
 *         dds::pub::AnyDataWriter& writer,
 *         const dds::core::status::OfferedIncompatibleQosStatus& status)
 *     {
 *         std::cout << "on_offered_incompatible_qos" << std::endl;
 *     }
 *
 *     virtual void on_liveliness_lost (
 *         dds::pub::AnyDataWriter& writer,
 *         const dds::core::status::LivelinessLostStatus& status)
 *     {
 *         std::cout << "on_liveliness_lost" << std::endl;
 *     }
 *
 *     virtual void on_publication_matched (
 *         dds::pub::AnyDataWriter& writer,
 *         const dds::core::status::PublicationMatchedStatus& status)
 *     {
 *         std::cout << "on_publication_matched" << std::endl;
 *     }
 *
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
 * // Create DomainParticipant with the listener
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id(),
 *                                            dds::domain::DomainParticipant::default_participant_qos(),
 *                                            new ExampleListener(),
 *                                            dds::core::status::StatusMask::all());
 *
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_DomainParticipant "Domain Participant"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Listener "Listener information"
 */
class OMG_DDS_API DomainParticipantListener :
    public virtual dds::pub::PublisherListener,
    public virtual dds::sub::SubscriberListener,
    public virtual dds::topic::AnyTopicListener
{
public:
    /** @cond */
    virtual ~DomainParticipantListener() { }
    /** @endcond */
};


/**
 * @brief
 * DomainParticipant events Listener
 *
 * This listener is just like DomainParticipantListener, except
 * that the application doesn't have to implement all operations.
 *
 * @code{.cpp}
 * class ExampleListener :
 *                public virtual dds::domain::NoOpDomainParticipantListener
 * {
 *    // Not necessary to implement any Listener operations.
 * };
 * @endcode
 *
 * @see dds::domain::DomainParticipantListener
 */
class OMG_DDS_API NoOpDomainParticipantListener :
    public virtual DomainParticipantListener,
    public virtual dds::pub::NoOpPublisherListener,
    public virtual dds::sub::NoOpSubscriberListener,
    public virtual dds::topic::NoOpAnyTopicListener
{
public:
    /** @cond */
    virtual ~NoOpDomainParticipantListener()  { }
    /** @endcond */
};

DDSCXX_WARNING_MSVC_ON(4250)

}
}

#endif /* OMG_DDS_DOMAIN_DOMAINPARTICIPANT_LISTENER_HPP_ */
