#ifndef OMG_DDS_TOPIC_TANYTOPIC_HPP_
#define OMG_DDS_TOPIC_TANYTOPIC_HPP_

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


namespace dds
{
namespace topic
{
template <typename DELEGATE>
class TAnyTopic;
}
}


/**
 * @brief
 * Typeless base class for the typed Topic.
 *
 * Topics are created type specific (fi Topic<Foo::Bar> topic). However, there
 * are a few places in the API (and possibly application) where the type can not be known
 * while still some Topic has to be passed around, stored or even typeless functionality
 * called.<br>
 * The main examples in the API that needs typeless Topic is: DomainParticipantListener.
 *
 * @see dds::topic::Topic
 */
template <typename DELEGATE>
class dds::topic::TAnyTopic :
    public ::dds::core::TEntity< DELEGATE >,
    public ::dds::topic::TTopicDescription< DELEGATE >
{
    OMG_DDS_REF_TYPE_PROTECTED_DC(TAnyTopic, ::dds::topic::TTopicDescription, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TAnyTopic)

    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_DEFAULT(TAnyTopic)

public:

    //==========================================================================
    //== QoS Management

    /**
     * Gets the TopicQos setting for this instance.
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
    dds::topic::qos::TopicQos qos() const;

    /**
     * This operation replaces the existing set of QosPolicy settings for a Topic.
     *
     * The parameter qos contains the object with the QosPolicy settings which is
     * checked for self-consistency and mutability.
     *
     * When the application tries to change a
     * QosPolicy setting for an enabled Topic, which can only be set before the
     * Topic is enabled, the operation will fail and a
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
    void qos(const dds::topic::qos::TopicQos& qos);

    /** @copydoc dds::topic::TAnyTopic::qos(const dds::topic::qos::TopicQos& qos) */
    TAnyTopic& operator << (const dds::topic::qos::TopicQos& qos);

    /** @copydoc dds::topic::TAnyTopic::qos() */
    const TAnyTopic& operator >> (dds::topic::qos::TopicQos& qos) const;

    /**
     * This operation obtains the InconsistentTopicStatus object of the Topic.
     *
     * The InconsistentTopicStatus can also be monitored using a
     * TopicListener or by using the associated StatusCondition.
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
    ::dds::core::status::InconsistentTopicStatus
    inconsistent_topic_status() const;

};


#endif /* OMG_DDS_TOPIC_TANYTOPIC_HPP_ */
