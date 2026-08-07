#ifndef OMG_DDS_TOPIC_TOPIC_LISTENER_HPP_
#define OMG_DDS_TOPIC_TOPIC_LISTENER_HPP_

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

#include "dds/topic/Topic.hpp"

namespace dds
{
namespace topic
{

/**
 * @brief
 * Topic events Listener
 *
 * Since a Topic is an Entity, it has the ability to have a Listener
 * associated with it. In this case, the associated Listener should be of type
 * TopicListener. This interface must be implemented by the
 * application. A user-defined class must be provided by the application which must
 * extend from the TopicListener class.
 *
 * <b><i>
 * All operations for this interface must be implemented in the user-defined class, it is
 * up to the application whether an operation is empty or contains some functionality.
 * </i></b>
 *
 * The TopicListener provides a generic mechanism (actually a
 * callback function) for the Data Distribution Service to notify the application of
 * relevant asynchronous status change events, such as a missed deadline, violation of
 * a QosPolicy setting, etc. The TopicListener is related to
 * changes in communication status StatusConditions.
 *
 * @code{.cpp}
 * // Application example listener
 * class ExampleListener :
 *                public virtual dds::topic::TopicListener<Foo::Bar>
 * {
 * public:
 *     virtual void on_inconsistent_topic (
 *         dds::topic::Topic<Foo::Bar>& topic,
 *         const dds::core::status::InconsistentTopicStatus& status)
 *     {
 *         std::cout << "on_inconsistent_topic" << std::endl;
 *     }
 * };
 *
 * // Create Topic with the listener
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant,
 *                                   "TopicName",
 *                                   participant.default_topic_qos(),
 *                                   new ExampleListener(),
 *                                   dds::core::status::StatusMask::all());
 *
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Topic "Topic"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Listener "Listener information"
 */
template <typename T>
class TopicListener
{
public:
    /** @cond */
    virtual ~TopicListener() { }
    /** @endcond */

public:
    /**
     * This operation is called by the Data Distribution Service when the
     * InconsistentTopicStatus changes.
     *
     * The implementation may be left empty
     * when this functionality is not needed. This operation will only be called when the
     * relevant TopicListener is installed and enabled with the
     * StatusMask::inconsistent_topic(). The InconsistentTopicStatus will change
     * when another Topic exists with the same topic_name but different
     * characteristics.
     *
     * @param topic  contain a pointer to the Topic on which the conflict
     *               occurred (this is an input to the application).
     * @param status contain the InconsistentTopicStatus object (this is
     *               an input to the application).
     */
    virtual void on_inconsistent_topic(
        Topic<T>& topic,
        const dds::core::status::InconsistentTopicStatus& status) = 0;
};


/**
 * @brief
 * Topic events Listener
 *
 * This listener is just like TopicListener, except
 * that the application doesn't have to implement all operations.
 *
 * @code{.cpp}
 * class ExampleListener : public virtual dds::topic::NoOpTopicListener<Foo::Bar>
 * {
 *    // Not necessary to implement any Listener operations.
 * };
 * @endcode
 *
 * @see dds::topic::TopicListener
 */
template <typename T>
class NoOpTopicListener : public virtual TopicListener<T>
{
/** @cond
 * All these functions have already been documented in the non-NoOp listener.
 * Ignore these functions for the doxygen API documentation for clarity.
 */
public:
    virtual ~NoOpTopicListener() { }

public:
    virtual void on_inconsistent_topic(
        Topic<T>&,
        const dds::core::status::InconsistentTopicStatus&) { }
/** @endcond */
};

}
}

#endif /* OMG_DDS_TOPIC_TOPIC_LISTENER_HPP_ */
