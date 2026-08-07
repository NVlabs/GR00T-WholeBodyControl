#ifndef OMG_DDS_TOPIC_ANY_TOPIC_LISTENER_HPP_
#define OMG_DDS_TOPIC_ANY_TOPIC_LISTENER_HPP_

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

#include <dds/topic/AnyTopic.hpp>

namespace dds
{
namespace topic
{


/**
 * @brief
 * AnyTopic events Listener
 *
 * Because the DomainParticipant does not have knowledge of data types,
 * it has to use non-data-type-listeners. In other words Any* listeners.
 *
 * <b><i>
 * All operations for this interface must be implemented in the user-defined class, it is
 * up to the application whether an operation is empty or contains some functionality.
 * </i></b>
 *
 * This class is used as a base for other listeners and is not used on its own.
 *
 * @see dds::domain::DomainParticipantListener
 */
class OMG_DDS_API AnyTopicListener
{
public:
    /** @cond */
    virtual ~AnyTopicListener() { }
    /** @endcond */

public:
    /** @copydoc dds::topic::TopicListener::on_inconsistent_topic() */
    virtual void on_inconsistent_topic(
        AnyTopic& topic,
        const dds::core::status::InconsistentTopicStatus& status) = 0;
};


/**
 * @brief
 * AnyTopic events Listener
 *
 * This listener is just like AnyTopicListener, except
 * that the application doesn't have to implement all operations.
 *
 * This class is used as a base for other listeners and is not used on its own.
 *
 * @see dds::topic::AnyTopicListener
 * @see dds::domain::NoOpDomainParticipantListener
 */
class OMG_DDS_API NoOpAnyTopicListener : public virtual AnyTopicListener
{
/** @cond
 * All these functions have already been documented in the non-NoOp listener.
 * Ignore these functions for the doxygen API documentation for clarity.
 */
public:
    virtual ~NoOpAnyTopicListener() { }

public:
    virtual void on_inconsistent_topic(
        AnyTopic&,
        const dds::core::status::InconsistentTopicStatus&) { }
/** @endcond */
};

}
}

#endif /* OMG_DDS_TOPIC_ANY_TOPIC_LISTENER_HPP_ */
