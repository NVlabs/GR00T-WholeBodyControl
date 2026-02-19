/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

#ifndef CYCLONEDDS_TOPIC_TOPIC_LISTENER_HPP_
#define CYCLONEDDS_TOPIC_TOPIC_LISTENER_HPP_


#include "dds/topic/TopicListener.hpp"

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace topic
{

template <typename T>
class TopicListener : public dds::topic::TopicListener<T>
{
public:
    virtual ~TopicListener() { }
};

template <typename T>
class NoOpTopicListener : public virtual TopicListener<T>
{
public:
    virtual ~NoOpTopicListener() { }

public:
    virtual void on_inconsistent_topic(
        dds::topic::Topic<T>&,
        const dds::core::status::InconsistentTopicStatus&) { }
};

}
}
}
}

#endif /* CYCLONEDDS_TOPIC_TOPIC_LISTENER_HPP_ */
