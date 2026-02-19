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


#ifndef CYCLONEDDS_TOPIC_ANY_TOPIC_LISTENER_HPP_
#define CYCLONEDDS_TOPIC_ANY_TOPIC_LISTENER_HPP_


#include <dds/topic/AnyTopicListener.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace topic
{

class OMG_DDS_API AnyTopicListener
{
public:
    virtual ~AnyTopicListener() { }
};


class OMG_DDS_API NoOpAnyTopicListener : public virtual AnyTopicListener
{
public:
    virtual ~NoOpAnyTopicListener() { }
};

}
}
}
}

#endif /* CYCLONEDDS_TOPIC_ANY_TOPIC_LISTENER_HPP_ */
