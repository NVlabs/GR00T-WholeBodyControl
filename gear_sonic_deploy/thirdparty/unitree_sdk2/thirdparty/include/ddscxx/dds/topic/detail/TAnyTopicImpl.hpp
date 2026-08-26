/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_DDS_TOPIC_TANYTOPIC_IMPL_HPP_
#define CYCLONEDDS_DDS_TOPIC_TANYTOPIC_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/topic/TAnyTopic.hpp>

// Implementation
namespace dds
{
namespace topic
{

template <typename DELEGATE>
dds::topic::qos::TopicQos
TAnyTopic<DELEGATE>::qos() const
{
    return this->delegate()->qos();
}

template <typename DELEGATE>
void
TAnyTopic<DELEGATE>::qos(const dds::topic::qos::TopicQos& qos)
{
    this->delegate()->qos(qos);
}

template <typename DELEGATE>
TAnyTopic<DELEGATE>&
TAnyTopic<DELEGATE>::operator << (const dds::topic::qos::TopicQos& qos)
{
    this->delegate()->qos(qos);
    return *this;
}

template <typename DELEGATE>
const TAnyTopic<DELEGATE>&
TAnyTopic<DELEGATE>::operator >> (dds::topic::qos::TopicQos& qos) const
{
    qos = this->delegate()->qos();
    return *this;
}

template <typename DELEGATE>
dds::core::status::InconsistentTopicStatus
TAnyTopic<DELEGATE>::inconsistent_topic_status() const
{
    return this->delegate()->inconsistent_topic_status();
}


}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_TOPIC_TANYTOPIC_IMPL_HPP_ */
