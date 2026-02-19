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
#ifndef CYCLONEDDS_DDS_TOPIC_TOPICINSTANCE_HPP_
#define CYCLONEDDS_DDS_TOPIC_TOPICINSTANCE_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/topic/TopicInstance.hpp>

// Implementation

namespace dds
{
namespace topic
{

template <typename T>
TopicInstance<T>::TopicInstance() : h_(dds::core::null) {}

template <typename T>
TopicInstance<T>::TopicInstance(const ::dds::core::InstanceHandle& h)
    : h_(h), sample_() {}

template <typename T>
TopicInstance<T>::TopicInstance(const ::dds::core::InstanceHandle& h, const T& sample)
    : h_(h), sample_(sample) { }

template <typename T>
TopicInstance<T>::operator const ::dds::core::InstanceHandle() const
{
    return h_;
}

template <typename T>
const ::dds::core::InstanceHandle TopicInstance<T>::handle() const
{
    return h_;
}

template <typename T>
void TopicInstance<T>::handle(const ::dds::core::InstanceHandle& h)
{
    h_ = h;
}

template <typename T>
const T& TopicInstance<T>::sample() const
{
    return sample_;
}

template <typename T>
T& TopicInstance<T>::sample()
{
    return sample_;
}

template <typename T>
void TopicInstance<T>::sample(const T& sample)
{
    sample_ = sample;
}

}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_TOPIC_TOPICINSTANCE_HPP_ */
