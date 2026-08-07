#ifndef OMG_DDS_TOPIC_TOPIC_INSTANCE_HPP_
#define OMG_DDS_TOPIC_TOPIC_INSTANCE_HPP_

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

#include <dds/core/Time.hpp>
#include <dds/core/InstanceHandle.hpp>

namespace dds
{
namespace topic
{
template <typename T>
class TopicInstance;
}
}

/**
 * @brief
 * A TopicInstance encapsulates a dds::sub::Sample and its associated
 * dds::core::InstanceHandle.
 *
 * @see for more information: @ref DCPS_Modules_TopicDefinition "Topic Definition"
 * @see for more information: @ref dds::sub::Sample
 * @see for more information: @ref dds::core::InstanceHandle
 */
template <typename T>
class dds::topic::TopicInstance
{
public:
    /**
     * Construct a TopicInstance.
     */
    TopicInstance();

    /**
     * Construct a TopicInstance with an InstanceHandle.
     *
     * @param h the InstanceHandle
     */
    explicit TopicInstance(const ::dds::core::InstanceHandle& h);

    /**
     * Construct a TopicInstance with an InstanceHandle and a sample type.
     *
     * @param h the InstanceHandle
     * @param sample the <Type>
     */
    TopicInstance(const ::dds::core::InstanceHandle& h, const T& sample);

public:
    /**
     * Conversion operator to get the InstanceHandle.
     *
     * @return the InstanceHandle for the TopicInstance
     */
    operator const ::dds::core::InstanceHandle() const;

    /**
     * Get the InstanceHandle.
     *
     * @return the InstanceHandle for the TopicInstance
     */
    const ::dds::core::InstanceHandle handle() const;

    /**
     * Set the InstanceHandle.
     *
     * @param h the InstanceHandle to set to the TopicInstance
     */
    void handle(const ::dds::core::InstanceHandle& h);

    /**
     * Get the data sample
     *
     * @return the sample for the TopicInstance
     */
    const T& sample() const;

    /**
     * Get the data sample
     *
     * @return the sample for the TopicInstance
     */
    T& sample();

    /**
     * Set the data sample
     *
     * @param sample send a sample for this TopicInstance
     */
    void sample(const T& sample);

private:
    ::dds::core::InstanceHandle h_;
    T sample_;
};


#endif /* OMG_DDS_TOPIC_TOPIC_INSTANCE_HPP_ */
