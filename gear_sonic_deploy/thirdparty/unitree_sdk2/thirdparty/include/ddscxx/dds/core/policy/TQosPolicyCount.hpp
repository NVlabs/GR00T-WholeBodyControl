#ifndef OMG_TDDS_CORE_POLICY_QOS_POLICY_COUNT_HPP_
#define OMG_TDDS_CORE_POLICY_QOS_POLICY_COUNT_HPP_

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

#include <dds/core/Value.hpp>

namespace dds
{
namespace core
{
namespace policy
{

#if defined(__GNUC__) && (__GNUC__ >= 10)
_Pragma("GCC diagnostic push")
_Pragma("GCC diagnostic ignored \"-Wanalyzer-null-dereference\"")
#endif

/**
 * The QosPolicyCount object shows, for a QosPolicy, the total number of
 * times that the concerned DataWriter discovered a DataReader for the
 * same Topic and a requested DataReaderQos that is incompatible with
 * the one offered by the DataWriter.
 */
template <typename D>
class TQosPolicyCount : public dds::core::Value<D>
{
public:
    /**
     * Creates a QosPolicyCount instance
     *
     * @param policy_id the policy_id
     * @param count the count
     */
    TQosPolicyCount(QosPolicyId policy_id, int32_t count);

    /**
     * Copies a QosPolicyCount instance
     *
     * @param other the QosPolicyCount instance to copy
     */
    TQosPolicyCount(const TQosPolicyCount& other);

    /**
     * Copies a QosPolicyCount instance
     *
     * @param other the QosPolicyCount instance to copy
     *
     * @return a reference to the QosPolicyCount that was copied to
     */
    TQosPolicyCount& operator=(const TQosPolicyCount& other) = default;

public:
    /**
     * Gets the policy_id
     *
     * @return the policy_id
     */
    QosPolicyId policy_id() const;

    /**
     * Gets the count
     *
     * @return the count
     */
    int32_t count() const;
};

}
}
}

#if defined(__GNUC__) && (__GNUC__ >= 10)
_Pragma("GCC diagnostic pop")
#endif

#endif // !defined(OMG_TDDS_CORE_POLICY_QOS_POLICY_COUNT_HPP_)
