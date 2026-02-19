#ifndef OMG_TDDS_CORE_QOS_ENTITY_QOS_HPP_
#define OMG_TDDS_CORE_QOS_ENTITY_QOS_HPP_

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
template <typename DELEGATE>
class TEntityQos;
}
}

/**
 * @brief
 * QoS Container
 *
 * Acts as a container for Qos policies allowing all the policies of
 * an entity to be set and retrieved as a unit.
 *
 * For more information see \ref DCPS_Modules_Infrastructure "Infrastructure Module"
 * and \ref DCPS_QoS "Supported Quality of Service"
 */
template <typename DELEGATE>
class dds::core::TEntityQos : public dds::core::Value<DELEGATE>
{
public:
    /**
     * Create default QoS.
     */
    TEntityQos();

    /**
     * Create copied QoS.
     *
     * @param other the QoS to copy.
     */
    TEntityQos(const TEntityQos& other);

    /**
     * Create moved QoS.
     *
     * @param other the QoS to move.
     */
    TEntityQos(TEntityQos&& other);

    /**
     * Copy to this QoS.
     *
     * @param other the QoS to copy.
     *
     * @return reference to the Qos which was copied to.
     */
    TEntityQos<DELEGATE>& operator=(const TEntityQos<DELEGATE>& other) = default;

    /**
     * Move to this QoS.
     *
     * @param other the QoS to move.
     *
     * @return reference to the Qos which was moved to.
     */
    TEntityQos<DELEGATE>& operator=(TEntityQos<DELEGATE>&& other) = default;

    /**
     * Create/copy QoS from different QoS type.
     *
     * @param qos the QoS to copy policies from.
     */
    template <typename T>
    TEntityQos(const TEntityQos<T>& qos);

public:
    /** @cond */
    ~TEntityQos() = default;
    /** @endcond */

public:
    /**
     * Generic function for setting a policy applicable to this QoS object.
     * Available policies depend on the actual instantiation of the template
     * class, which might be DomainParticipantQos, TopicQos, PublisherQos, etc.
     *
     * @param p the policy to be set for this QoS instance.
     */
    template <typename POLICY>
    TEntityQos& policy(const POLICY& p);

    /**
     * Generic function for obtaining the value of a specific policy
     * belonging to this QoS instance.
     *
     * @return policy
     */
    template <typename POLICY>
    const POLICY& policy() const;

    /**
     * Generic function for obtaining the value of a specific policy
     * belonging to this QoS instance.
     *
     * @return policy
     */
    template <typename POLICY>
    POLICY& policy();

    /**
     * Generic function for setting a policy applicable to this QoS object.
     * Available policies depend on the actual instantiation of the template
     * class, which might be DomainParticipantQos, TopicQos, PublisherQos, etc.
     *
     * @param p the policy to be set for this QoS instance.
     */
    template <typename POLICY>
    TEntityQos& operator << (const POLICY& p);

    /**
     * Generic function for obtaining the value of a specific policy
     * belonging to this QoS instance.
     *
     * @return policy
     */
    template <typename POLICY>
    const TEntityQos& operator >> (POLICY& p) const;

    /**
     * Generic function for setting a policy applicable to this QoS object.
     * Available policies depend on the actual instantiation of the template
     * class, which might be DomainParticipantQos, TopicQos, PublisherQos, etc.
     *
     * @param other the TEntityQos to set
     */
    template <typename T>
    TEntityQos<DELEGATE>& operator = (const TEntityQos<T>& other);

};
/* namespace tdds / namespace core / namespace qos */

#endif /* OMG_TDDS_CORE_QOS_ENTITY_QOS_HPP_ */
