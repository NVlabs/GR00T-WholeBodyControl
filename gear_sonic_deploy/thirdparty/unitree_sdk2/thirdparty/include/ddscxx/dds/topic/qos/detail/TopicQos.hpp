#ifndef OMG_DDS_TOPIC_QOS_DETAIL_TOPIC_QOS_HPP_
#define OMG_DDS_TOPIC_QOS_DETAIL_TOPIC_QOS_HPP_

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

#include <dds/core/detail/TEntityQosImpl.hpp>
#include <org/eclipse/cyclonedds/topic/qos/TopicQosDelegate.hpp>

#ifdef DOXYGEN_FOR_ISOCPP
/* The above macro is never (and must never) be defined in normal compilation.
 *
 * The class below is just used to create informative API documentation.
 * The predoxygen.py script will copy this over the QoS API header file.
 */
/**
 * @brief
 * This struct provides the basic mechanism for an application to specify Quality of
 * Service attributes for a Topic.
 *
 * @par Attributes
 * @anchor anchor_dds_topic_qos_defaults
 * QosPolicy                            | Desciption                                                                 | Default Value
 * ------------------------------------ | -------------------------------------------------------------------------- | --------------------
 * dds::core::policy::TopicData         | Additional information (@ref DCPS_QoS_TopicData "info")                    | TopicData::TopicData(empty)
 * dds::core::policy::Durability        | Data storage settings for late joiners (@ref DCPS_QoS_Durability "info")   | Durability::Volatile()
 * dds::core::policy::DurabilityService | Transient/persistent behaviour (@ref DCPS_QoS_DurabilityService "info")    | DurabilityService::DurabilityService()
 * dds::core::policy::Deadline          | Period in which new sample is written (@ref DCPS_QoS_Deadline "info")      | Deadline::Deadline(infinite)
 * dds::core::policy::LatencyBudget     | Used for optimization (@ref DCPS_QoS_LatencyBudget "info")                 | LatencyBudget::LatencyBudget(zero)
 * dds::core::policy::Liveliness        | Liveliness assertion mechanism (@ref DCPS_QoS_Liveliness "info")           | Liveliness::Automatic()
 * dds::core::policy::Reliability       | Reliability settings (@ref DCPS_QoS_Reliability "info")                    | Reliability::Reliable()
 * dds::core::policy::DestinationOrder  | DataReader data order settings (@ref DCPS_QoS_DestinationOrder "info")     | DestinationOrder::ReceptionTimestamp()
 * dds::core::policy::History           | Data storage settings (@ref DCPS_QoS_History "info")                       | History::KeepLast(depth 1)
 * dds::core::policy::ResourceLimits    | Maximum resource settings (@ref DCPS_QoS_ResourceLimits "info")            | ResourceLimits::ResourceLimits(all unlimited)
 * dds::core::policy::TransportPriority | Priority hint for transport layer (@ref DCPS_QoS_TransportPriority "info") | TransportPriority::TTransportPriority(0)
 * dds::core::policy::Lifespan          | Maximum duration of validity of data (@ref DCPS_QoS_Lifespan "info")       | Lifespan::Lifespan(infinite)
 * dds::core::policy::Ownership         | Exclusive ownership or not (@ref DCPS_QoS_Ownership "info")                | Ownership::Shared()
 * dds::core::policy::DataRepresentation | Supported data representation kinds (@ref DCPS_QoS_DataRepresentation "info") | DataRepresentation::DataRepresentation(dds::core::policy::DataRepresentationId::XCDR1)
 * dds::core::policy::TypeConsistencyEnforcement | Type consistency enforcement policies (@ref DCPS_QoS_TypeConsistencyEnforcement "info") | dds::core::policy::TypeConsistencyKind::DISALLOW_TYPE_COERCION
 *
 * A QosPolicy can be set when the Topic is created or modified with the set
 * qos operation.
 * Both operations take the TopicQos object as a parameter. There may be cases
 * where several policies are in conflict. Consistency checking is performed each time
 * the policies are modified when they are being created and, in case they are already
 * enabled, via the set qos operation.
 *
 * Some QosPolicy have "immutable" semantics meaning that they can only be
 * specified either at Topic creation time or prior to calling the enable
 * operation on the Topic.
 *
 * @see for more information: @ref DCPS_QoS
 */
class dds::topic::qos::TopicQos : public ::dds::core::EntityQos<org::eclipse::cyclonedds::topic::qos::TopicQosDelegate>
{
public:
    /**
     * Create @ref anchor_dds_topic_qos_defaults "default" QoS.
     */
    TopicQos() {}

    /**
     * Create copied QoS type.
     *
     * @param qos the QoS to copy policies from.
     */
    TopicQos(const TopicQos& qos);
};

#else /* DOXYGEN_FOR_ISOCPP */

namespace dds { namespace topic { namespace qos { namespace detail {
	typedef ::dds::core::TEntityQos< ::org::eclipse::cyclonedds::topic::qos::TopicQosDelegate > TopicQos;
} } } }

#endif /* DOXYGEN_FOR_ISOCPP */

#endif /* OMG_DDS_TOPIC_QOS_DETAIL_TOPIC_QOS_HPP_ */
