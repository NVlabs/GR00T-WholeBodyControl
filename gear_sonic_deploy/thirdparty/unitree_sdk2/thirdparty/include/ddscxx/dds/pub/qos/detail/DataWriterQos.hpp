#ifndef OMG_DDS_PUB_QOS_DETAIL_DATAWRITER_QOS_HPP_
#define OMG_DDS_PUB_QOS_DETAIL_DATAWRITER_QOS_HPP_

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
#include <org/eclipse/cyclonedds/pub/qos/DataWriterQosDelegate.hpp>

#ifdef DOXYGEN_FOR_ISOCPP
/* The above macro is never (and must never) be defined in normal compilation.
 *
 * The class below is just used to create informative API documentation.
 * The predoxygen.py script will copy this over the QoS API header file.
 */
/**
 * @brief
 * This object provides the basic mechanism for an application to specify Quality of
 * Service attributes for a DataWriter.
 *
 * @par Attributes
 * @anchor anchor_dds_pub_datawriter_qos_defaults
 * QosPolicy                               | Desciption                                                                 | Default Value
 * --------------------------------------- | -------------------------------------------------------------------------- | --------------------
 * dds::core::policy::UserData             | Additional information (@ref DCPS_QoS_UserData "info")                     | UserData::UserData(empty)
 * dds::core::policy::Durability           | Data storage settings for late joiners (@ref DCPS_QoS_Durability "info")   | Durability::Volatile()
 * dds::core::policy::Deadline             | Period in which new sample is written (@ref DCPS_QoS_Deadline "info")      | Deadline::Deadline(infinite)
 * dds::core::policy::LatencyBudget        | Used for optimization (@ref DCPS_QoS_LatencyBudget "info")                 | LatencyBudget::LatencyBudget(zero)
 * dds::core::policy::Liveliness           | Liveliness assertion mechanism (@ref DCPS_QoS_Liveliness "info")           | Liveliness::Automatic()
 * dds::core::policy::Reliability          | Reliability settings (@ref DCPS_QoS_Reliability "info")                    | Reliability::Reliable()
 * dds::core::policy::DestinationOrder     | DataReader data order settings (@ref DCPS_QoS_DestinationOrder "info")     | DestinationOrder::ReceptionTimestamp()
 * dds::core::policy::History              | Data storage settings (@ref DCPS_QoS_History "info")                       | History::KeepLast(depth 1)
 * dds::core::policy::ResourceLimits       | Maximum resource settings (@ref DCPS_QoS_ResourceLimits "info")            | ResourceLimits::ResourceLimits(all unlimited)
 * dds::core::policy::TransportPriority    | Priority hint for transport layer (@ref DCPS_QoS_TransportPriority "info") | TransportPriority::TTransportPriority(0)
 * dds::core::policy::Lifespan             | Maximum duration of validity of data (@ref DCPS_QoS_Lifespan "info")       | Lifespan::Lifespan(infinite)
 * dds::core::policy::Ownership            | Exclusive ownership or not (@ref DCPS_QoS_Ownership "info")                | Ownership::Shared()
 * dds::core::policy::OwnershipStrength    | Ownership strenght (@ref DCPS_QoS_OwnershipStrength "info")                | OwnershipStrength::OwnershipStrength(0)
 * dds::core::policy::WriterDataLifecycle  | Dispose with unregister or not (@ref DCPS_QoS_WriterDataLifecycle "info")  | WriterDataLifecycle::AutoDisposeUnregisteredInstances()
 * dds::core::policy::DataRepresentation   | Supported data representation kinds (@ref DCPS_QoS_DataRepresentation "info") | DataRepresentation::DataRepresentation(dds::core::policy::DataRepresentationId::XCDR1)
 * dds::core::policy::TypeConsistencyEnforcement | Type consistency enforcement policies (@ref DCPS_QoS_TypeConsistencyEnforcement "info") | dds::core::policy::TypeConsistencyKind::DISALLOW_TYPE_COERCION
 *
 * A QosPolicy can be set when the DataWriter is created or modified with the set
 * qos operation.
 * Both operations take the DataWriterQos object as a parameter. There may be cases
 * where several policies are in conflict. Consistency checking is performed each time
 * the policies are modified when they are being created and, in case they are already
 * enabled, via the set qos operation.
 *
 * Some QosPolicy have "immutable" semantics meaning that they can only be
 * specified either at DataWriter creation time or prior to calling the enable
 * operation on the DataWriter.
 *
 * @see for more information: @ref DCPS_QoS
 */
class dds::pub::qos::DataWriterQos : public ::dds::core::EntityQos<org::eclipse::cyclonedds::pub::qos::DataWriterQosDelegate>
{
public:
    /**
     * Create @ref anchor_dds_pub_datawriter_qos_defaults "default" QoS.
     */
    DataWriterQos() {}

    /**
     * Create copied QoS type.
     *
     * @param qos the QoS to copy policies from.
     */
    DataWriterQos(const DataWriterQos& qos);

    /**
     * Create a DataWriter QoS from a TopicQos.
     *
     * This operation will copy the QosPolicy settings from the TopicQos to the
     * corresponding QosPolicy settings in the DataWriterQos. The related value
     * in DataWriterQos will be repliced, while the other policies will get the
     * @ref anchor_dds_pub_datawriter_qos_defaults "default" QoS policies.
     *
     * This is a "convenience" operation. It can be used to merge
     * @ref anchor_dds_pub_datawriter_qos_defaults "default" DataWriter
     * QosPolicy settings with the corresponding ones on the Topic. The resulting
     * DataWriterQos can then be used to create a new DataWriter, or set its
     * DataWriterQos.
     * @code{.cpp}
     * dds::topic::qos::TopicQos topicQos = topic.qos();
     * dds::pub::qos::DataWriterQos writerQos(topicQos);
     * // Policies of the DataWriterQos that are not present in the TopicQos
     * // have the default value.
     * @endcode
     *
     * This operation does not check the resulting DataWriterQos for self
     * consistency. This is because the "merged" DataWriterQos may not be the
     * final one, as the application can still modify some QosPolicy settings prior to
     * applying the DataWriterQos to the DataWriter.
     *
     * @param qos the QoS to copy policies from.
     */
    DataWriterQos(const dds::topic::qos::TopicQos& qos);

    /**
     * Assign dds::topic::qos::TopicQos policies to the DataWriterQos.
     *
     * This operation will copy the QosPolicy settings from the TopicQos to the
     * corresponding QosPolicy settings in the DataWriterQos (replacing the values,
     * if present).
     *
     * This is a "convenience" operation, useful in combination with the operations
     * Publisher::default_datawriter_qos() and dds::topic::Topic::qos().
     * This operation can be used to merge the DataWriter
     * QosPolicy settings with the corresponding ones on the Topic. The resulting
     * DataWriterQos can then be used to create a new DataWriter, or set its
     * DataWriterQos.
     * @code{.cpp}
     * dds::topic::qos::TopicQos topicQos = topic.qos();
     * dds::pub::qos::DataWriterQos writerQos = publisher.default_datawriter_qos();
     * writerQos = topicQos;
     * // Policies of the DataWriterQos that are not present in the TopicQos are untouched.
     * @endcode
     *
     * This operation does not check the resulting DataWriterQos for self
     * consistency. This is because the "merged" DataWriterQos may not be the
     * final one, as the application can still modify some QosPolicy settings prior to
     * applying the DataWriterQos to the DataWriter.
     *
     * @param qos the QoS to copy policies from.
     */
    DataWriterQos& operator= (const dds::topic::qos::TopicQos& other);
};

#else /* DOXYGEN_FOR_ISOCPP */

namespace dds { namespace pub { namespace qos { namespace detail {
    typedef dds::core::TEntityQos<org::eclipse::cyclonedds::pub::qos::DataWriterQosDelegate> DataWriterQos;
} } } }

#endif /* DOXYGEN_FOR_ISOCPP */

#endif /* OMG_DDS_PUB_QOS_DETAIL_DATAWRITER_QOS_HPP_ */
