#ifndef OMG_DDS_SUB_QOS_DETAIL_DATA_READER_QOS_HPP_
#define OMG_DDS_SUB_QOS_DETAIL_DATA_READER_QOS_HPP_

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
#include <org/eclipse/cyclonedds/sub/qos/DataReaderQosDelegate.hpp>

#ifdef DOXYGEN_FOR_ISOCPP
/* The above macro is never (and must never) be defined in normal compilation.
 *
 * The class below is just used to create informative API documentation.
 * The predoxygen.py script will copy this over the QoS API header file.
 */
/**
 * @brief
 * This class provides the basic mechanism for an application to specify Quality of
 * Service attributes for a DataReader.
 *
 * @par Attributes
 * @anchor anchor_dds_sub_datareader_qos_defaults
 * QosPolicy                                      | Desciption                                                                          | Default Value
 * ---------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------
 * dds::core::policy::Durability                  | Data storage settings for late joiners (@ref DCPS_QoS_Durability "info")            | Durability::Volatile()
 * dds::core::policy::Deadline                    | Period in which new sample is written (@ref DCPS_QoS_Deadline "info")               | Deadline::Deadline(infinite)
 * dds::core::policy::LatencyBudget               | Used for optimization (@ref DCPS_QoS_LatencyBudget "info")                          | LatencyBudget::LatencyBudget(zero)
 * dds::core::policy::Liveliness                  | Liveliness assertion mechanism (@ref DCPS_QoS_Liveliness "info")                    | Liveliness::Automatic()
 * dds::core::policy::Reliability                 | Reliability settings (@ref DCPS_QoS_Reliability "info")                             | Reliability::BestEffort()
 * dds::core::policy::DestinationOrder            | DataReader data order settings (@ref DCPS_QoS_DestinationOrder "info")              | DestinationOrder::ReceptionTimestamp()
 * dds::core::policy::History                     | Data storage settings (@ref DCPS_QoS_History "info")                                | History::KeepLast(depth 1)
 * dds::core::policy::ResourceLimits              | Maximum resource settings (@ref DCPS_QoS_ResourceLimits "info")                     | ResourceLimits::ResourceLimits(all unlimited)
 * dds::core::policy::UserData                    | Additional information (@ref DCPS_QoS_UserData "info")                              | UserData::UserData(empty)
 * dds::core::policy::Ownership                   | Exclusive ownership or not (@ref DCPS_QoS_Ownership "info")                         | Ownership::Shared()
 * dds::core::policy::TimeBasedFilter             | Maximum data rate (@ref DCPS_QoS_TimeBasedFilter "info")                            | TimeBasedFilter::TimeBasedFilter(0)
 * dds::core::policy::ReaderDataLifecycle         | Instance state changes and notifications (@ref DCPS_QoS_ReaderDataLifecycle "info") | ReaderDataLifecycle::NoAutoPurgeDisposedSamples()
 * dds::core::policy::DataRepresentation          | Supported data representation kinds (@ref DCPS_QoS_DataRepresentation "info")  | DataRepresentation::DataRepresentation(dds::core::policy::DataRepresentationId::XCDR1)
 * dds::core::policy::TypeConsistencyEnforcement | Type consistency enforcement policies (@ref DCPS_QoS_TypeConsistencyEnforcement "info") | dds::core::policy::TypeConsistencyKind::DISALLOW_TYPE_COERCION
 *
 * A QosPolicy can be set when the DataReader is created or modified with the set
 * qos operation.
 * Both operations take the DataReaderQos object as a parameter. There may be cases
 * where several policies are in conflict. Consistency checking is performed each time
 * the policies are modified when they are being created and, in case they are already
 * enabled, via the set qos operation.
 *
 * Some QosPolicy have "immutable" semantics meaning that they can only be
 * specified either at DataReader creation time or prior to calling the enable
 * operation on the DataReader.
 *
 * @see for more information: @ref DCPS_QoS
 */
class dds::sub::qos::DataReaderQos : public ::dds::core::EntityQos<org::eclipse::cyclonedds::sub::qos::DataReaderQosDelegate>
{
public:
    /**
     * Create a @ref anchor_dds_sub_datareader_qos_defaults "default" QoS.
     */
    DataReaderQos() {}

    /**
     * Create copied QoS.
     *
     * @param qos the QoS to copy policies from.
     */
    DataReaderQos(const DataReaderQos& qos);

    /**
     * Create a DataReader QoS from a TopicQos.
     *
     * This operation will copy the QosPolicy settings from the TopicQos to the
     * corresponding QosPolicy settings in the DataReaderQos. The related value
     * in DataReaderQos will be repliced, while the other policies will get the
     * @ref anchor_dds_sub_datareader_qos_defaults "default" QoS policies.
     *
     * This is a "convenience" operation. It can be used to merge
     * @ref anchor_dds_sub_datareader_qos_defaults "default" DataReader
     * QosPolicy settings with the corresponding ones on the Topic. The resulting
     * DataReaderQos can then be used to create a new DataReader, or set its
     * DataReaderQos.
     * @code{.cpp}
     * dds::topic::qos::TopicQos topicQos = topic.qos();
     * dds::sub::qos::DataReaderQos readerQos(topicQos);
     * // Policies of the DataReaderQos that are not present in the TopicQos
     * // have the default value.
     * @endcode
     *
     * This operation does not check the resulting DataReaderQos for self
     * consistency. This is because the "merged" DataReaderQos may not be the
     * final one, as the application can still modify some QosPolicy settings prior to
     * applying the DataReaderQos to the DataReader.
     *
     * @param qos the QoS to copy policies from.
     */
    DataReaderQos(const dds::topic::qos::TopicQos& qos);

    /**
     * Assign dds::topic::qos::TopicQos policies to the DataReaderQos.
     *
     * This operation will copy the QosPolicy settings from the TopicQos to the
     * corresponding QosPolicy settings in the DataReaderQos (replacing the values,
     * if present).
     *
     * This is a "convenience" operation, useful in combination with the operations
     * Subscriber::default_datareader_qos() and dds::topic::Topic::qos().
     * This operation can be used to merge the DataReader
     * QosPolicy settings with the corresponding ones on the Topic. The resulting
     * DataReaderQos can then be used to create a new DataReader, or set its
     * DataReaderQos.
     * @code{.cpp}
     * dds::topic::qos::TopicQos topicQos = topic.qos();
     * dds::sub::qos::DataReaderQos readerQos = subscriber.default_datareader_qos();
     * readerQos = topicQos;
     * // Policies of the DataReaderQos that are not present in the TopicQos are untouched.
     * @endcode
     *
     * This operation does not check the resulting DataReaderQos for self
     * consistency. This is because the "merged" DataReaderQos may not be the
     * final one, as the application can still modify some QosPolicy settings prior to
     * applying the DataReaderQos to the DataReader.
     *
     * @param qos the QoS to copy policies from.
     */
    DataReaderQos& operator= (const dds::topic::qos::TopicQos& other);
};

#else /* DOXYGEN_FOR_ISOCPP */

namespace dds { namespace sub { namespace qos { namespace detail {
	typedef ::dds::core::TEntityQos< ::org::eclipse::cyclonedds::sub::qos::DataReaderQosDelegate > DataReaderQos;
} } } }

#endif /* DOXYGEN_FOR_ISOCPP */

#endif /* OMG_DDS_SUB_QOS_DETAIL_DATA_READER_QOS_HPP_ */
