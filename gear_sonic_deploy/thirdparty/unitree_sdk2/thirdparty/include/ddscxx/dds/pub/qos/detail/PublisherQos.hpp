#ifndef OMG_DDS_PUB_QOS_DETAIL_PUBLISER_QOS_HPP_
#define OMG_DDS_PUB_QOS_DETAIL_PUBLISER_QOS_HPP_

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
#include <org/eclipse/cyclonedds/pub/qos/PublisherQosDelegate.hpp>

#ifdef DOXYGEN_FOR_ISOCPP
/* The above macro is never (and must never) be defined in normal compilation.
 *
 * The class below is just used to create informative API documentation.
 * The predoxygen.py script will copy this over the QoS API header file.
 */
/**
 * @brief
 * This object provides the basic mechanism for an application to specify Quality of
 * Service attributes for a Publisher.
 *
 * @par Attributes
 * @anchor anchor_dds_pub_publisher_qos_defaults
 * QosPolicy                               | Desciption                                                            | Default Value
 * --------------------------------------- | --------------------------------------------------------------------- | --------------------
 * dds::core::policy::Presentation         | Data-instance change dependencies (@ref DCPS_QoS_Presentation "info") | Presentation::InstanceAccessScope(coherent=false, ordered=false)
 * dds::core::policy::Partition            | Active partitions (@ref DCPS_QoS_Partition "info")                    | Partition::Partition(empty)
 * dds::core::policy::GroupData            | Additional information (@ref DCPS_QoS_GroupData "info")               | GroupData::GroupData(empty)
 * dds::core::policy::EntityFactory        | Create enabled (@ref DCPS_QoS_EntityFactory "info")                   | EntityFactory::AutoEnable()
 *
 * A QosPolicy can be set when the Publisher is created or modified with the set qos
 * operation.
 * Both operations take the PublisherQos object as a parameter. There may be cases where
 * several policies are in conflict. Consistency checking is performed each time the
 * policies are modified when they are being created and, in case they are already
 * enabled, via the set qos operation.
 *
 * Some QosPolicy have "immutable" semantics meaning that they can only be
 * specified either at Publisher creation time or prior to calling the enable operation
 * on the Publisher.
 *
 * @see for more information: @ref DCPS_QoS
 */
class dds::pub::qos::PublisherQos : public ::dds::core::EntityQos<org::eclipse::cyclonedds::pub::qos::PublisherQosDelegate>
{
public:
    /**
     * Create @ref anchor_dds_pub_publisher_qos_defaults "default" QoS.
     */
    PublisherQos() {}

    /**
     * Create copied QoS type.
     *
     * @param qos the QoS to copy policies from.
     */
    PublisherQos(const PublisherQos& qos);
};

#else /* DOXYGEN_FOR_ISOCPP */

namespace dds { namespace pub { namespace qos { namespace detail {
	typedef ::dds::core::TEntityQos< ::org::eclipse::cyclonedds::pub::qos::PublisherQosDelegate> PublisherQos;
} } } }

#endif /* DOXYGEN_FOR_ISOCPP */

#endif /* OMG_DDS_PUB_QOS_DETAIL_PUBLISER_QOS_HPP_ */
