/*
 * Copyright(c) 2006 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

#ifndef DDS_QOSDEFS_H
#define DDS_QOSDEFS_H

#include <stdint.h>

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @anchor DDS_LENGTH_UNLIMITED
 * @ingroup qos
 * @brief Used for indicating unlimited length in dds_qset_resource_limits()
 */
#define DDS_LENGTH_UNLIMITED -1

/**
 * @brief Qos Policy IDs
 * @ingroup internal
 * Used internally to mark the QoS policy type
 */
typedef enum dds_qos_policy_id {
  DDS_INVALID_QOS_POLICY_ID, /**< Invalid Policy */
  DDS_USERDATA_QOS_POLICY_ID, /**< Userdata policy dds_qset_userdata() */
  DDS_DURABILITY_QOS_POLICY_ID, /**< Durability policy dds_qset_durability() */
  DDS_PRESENTATION_QOS_POLICY_ID, /**< Presentation policy dds_qset_presentation() */
  DDS_DEADLINE_QOS_POLICY_ID, /**< Deadline policy dds_qset_deadline() */
  DDS_LATENCYBUDGET_QOS_POLICY_ID, /**< LatencyBudget policy dds_qset_latency_budget() */
  DDS_OWNERSHIP_QOS_POLICY_ID, /**< Ownership policy dds_qset_ownership() */
  DDS_OWNERSHIPSTRENGTH_QOS_POLICY_ID, /**< OwnershipStrength policy dds_qset_ownership_strength() */
  DDS_LIVELINESS_QOS_POLICY_ID, /**< Liveliness policy dds_qset_liveliness() */
  DDS_TIMEBASEDFILTER_QOS_POLICY_ID, /**< TimeBasedFilter policy dds_qset_time_based_filter() */
  DDS_PARTITION_QOS_POLICY_ID, /**< Partition policy dds_qset_partition() */
  DDS_RELIABILITY_QOS_POLICY_ID, /**< Reliability policy dds_qset_reliability() */
  DDS_DESTINATIONORDER_QOS_POLICY_ID, /**< DestinationOrder policy dds_qset_destination_order() */
  DDS_HISTORY_QOS_POLICY_ID, /**< History policy dds_qset_history() */
  DDS_RESOURCELIMITS_QOS_POLICY_ID, /**< ResourceLimits policy dds_qset_resource_limits() */
  DDS_ENTITYFACTORY_QOS_POLICY_ID, /**< EntityFactory policy */
  DDS_WRITERDATALIFECYCLE_QOS_POLICY_ID, /**< WriterDataLifecycle policy dds_qset_writer_data_lifecycle() */
  DDS_READERDATALIFECYCLE_QOS_POLICY_ID, /**< ReaderDataLifecycle policy dds_qset_reader_data_lifecycle() */
  DDS_TOPICDATA_QOS_POLICY_ID, /**< Topicdata policy dds_qset_topicdata() */
  DDS_GROUPDATA_QOS_POLICY_ID, /**< Groupdata policy dds_qset_groupdata() */
  DDS_TRANSPORTPRIORITY_QOS_POLICY_ID, /**< TransportPriority policy dds_qset_transport_priority() */
  DDS_LIFESPAN_QOS_POLICY_ID, /**< Livespan policy dds_qset_lifespan() */
  DDS_DURABILITYSERVICE_QOS_POLICY_ID, /**< DurabilityService policy dds_qset_durability_service() */
  DDS_PROPERTY_QOS_POLICY_ID, /**< Property policy dds_qset_property() */
  DDS_TYPE_CONSISTENCY_ENFORCEMENT_QOS_POLICY_ID, /**< TypeConsistencyEnforcement policy dds_qset_type_consistency_enforcements() */
  DDS_DATA_REPRESENTATION_QOS_POLICY_ID /**< DataRepresentation policy dds_qset_data_representation() */
} dds_qos_policy_id_t;


/**
 * @brief QoS datatype
 * @ingroup qos
 * QoS structure is opaque
 */
typedef struct dds_qos dds_qos_t;

/**
 * @brief Durability QoS: Applies to Topic, DataReader, DataWriter
 * @ingroup qos
 */
typedef enum dds_durability_kind
{
    DDS_DURABILITY_VOLATILE, /**< Volatile durability */
    DDS_DURABILITY_TRANSIENT_LOCAL, /**< Transient Local durability */
    DDS_DURABILITY_TRANSIENT, /**< Transient durability */
    DDS_DURABILITY_PERSISTENT /**< Persistent durability */
}
dds_durability_kind_t;

/**
 * @brief History QoS: Applies to Topic, DataReader, DataWriter
 * @ingroup qos
 */
typedef enum dds_history_kind
{
    DDS_HISTORY_KEEP_LAST, /**< Keep Last history */
    DDS_HISTORY_KEEP_ALL /**< Keep All history */
}
dds_history_kind_t;

/**
 * @brief Ownership QoS: Applies to Topic, DataReader, DataWriter
 * @ingroup qos
 */
typedef enum dds_ownership_kind
{
    DDS_OWNERSHIP_SHARED, /**< Shared Ownership */
    DDS_OWNERSHIP_EXCLUSIVE /**< Exclusive Ownership */
}
dds_ownership_kind_t;

/**
 * @brief Liveliness QoS: Applies to Topic, DataReader, DataWriter
 * @ingroup qos
 */
typedef enum dds_liveliness_kind
{
    DDS_LIVELINESS_AUTOMATIC, /**< Automatic liveliness */
    DDS_LIVELINESS_MANUAL_BY_PARTICIPANT, /**< Manual by Participant liveliness */
    DDS_LIVELINESS_MANUAL_BY_TOPIC /**< Manual by Topic liveliness */
}
dds_liveliness_kind_t;

/**
 * @brief Reliability QoS: Applies to Topic, DataReader, DataWriter
 * @ingroup qos
 */
typedef enum dds_reliability_kind
{
    DDS_RELIABILITY_BEST_EFFORT, /**< Best Effort reliability */
    DDS_RELIABILITY_RELIABLE /**< Reliable reliability */
}
dds_reliability_kind_t;

/**
 * @brief DestinationOrder QoS: Applies to Topic, DataReader, DataWriter
 * @ingroup qos
 */
typedef enum dds_destination_order_kind
{
    DDS_DESTINATIONORDER_BY_RECEPTION_TIMESTAMP, /**< order by reception timestamp */
    DDS_DESTINATIONORDER_BY_SOURCE_TIMESTAMP /**< order by source timestamp */
}
dds_destination_order_kind_t;

/**
 * @brief Presentation QoS: Applies to Publisher, Subscriber
 * @ingroup qos
 */
typedef enum dds_presentation_access_scope_kind
{
    DDS_PRESENTATION_INSTANCE, /**< presentation scope per instance */
    DDS_PRESENTATION_TOPIC, /**< presentation scope per topic */
    DDS_PRESENTATION_GROUP /**< presentation scope per group */
}
dds_presentation_access_scope_kind_t;

/**
 * @brief Ignore-local QoS: Applies to DataReader, DataWriter
 * @ingroup qos
 */
typedef enum dds_ignorelocal_kind
{
    DDS_IGNORELOCAL_NONE, /**< Don't ignore local data */
    DDS_IGNORELOCAL_PARTICIPANT, /**< Ignore local data from same participant */
    DDS_IGNORELOCAL_PROCESS /**< Ignore local data from same process */
}
dds_ignorelocal_kind_t;

/**
 * @brief Type-consistency QoS: Applies to DataReader, DataWriter
 * @ingroup qos
 */
typedef enum dds_type_consistency_kind
{
    DDS_TYPE_CONSISTENCY_DISALLOW_TYPE_COERCION, /**< Do not allow type coercion */
    DDS_TYPE_CONSISTENCY_ALLOW_TYPE_COERCION /**< Allow type coercion */
}
dds_type_consistency_kind_t;

/**
 * @brief Data Representation QoS: Applies to Topic, DataReader, DataWriter
 * @ingroup qos
 */
typedef int16_t dds_data_representation_id_t;

#if defined (__cplusplus)
}
#endif
#endif
