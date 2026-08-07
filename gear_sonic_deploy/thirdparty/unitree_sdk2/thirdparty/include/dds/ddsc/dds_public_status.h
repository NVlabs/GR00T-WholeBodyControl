/*
 * Copyright(c) 2006 to 2019 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

/**
 * @defgroup dcps_status (DDS C Communication Status API)
 * @ingroup dds
 * This defines the public API of the Communication Status in the
 * Eclipse Cyclone DDS C language binding. Listeners are implemented
 * as structs containing callback functions that take listener status types
 * as arguments.
 */
#ifndef DDS_STATUS_H
#define DDS_STATUS_H

#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_OfferedDeadlineMissed
 * DOC_TODO
 */
typedef struct dds_offered_deadline_missed_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
  dds_instance_handle_t last_instance_handle;  /**< DOC_TODO */
}
dds_offered_deadline_missed_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_OfferedIncompatibleQoS
 * DOC_TODO
 */
typedef struct dds_offered_incompatible_qos_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
  uint32_t last_policy_id;  /**< DOC_TODO */
}
dds_offered_incompatible_qos_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_PublicationMatched
 * DOC_TODO
 */
typedef struct dds_publication_matched_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
  uint32_t current_count;  /**< DOC_TODO */
  int32_t current_count_change;  /**< DOC_TODO */
  dds_instance_handle_t last_subscription_handle;  /**< DOC_TODO */
}
dds_publication_matched_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_LivelinessLost
 * DOC_TODO
 */
typedef struct dds_liveliness_lost_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
}
dds_liveliness_lost_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_SubscriptionMatched
 * DOC_TODO
 */
typedef struct dds_subscription_matched_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
  uint32_t current_count;  /**< DOC_TODO */
  int32_t current_count_change;  /**< DOC_TODO */
  dds_instance_handle_t last_publication_handle;  /**< DOC_TODO */
}
dds_subscription_matched_status_t;

/**
 * @ingroup dcps_status
 * @brief Rejected Status
 * DOC_TODO
 */
typedef enum
{
  DDS_NOT_REJECTED,  /**< DOC_TODO */
  DDS_REJECTED_BY_INSTANCES_LIMIT,  /**< DOC_TODO */
  DDS_REJECTED_BY_SAMPLES_LIMIT,  /**< DOC_TODO */
  DDS_REJECTED_BY_SAMPLES_PER_INSTANCE_LIMIT  /**< DOC_TODO */
}
dds_sample_rejected_status_kind;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_SampleRejected
 * DOC_TODO
 */
typedef struct dds_sample_rejected_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
  dds_sample_rejected_status_kind last_reason;  /**< DOC_TODO */
  dds_instance_handle_t last_instance_handle;  /**< DOC_TODO */
}
dds_sample_rejected_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_LivelinessChanged
 * DOC_TODO
 */
typedef struct dds_liveliness_changed_status
{
  uint32_t alive_count;  /**< DOC_TODO */
  uint32_t not_alive_count;  /**< DOC_TODO */
  int32_t alive_count_change;  /**< DOC_TODO */
  int32_t not_alive_count_change;  /**< DOC_TODO */
  dds_instance_handle_t last_publication_handle;  /**< DOC_TODO */
}
dds_liveliness_changed_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_RequestedDeadlineMissed
 * DOC_TODO
 */
typedef struct dds_requested_deadline_missed_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
  dds_instance_handle_t last_instance_handle;  /**< DOC_TODO */
}
dds_requested_deadline_missed_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_RequestedIncompatibleQoS
 * DOC_TODO
 */
typedef struct dds_requested_incompatible_qos_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
  uint32_t last_policy_id;  /**< DOC_TODO */
}
dds_requested_incompatible_qos_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_SampleLost
 * DOC_TODO
 */
typedef struct dds_sample_lost_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
}
dds_sample_lost_status_t;

/**
 * @ingroup dcps_status
 * @brief DCPS_Status_InconsistentTopic
 * DOC_TODO
 */
typedef struct dds_inconsistent_topic_status
{
  uint32_t total_count;  /**< DOC_TODO */
  int32_t total_count_change;  /**< DOC_TODO */
}
dds_inconsistent_topic_status_t;


/**
 * @defgroup dcps_status_getters (DCPS Status Getters)
 * @ingroup dcps_status
 * get_<status> APIs return the status of an entity and resets the status
*/

/**
 * @ingroup dcps_status_getters
 * @brief Get INCONSISTENT_TOPIC status
 *
 * This operation gets the status value corresponding to INCONSISTENT_TOPIC
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  topic  The entity to get the status
 * @param[out] status The pointer to @ref dds_inconsistent_topic_status_t to get the status
 *
 * @returns  0 - Success
 * @returns <0 - Failure
 *
 * @retval DDS_RETCODE_ERROR
 *                  An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *                  One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *                  The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *                  The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_inconsistent_topic_status (
  dds_entity_t topic,
  dds_inconsistent_topic_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get PUBLICATION_MATCHED status
 *
 * This operation gets the status value corresponding to PUBLICATION_MATCHED
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  writer  The entity to get the status
 * @param[out] status  The pointer to @ref dds_publication_matched_status_t to get the status
 *
 * @returns  0 - Success
 * @returns <0 - Failure
 *
 * @retval DDS_RETCODE_ERROR
 *                  An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *                  One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *                  The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *                  The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_publication_matched_status (
  dds_entity_t writer,
  dds_publication_matched_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get LIVELINESS_LOST status
 *
 * This operation gets the status value corresponding to LIVELINESS_LOST
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  writer  The entity to get the status
 * @param[out] status  The pointer to @ref dds_liveliness_lost_status_t to get the status
 *
 * @returns  0 - Success
 * @returns <0 - Failure
 *
 * @retval DDS_RETCODE_ERROR
 *                  An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *                  One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *                  The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *                  The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_liveliness_lost_status (
  dds_entity_t writer,
  dds_liveliness_lost_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get OFFERED_DEADLINE_MISSED status
 *
 * This operation gets the status value corresponding to OFFERED_DEADLINE_MISSED
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  writer  The entity to get the status
 * @param[out] status  The pointer to @ref dds_offered_deadline_missed_status_t to get the status
 *
 * @returns  0 - Success
 * @returns <0 - Failure
 *
 * @retval DDS_RETCODE_ERROR
 *                  An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *                  One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *                  The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *                  The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_offered_deadline_missed_status(
  dds_entity_t writer,
  dds_offered_deadline_missed_status_t *status);

/**
 * @ingroup dcps_status_getters
 * @brief Get OFFERED_INCOMPATIBLE_QOS status
 *
 * This operation gets the status value corresponding to OFFERED_INCOMPATIBLE_QOS
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  writer  The writer entity to get the status
 * @param[out] status  The pointer to @ref dds_offered_incompatible_qos_status_t to get the status
 *
 * @returns  0 - Success
 * @returns <0 - Failure
 *
 * @retval DDS_RETCODE_ERROR
 *                  An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *                  One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *                  The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *                  The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_offered_incompatible_qos_status (
  dds_entity_t writer,
  dds_offered_incompatible_qos_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get SUBSCRIPTION_MATCHED status
 *
 * This operation gets the status value corresponding to SUBSCRIPTION_MATCHED
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  reader  The reader entity to get the status
 * @param[out] status  The pointer to @ref dds_subscription_matched_status_t to get the status
 *
 * @returns  0 - Success
 * @returns <0 - Failure
 *
 * @retval DDS_RETCODE_ERROR
 *                  An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *                  One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *                  The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *                  The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_subscription_matched_status (
  dds_entity_t reader,
  dds_subscription_matched_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get LIVELINESS_CHANGED status
 *
 * This operation gets the status value corresponding to LIVELINESS_CHANGED
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  reader  The entity to get the status
 * @param[out] status  The pointer to @ref dds_liveliness_changed_status_t to get the status
 *
 * @returns  0 - Success
 * @returns <0 - Failure
 *
 * @retval DDS_RETCODE_ERROR
 *                  An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *                  One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *                  The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *                  The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_liveliness_changed_status (
  dds_entity_t reader,
  dds_liveliness_changed_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get SAMPLE_REJECTED status
 *
 * This operation gets the status value corresponding to SAMPLE_REJECTED
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  reader  The entity to get the status
 * @param[out] status  The pointer to @ref dds_sample_rejected_status_t to get the status
 *
 * @returns  0 - Success
 * @returns <0 - Failure
 *
 * @retval DDS_RETCODE_ERROR
 *                  An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *                  One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *                  The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *                  The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_sample_rejected_status (
  dds_entity_t reader,
  dds_sample_rejected_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get SAMPLE_LOST status
 *
 * This operation gets the status value corresponding to SAMPLE_LOST
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  reader  The entity to get the status
 * @param[out] status  The pointer to @ref dds_sample_lost_status_t to get the status
 *
 * @returns A dds_return_t indicating success or failure
 *
 * @retval DDS_RETCODE_OK
 *            Success
 * @retval DDS_RETCODE_ERROR
 *            An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *            One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *            The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *            The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_sample_lost_status (
  dds_entity_t reader,
  dds_sample_lost_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get REQUESTED_DEADLINE_MISSED status
 *
 * This operation gets the status value corresponding to REQUESTED_DEADLINE_MISSED
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  reader  The entity to get the status
 * @param[out] status  The pointer to @ref dds_requested_deadline_missed_status_t to get the status
 *
 * @returns A dds_return_t indicating success or failure
 *
 * @retval DDS_RETCODE_OK
 *            Success
 * @retval DDS_RETCODE_ERROR
 *            An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *            One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *            The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *            The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_requested_deadline_missed_status (
  dds_entity_t reader,
  dds_requested_deadline_missed_status_t * status);

/**
 * @ingroup dcps_status_getters
 * @brief Get REQUESTED_INCOMPATIBLE_QOS status
 *
 * This operation gets the status value corresponding to REQUESTED_INCOMPATIBLE_QOS
 * and reset the status. The value can be obtained, only if the status is enabled for an entity.
 * NULL value for status is allowed and it will reset the trigger value when status is enabled.
 *
 * @param[in]  reader  The entity to get the status
 * @param[out] status  The pointer to @ref dds_requested_incompatible_qos_status_t to get the status
 *
 * @returns A dds_return_t indicating success or failure
 *
 * @retval DDS_RETCODE_OK
 *            Success
 * @retval DDS_RETCODE_ERROR
 *            An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *            One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *            The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *            The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_requested_incompatible_qos_status (
  dds_entity_t reader,
  dds_requested_incompatible_qos_status_t * status);

#if defined (__cplusplus)
}
#endif
#endif
