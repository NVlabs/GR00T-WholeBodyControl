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

/**
 * @defgroup qos (DDS C QoS API)
 * @ingroup dds
 *
 * This defines the public API of QoS and Policies in the
 * Eclipse Cyclone DDS C language binding.
 */
#ifndef DDS_QOS_H
#define DDS_QOS_H

#include "dds/export.h"
#include "dds/ddsc/dds_public_qosdefs.h"

/**
 * @anchor DDS_HAS_PROPERTY_LIST_QOS
 * @ingroup qos
 * @brief Whether or not the "property list" QoS setting is supported in this version.  If it is,
 * the "dds.sec." properties are treated specially, preventing the accidental creation of
 * an non-secure participant by an implementation built without support for DDS Security.
 */
#define DDS_HAS_PROPERTY_LIST_QOS 1

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @ingroup qos
 * @brief Allocate memory and initialize default QoS-policies
 *
 * @returns - Pointer to the initialized dds_qos_t structure, NULL if unsuccessful.
 */
DDS_EXPORT
dds_qos_t * dds_create_qos (void);

/**
 * @ingroup deprecated
 * @deprecated use \ref dds_create_qos instead.
 * Allocate memory and initialize default QoS-policies
 *
 * @returns - Pointer to the initialized dds_qos_t structure, NULL if unsuccessful.
 */
DDS_DEPRECATED_EXPORT
dds_qos_t * dds_qos_create (void);

/**
 * @ingroup qos
 * @brief Delete memory allocated to QoS-policies structure
 *
 * @param[in] qos - Pointer to dds_qos_t structure
 */
DDS_EXPORT void
dds_delete_qos (dds_qos_t * __restrict qos);

DDS_DEPRECATED_EXPORT void
dds_qos_delete (dds_qos_t * __restrict qos);

/**
 * @ingroup qos
 * @brief Reset a QoS-policies structure to default values
 *
 * @param[in,out] qos - Pointer to the dds_qos_t structure
 */
DDS_EXPORT void
dds_reset_qos(dds_qos_t * __restrict qos);

DDS_DEPRECATED_EXPORT
void dds_qos_reset (dds_qos_t * __restrict qos
);

/**
 * @ingroup qos
 * @brief Copy all QoS-policies from one structure to another
 *
 * @param[in,out] dst - Pointer to the destination dds_qos_t structure
 * @param[in] src - Pointer to the source dds_qos_t structure
 *
 * @returns - Return-code indicating success or failure
 */
DDS_EXPORT dds_return_t
dds_copy_qos (dds_qos_t * __restrict dst, const dds_qos_t * __restrict src);

DDS_DEPRECATED_EXPORT dds_return_t
dds_qos_copy (dds_qos_t * __restrict dst, const dds_qos_t * __restrict src);

/**
 * @ingroup qos
 * @brief Copy all QoS-policies from one structure to another, unless already set
 *
 * Policies are copied from src to dst, unless src already has the policy set to a non-default value.
 *
 * @param[in,out] dst - Pointer to the destination qos structure
 * @param[in] src - Pointer to the source qos structure
 */
DDS_EXPORT void
dds_merge_qos (dds_qos_t * __restrict dst, const dds_qos_t * __restrict src);

DDS_DEPRECATED_EXPORT void
dds_qos_merge (dds_qos_t * __restrict dst, const dds_qos_t * __restrict src);

/**
 * @ingroup qos
 * @brief Copy all QoS-policies from one structure to another, unless already set
 *
 * Policies are copied from src to dst, unless src already has the policy set to a non-default value.
 *
 * @param[in,out] a - Pointer to the destination qos structure
 * @param[in] b - Pointer to the source qos structure
 *
 * @returns whether the copy was successful.
 */
DDS_EXPORT bool
dds_qos_equal (const dds_qos_t * __restrict a, const dds_qos_t * __restrict b);

/**
 * @defgroup qos_setters (Qos Setters)
 * @ingroup qos
 */

/**
 * @ingroup qos_setters
 * @brief Set the userdata of a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the userdata
 * @param[in] value - Pointer to the userdata
 * @param[in] sz - Size of userdata stored in value
 */
DDS_EXPORT void
dds_qset_userdata (
  dds_qos_t * __restrict qos,
  const void * __restrict value,
  size_t sz);

/**
 * @ingroup qos_setters
 * @brief Set the topicdata of a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the topicdata
 * @param[in] value - Pointer to the topicdata
 * @param[in] sz - Size of the topicdata stored in value
 */
DDS_EXPORT void
dds_qset_topicdata (
  dds_qos_t * __restrict qos,
  const void * __restrict value,
  size_t sz);

/**
 * @ingroup qos_setters
 * @brief Set the groupdata of a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the groupdata
 * @param[in] value - Pointer to the group data
 * @param[in] sz - Size of groupdata stored in value
 */
DDS_EXPORT void
dds_qset_groupdata (
  dds_qos_t * __restrict qos,
  const void * __restrict value,
  size_t sz);

/**
 * @ingroup qos_setters
 * @brief Set the durability policy of a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] kind - Durability kind value
 */
DDS_EXPORT void
dds_qset_durability (dds_qos_t * __restrict qos, dds_durability_kind_t kind);

/**
 * @ingroup qos_setters
 * @brief Set the history policy of a qos structure.
 *
 * Note that depth is only relevant for keep last. If you want limited history for keep all, use dds_qset_resource_limits().
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] kind - History kind value
 * @param[in] depth - History depth value
 */
DDS_EXPORT void
dds_qset_history (
  dds_qos_t * __restrict qos,
  dds_history_kind_t kind,
  int32_t depth);

/**
 * @ingroup qos_setters
 * @brief Set the resource limits policy of a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] max_samples - Number of samples resource-limit value
 * @param[in] max_instances - Number of instances resource-limit value
 * @param[in] max_samples_per_instance - Number of samples per instance resource-limit value
 */
DDS_EXPORT void
dds_qset_resource_limits (
  dds_qos_t * __restrict qos,
  int32_t max_samples,
  int32_t max_instances,
  int32_t max_samples_per_instance);

/**
 * @ingroup qos_setters
 * @brief Set the presentation policy of a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] access_scope - Access-scope kind
 * @param[in] coherent_access - Coherent access enable value
 * @param[in] ordered_access - Ordered access enable value
 */
DDS_EXPORT void
dds_qset_presentation (
  dds_qos_t * __restrict qos,
  dds_presentation_access_scope_kind_t access_scope,
  bool coherent_access,
  bool ordered_access);

/**
 * @ingroup qos_setters
 * @brief Set the lifespan policy of a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] lifespan - Lifespan duration (expiration time relative to source timestamp of a sample)
 */
DDS_EXPORT void
dds_qset_lifespan (
  dds_qos_t * __restrict qos,
  dds_duration_t lifespan);

/**
 * @ingroup qos_setters
 * @brief Set the deadline policy of a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] deadline - Deadline duration
 */
DDS_EXPORT void
dds_qset_deadline (
  dds_qos_t * __restrict qos,
  dds_duration_t deadline);

/**
 * @ingroup qos_setters
 * @brief Set the latency-budget policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] duration - Latency budget duration
 */
DDS_EXPORT void
dds_qset_latency_budget (
  dds_qos_t * __restrict qos,
  dds_duration_t duration);

/**
 * @ingroup qos_setters
 * @brief Set the ownership policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] kind - Ownership kind
 */
DDS_EXPORT void
dds_qset_ownership (
  dds_qos_t * __restrict qos,
  dds_ownership_kind_t kind);

/**
 * @ingroup qos_setters
 * @brief Set the ownership strength policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] value - Ownership strength value
 */
DDS_EXPORT void
dds_qset_ownership_strength (dds_qos_t * __restrict qos, int32_t value);

/**
 * @ingroup qos_setters
 * @brief Set the liveliness policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] kind - Liveliness kind
 * @param[in] lease_duration - Lease duration
 */
DDS_EXPORT void
dds_qset_liveliness (
  dds_qos_t * __restrict qos,
  dds_liveliness_kind_t kind,
  dds_duration_t lease_duration);

/**
 * @ingroup qos_setters
 * @brief Set the time-based filter policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] minimum_separation - Minimum duration between sample delivery for an instance
 */
DDS_EXPORT void
dds_qset_time_based_filter (
  dds_qos_t * __restrict qos,
  dds_duration_t minimum_separation);

/**
 * @ingroup qos_setters
 * @brief Set the partition policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] n - Number of partitions stored in ps
 * @param[in] ps - Pointer to string(s) storing partition name(s)
 */
DDS_EXPORT void
dds_qset_partition (
  dds_qos_t * __restrict qos,
  uint32_t n,
  const char ** __restrict ps);

/**
 * @ingroup qos_setters
 * @brief Convenience function to set the partition policy of a qos structure to a
 * single name.  Name may be a null pointer.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] name - Pointer to the name
 */
DDS_EXPORT void
dds_qset_partition1 (
  dds_qos_t * __restrict qos,
  const char * __restrict name);

/**
 * @ingroup qos_setters
 * @brief Set the reliability policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] kind - Reliability kind
 * @param[in] max_blocking_time - Max blocking duration applied when kind is reliable. This is how long the writer will block when its history is full.
 */
DDS_EXPORT void
dds_qset_reliability (
  dds_qos_t * __restrict qos,
  dds_reliability_kind_t kind,
  dds_duration_t max_blocking_time);

/**
 * @ingroup qos_setters
 * @brief Set the transport-priority policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] value - Priority value
 */
DDS_EXPORT void
dds_qset_transport_priority (dds_qos_t * __restrict qos, int32_t value);

/**
 * @ingroup qos_setters
 * @brief Set the destination-order policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] kind - Destination-order kind
 */
DDS_EXPORT void
dds_qset_destination_order (
  dds_qos_t * __restrict qos,
  dds_destination_order_kind_t kind);

/**
 * @ingroup qos_setters
 * @brief Set the writer data-lifecycle policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] autodispose - Automatic disposal of unregistered instances
 */
DDS_EXPORT void
dds_qset_writer_data_lifecycle (dds_qos_t * __restrict qos, bool autodispose);

/**
 * @ingroup qos_setters
 * @brief Set the reader data-lifecycle policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] autopurge_nowriter_samples_delay - Delay for purging of samples from instances in a no-writers state
 * @param[in] autopurge_disposed_samples_delay - Delay for purging of samples from disposed instances
 */
DDS_EXPORT void
dds_qset_reader_data_lifecycle (
  dds_qos_t * __restrict qos,
  dds_duration_t autopurge_nowriter_samples_delay,
  dds_duration_t autopurge_disposed_samples_delay);

/**
 * @ingroup qos_setters
 * @brief Set the durability-service policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] service_cleanup_delay - Delay for purging of abandoned instances from the durability service
 * @param[in] history_kind - History policy kind applied by the durability service
 * @param[in] history_depth - History policy depth applied by the durability service
 * @param[in] max_samples - Number of samples resource-limit policy applied by the durability service
 * @param[in] max_instances - Number of instances resource-limit policy applied by the durability service
 * @param[in] max_samples_per_instance - Number of samples per instance resource-limit policy applied by the durability service
 */
DDS_EXPORT void
dds_qset_durability_service (
  dds_qos_t * __restrict qos,
  dds_duration_t service_cleanup_delay,
  dds_history_kind_t history_kind,
  int32_t history_depth,
  int32_t max_samples,
  int32_t max_instances,
  int32_t max_samples_per_instance);

/**
 * @ingroup qos_setters
 * @brief Set the ignore-local policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] ignore - True if readers and writers owned by the same participant should be ignored
 */
DDS_EXPORT void
dds_qset_ignorelocal (
  dds_qos_t * __restrict qos,
  dds_ignorelocal_kind_t ignore);

/**
 * @ingroup qos_setters
 * @brief Stores a property with the provided name and string value in a qos structure.
 *
 * In the case a property with the provided name already exists in the qos structure,
 * the value for this entry is overwritten with the provided string value. If more than
 * one property with the provided name exists, only the value of the first of these
 * properties is updated.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the property
 * @param[in] name - Pointer to name of the property
 * @param[in] value - Pointer to a (null-terminated) string that will be stored
 */
DDS_EXPORT void
dds_qset_prop (
  dds_qos_t * __restrict qos,
  const char * name,
  const char * value);

/**
 * @ingroup qos_setters
 * @brief Removes the property with the provided name from a qos structure.
 *
 * In case more than one property exists with this name, only the first property
 * is removed.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that contains the property
 * @param[in] name - Pointer to name of the property
 */
DDS_EXPORT void
dds_qunset_prop (
  dds_qos_t * __restrict qos,
  const char * name);

/**
 * @ingroup qos_setters
 * @brief Stores the provided binary data as a property in a qos structure
 *
 * In the case a property with the provided name already exists in the qos structure,
 * the value for this entry is overwritten with the provided data. If more than one
 * property with the provided name exists, only the value of the first of these
 * properties is updated.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the property
 * @param[in] name - Pointer to name of the property
 * @param[in] value - Pointer to data to be stored in the property
 * @param[in] sz - Size of the data
 */
DDS_EXPORT void
dds_qset_bprop (
  dds_qos_t * __restrict qos,
  const char * name,
  const void * value,
  const size_t sz);

/**
 * @ingroup qos_setters
 * @brief Removes the binary property with the provided name from a qos structure.
 *
 * In case more than one binary property exists with this name, only the first binary
 * property is removed.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that contains the binary property
 * @param[in] name - Pointer to name of the property
 */
DDS_EXPORT void
dds_qunset_bprop (
  dds_qos_t * __restrict qos,
  const char * name);

/**
 * @ingroup qos_setters
 * @brief Set the type consistency enforcement policy of a qos structure
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the policy
 * @param[in] kind - Type consistency policy kind
 * @param[in] ignore_sequence_bounds - Ignore sequence bounds in type assignability checking
 * @param[in] ignore_string_bounds - Ignore string bounds in type assignability checking
 * @param[in] ignore_member_names - Ignore member names in type assignability checking
 * @param[in] prevent_type_widening - Prevent type widening in type assignability checking
 * @param[in] force_type_validation - Force type validation in assignability checking
 */
DDS_EXPORT void
dds_qset_type_consistency (
  dds_qos_t * __restrict qos,
  dds_type_consistency_kind_t kind,
  bool ignore_sequence_bounds,
  bool ignore_string_bounds,
  bool ignore_member_names,
  bool prevent_type_widening,
  bool force_type_validation);

/**
 * @ingroup qos_setters
 * @brief Set the data representation of a qos structure
 *
 * @param[in,out] qos    - Pointer to a dds_qos_t structure that will store the policy
 * @param[in]     n      - Number of data representation values
 * @param[in]     values - Data representation values
 */
DDS_EXPORT void
dds_qset_data_representation (
  dds_qos_t * __restrict qos,
  uint32_t n,
  const dds_data_representation_id_t *values);

/**
 * @ingroup qos_setters
 * @brief Set the entity name.
 *
 * When using this QoS to initialize a participant, publisher, subscriber, reader or writer
 * it will take the name set here. This name is visible over discovery and can be used
 * to make sense of network in tooling.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that will store the entity name.
 * @param[in] name - Pointer to the entity name to set.
 */
DDS_EXPORT void
dds_qset_entity_name (
  dds_qos_t * __restrict qos,
  const char * name);


/**
 * @defgroup qos_getters (QoS Getters)
 * @ingroup qos
 */

/**
 * @ingroup qos_getters
 * @brief Get the userdata from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] value - Pointer that will store the userdata.  If sz = 0, then a null pointer, else it is a pointer to an allocated buffer of sz+1 bytes where the last byte is always 0
 * @param[in,out] sz - Pointer that will store the size of userdata
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool dds_qget_userdata (const dds_qos_t * __restrict qos, void **value, size_t *sz);

/**
 * @ingroup qos_getters
 * @brief Get the topicdata from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] value - Pointer that will store the topicdata.  If sz = 0, then a null pointer, else it is a pointer to an allocated buffer of sz+1 bytes where the last byte is always 0
 * @param[in,out] sz - Pointer that will store the size of topicdata
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
*/
DDS_EXPORT bool dds_qget_topicdata (const dds_qos_t * __restrict qos, void **value, size_t *sz);

/**
 * @ingroup qos_getters
 * @brief Get the groupdata from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] value - Pointer that will store the groupdata.  If sz = 0, then a null pointer, else it is a pointer to an allocated buffer of sz+1 bytes where the last byte is always 0
 * @param[in,out] sz - Pointer that will store the size of groupdata
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool dds_qget_groupdata (const dds_qos_t * __restrict qos, void **value, size_t *sz);

/**
 * @ingroup qos_getters
 * @brief Get the durability policy from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] kind - Pointer that will store the durability kind
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool dds_qget_durability (const dds_qos_t * __restrict qos, dds_durability_kind_t *kind);

/**
 * @ingroup qos_getters
 * @brief Get the history policy from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] kind - Pointer that will store the history kind (optional)
 * @param[in,out] depth - Pointer that will store the history depth (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool dds_qget_history (const dds_qos_t * __restrict qos, dds_history_kind_t *kind, int32_t *depth);

/**
 * @ingroup qos_getters
 * @brief Get the resource-limits policy from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] max_samples - Pointer that will store the number of samples resource-limit (optional)
 * @param[in,out] max_instances - Pointer that will store the number of instances resource-limit (optional)
 * @param[in,out] max_samples_per_instance - Pointer that will store the number of samples per instance resource-limit (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_resource_limits (
  const dds_qos_t * __restrict qos,
  int32_t *max_samples,
  int32_t *max_instances,
  int32_t *max_samples_per_instance);

/**
 * @ingroup qos_getters
 * @brief Get the presentation policy from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] access_scope - Pointer that will store access scope kind (optional)
 * @param[in,out] coherent_access - Pointer that will store coherent access enable value (optional)
 * @param[in,out] ordered_access - Pointer that will store orderede access enable value (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_presentation (
  const dds_qos_t * __restrict qos,
  dds_presentation_access_scope_kind_t *access_scope,
  bool *coherent_access,
  bool *ordered_access);

/**
 * @ingroup qos_getters
 * @brief Get the lifespan policy from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] lifespan - Pointer that will store lifespan duration
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_lifespan (
  const dds_qos_t * __restrict qos,
  dds_duration_t *lifespan);

/**
 * @ingroup qos_getters
 * @brief Get the deadline policy from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] deadline - Pointer that will store deadline duration
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_deadline (
  const dds_qos_t * __restrict qos,
  dds_duration_t *deadline);

/**
 * @ingroup qos_getters
 * @brief Get the latency-budget policy from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] duration - Pointer that will store latency-budget duration
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_latency_budget (
  const dds_qos_t * __restrict qos,
  dds_duration_t *duration);

/**
 * @ingroup qos_getters
 * @brief Get the ownership policy from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] kind - Pointer that will store the ownership kind
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_ownership (
  const dds_qos_t * __restrict qos,
  dds_ownership_kind_t *kind);

/**
 * @ingroup qos_getters
 * @brief Get the ownership strength qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] value - Pointer that will store the ownership strength value
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_ownership_strength (
  const dds_qos_t * __restrict qos,
  int32_t *value);

/**
 * @ingroup qos_getters
 * @brief Get the liveliness qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] kind - Pointer that will store the liveliness kind (optional)
 * @param[in,out] lease_duration - Pointer that will store the liveliness lease duration (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_liveliness (
  const dds_qos_t * __restrict qos,
  dds_liveliness_kind_t *kind,
  dds_duration_t *lease_duration);

/**
 * @ingroup qos_getters
 * @brief Get the time-based filter qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] minimum_separation - Pointer that will store the minimum separation duration (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_time_based_filter (
  const dds_qos_t * __restrict qos,
  dds_duration_t *minimum_separation);

/**
 * @ingroup qos_getters
 * @brief Get the partition qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] n - Pointer that will store the number of partitions (optional)
 * @param[in,out] ps - Pointer that will store the string(s) containing partition name(s) (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_partition (
  const dds_qos_t * __restrict qos,
  uint32_t *n,
  char ***ps);

/**
 * @ingroup qos_getters
 * @brief Get the reliability qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] kind - Pointer that will store the reliability kind (optional)
 * @param[in,out] max_blocking_time - Pointer that will store the max blocking time for reliable reliability (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_reliability (
  const dds_qos_t * __restrict qos,
  dds_reliability_kind_t *kind,
  dds_duration_t *max_blocking_time);

/**
 * @ingroup qos_getters
 * @brief Get the transport priority qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] value - Pointer that will store the transport priority value
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_transport_priority (
  const dds_qos_t * __restrict qos,
  int32_t *value);

/**
 * @ingroup qos_getters
 * @brief Get the destination-order qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] kind - Pointer that will store the destination-order kind
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_destination_order (
  const dds_qos_t * __restrict qos,
  dds_destination_order_kind_t *kind);

/**
 * @ingroup qos_getters
 * @brief Get the writer data-lifecycle qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] autodispose - Pointer that will store the autodispose unregistered instances enable value
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_writer_data_lifecycle (
  const dds_qos_t * __restrict qos,
  bool *autodispose);

/**
 * @ingroup qos_getters
 * @brief Get the reader data-lifecycle qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] autopurge_nowriter_samples_delay - Pointer that will store the delay for auto-purging samples from instances in a no-writer state (optional)
 * @param[in,out] autopurge_disposed_samples_delay - Pointer that will store the delay for auto-purging of disposed instances (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_reader_data_lifecycle (
  const dds_qos_t * __restrict qos,
  dds_duration_t *autopurge_nowriter_samples_delay,
  dds_duration_t *autopurge_disposed_samples_delay);

/**
 * @ingroup qos_getters
 * @brief Get the durability-service qos policy values.
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out]  service_cleanup_delay - Pointer that will store the delay for purging of abandoned instances from the durability service (optional)
 * @param[in,out] history_kind - Pointer that will store history policy kind applied by the durability service (optional)
 * @param[in,out] history_depth - Pointer that will store history policy depth applied by the durability service (optional)
 * @param[in,out] max_samples - Pointer that will store number of samples resource-limit policy applied by the durability service (optional)
 * @param[in,out] max_instances - Pointer that will store number of instances resource-limit policy applied by the durability service (optional)
 * @param[in,out] max_samples_per_instance - Pointer that will store number of samples per instance resource-limit policy applied by the durability service (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_durability_service (
  const dds_qos_t * __restrict qos,
  dds_duration_t *service_cleanup_delay,
  dds_history_kind_t *history_kind,
  int32_t *history_depth,
  int32_t *max_samples,
  int32_t *max_instances,
  int32_t *max_samples_per_instance);

/**
 * @ingroup qos_getters
 * @brief Get the ignore-local qos policy
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] ignore - Pointer that will store whether to ignore readers/writers owned by the same participant (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_ignorelocal (
  const dds_qos_t * __restrict qos,
  dds_ignorelocal_kind_t *ignore);

/**
 * @ingroup qos_getters
 * @brief Gets the names of the properties from a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that contains properties
 * @param[in,out] n - Pointer to number of property names that are returned (optional)
 * @param[in,out] names - Pointer that will store the string(s) containing property name(s) (optional). This function will allocate the memory for the list of names and for the strings containing the names; the caller gets ownership of the allocated memory
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_propnames (
  const dds_qos_t * __restrict qos,
  uint32_t * n,
  char *** names);

/**
 * @ingroup qos_getters
 * @brief Get the value of the property with the provided name from a qos structure.
 *
 * In case more than one property exists with this name, the value for the first
 * property with this name will be returned.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that contains the property
 * @param[in] name - Pointer to name of the property
 * @param[in,out] value - Pointer to a string that will store the value of the property. The memory for storing the string value will be allocated by this function and the caller gets ownership of the allocated memory
 *
 * @returns - false iff any of the arguments is invalid, the qos is not present in the qos object or there was no property found with the provided name
 */
DDS_EXPORT bool
dds_qget_prop (
  const dds_qos_t * __restrict qos,
  const char * name,
  char ** value);

/**
 * @ingroup qos_getters
 * @brief Gets the names of the binary properties from a qos structure.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that contains binary properties
 * @param[in,out] n - Pointer to number of binary property names that are returned (optional)
 * @param[in,out] names - Pointer that will store the string(s) containing binary property name(s) (optional). This function will allocate the memory for the list of names and for the strings containing the names; the caller gets ownership of the allocated memory
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_bpropnames (
  const dds_qos_t * __restrict qos,
  uint32_t * n,
  char *** names);

/**
 * @ingroup qos_getters
 * @brief Get the value of the binary property with the provided name from a qos structure.
 *
 * In case more than one binary property exists with this name, the value for the first
 * binary property with this name will be returned.
 *
 * @param[in,out] qos - Pointer to a dds_qos_t structure that contains the property
 * @param[in] name - Pointer to name of the binary property
 * @param[in,out] value - Pointer to a buffer that will store the value of the property. If sz = 0 then a NULL pointer. The memory for storing the value will be allocated by this function and the caller gets ownership of the allocated memory
 * @param[in,out] sz - Pointer that will store the size of the returned buffer.
 *
 * @returns - false iff any of the arguments is invalid, the qos is not present in the qos object or there was no binary property found with the provided name
 */
DDS_EXPORT bool
dds_qget_bprop (
  const dds_qos_t * __restrict qos,
  const char * name,
  void ** value,
  size_t * sz);

/**
 * @ingroup qos_getters
 * @brief Get the type consistency enforcement qos policy values.
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] kind - Pointer that will store the type consistency enforcement kind (optional)
 * @param[in,out] ignore_sequence_bounds - Pointer that will store the boolean value for ignoring sequence bounds in type assignability checking (optional)
 * @param[in,out] ignore_string_bounds - Pointer that will store the boolean value for ignoring string bounds in type assignability checking (optional)
 * @param[in,out] ignore_member_names - Pointer that will store the boolean value for ignoring member names in type assignability checking (optional)
 * @param[in,out] prevent_type_widening - Pointer that will store the boolean value to prevent type widening in type assignability checking (optional)
 * @param[in,out] force_type_validation - Pointer that will store the boolean value to force type validation in assignability checking (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_type_consistency (
  const dds_qos_t * __restrict qos,
  dds_type_consistency_kind_t *kind,
  bool *ignore_sequence_bounds,
  bool *ignore_string_bounds,
  bool *ignore_member_names,
  bool *prevent_type_widening,
  bool *force_type_validation);

/**
 * @ingroup qos_getters
 * @brief Get the data representation qos policy value.
 *
 * Returns the data representation values that are set in the provided QoS object
 * and stores the number of values in out parameter 'n'. In case the 'values' parameter
 * is provided, this function will allocate a buffer that contains the data representation
 * values, and set 'values' to point to this buffer. It is the responsibility of the caller
 * to free the memory of this buffer.
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the policy
 * @param[in,out] n - Pointer that will store the number of data representation values
 * @param[in,out] values - Pointer that will store the data representation values (optional)
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 */
DDS_EXPORT bool
dds_qget_data_representation (
  const dds_qos_t * __restrict qos,
  uint32_t *n,
  dds_data_representation_id_t **values);

/**
 * @ingroup qos_getters
 * @brief Get the entity name from a qos structure
 *
 * @param[in] qos - Pointer to a dds_qos_t structure storing the entity name
 * @param[in,out] name - Pointer to a string that will store the returned entity name
 *
 * @returns - false iff any of the arguments is invalid or the qos is not present in the qos object
 *            or if a buffer to store the name could not be allocated.
 */
DDS_EXPORT bool dds_qget_entity_name (const dds_qos_t * __restrict qos, char **name);

#if defined (__cplusplus)
}
#endif
#endif
