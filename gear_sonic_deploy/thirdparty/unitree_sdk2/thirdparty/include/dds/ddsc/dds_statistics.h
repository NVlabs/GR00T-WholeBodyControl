/*
 * Copyright(c) 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

#ifndef DDS_STATISTICS_H
#define DDS_STATISTICS_H

/**
 * @defgroup statistics (DDS Statistics)
 * @ingroup dds
 * @warning Unstable API
 *
 * A quick-and-dirty provisional interface
 */

#include "dds/dds.h"
#include "dds/ddsrt/attributes.h"
#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @brief Kind of statistical value
 * @ingroup statistics
 */
enum dds_stat_kind {
  DDS_STAT_KIND_UINT32,          /**< value is a 32-bit unsigned integer */
  DDS_STAT_KIND_UINT64,          /**< value is a 64-bit unsigned integer */
  DDS_STAT_KIND_LENGTHTIME       /**< value is integral(length(t) dt) */
};

/**
 * @brief KeyValue statistics entry
 * @ingroup statistics
 */
struct dds_stat_keyvalue {
  const char *name;              /**< name, memory owned by library */
  enum dds_stat_kind kind;       /**< value type */
  union {
    uint32_t u32; /**< used if kind == DDS_STAT_KIND_UINT32 */
    uint64_t u64; /**< used if kind == DDS_STAT_KIND_UINT64 */
    uint64_t lengthtime; /**< used if kind == DDS_STAT_KIND_LENGTHTIME */
  } u; /**< value */
};

/**
 * @brief Statistics container
 * @ingroup statistics
 */
struct dds_statistics {
  dds_entity_t entity;           /**< handle of entity to which this set of values applies */
  uint64_t opaque;               /**< internal data */
  dds_time_t time;               /**< time stamp of latest call to dds_refresh_statistics() */
  size_t count;                  /**< number of key-value pairs */
  struct dds_stat_keyvalue kv[]; /**< data */
};

/**
 * @brief Allocate a new statistics object for entity
 * @ingroup statistics
 *
 * This allocates and populates a newly allocated `struct dds_statistics` for the
 * specified entity.
 *
 * @param[in] entity       the handle of the entity
 *
 * @returns a newly allocated and populated statistics structure or NULL if entity is
 * invalid or doesn't support any statistics.
 */
DDS_EXPORT struct dds_statistics *dds_create_statistics (dds_entity_t entity);

/**
 * @brief Update a previously created statistics structure with current values
 * @ingroup statistics
 *
 * Only the time stamp and the values (and "opaque") may change.  The set of keys and the
 * types of the values do not change.
 *
 * @param[in,out] stat     statistics structure to update the values of
 *
 * @returns success or an error indication
 *
 * @retval DDS_RETCODE_OK
 *    the data was successfully updated
 * @retval DDS_RETCODE_BAD_PARAMETER
 *    stats is a null pointer or the referenced entity no longer exists
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *    library was deinitialized
 */
DDS_EXPORT dds_return_t dds_refresh_statistics (struct dds_statistics *stat);

/**
 * @brief Free a previously created statistics object
 * @ingroup statistics
 *
 * This frees the statistics object.  Passing a null pointer is a no-op.  The operation
 * succeeds also if the referenced entity no longer exists.
 *
 * @param[in] stat         statistics object to free
 */
DDS_EXPORT void dds_delete_statistics (struct dds_statistics *stat);

/**
 * @brief Lookup a specific value by name
 * @ingroup statistics
 *
 * This looks up the specified name in the list of keys in `stat` and returns the address
 * of the key-value pair if present, a null pointer if not.  If `stat` is a null pointer,
 * it returns a null pointer.
 *
 * @param[in] stat         statistics object to lookup a name in (or NULL)
 * @param[in] name         name to look for
 *
 * @returns The address of the key-value pair inside `stat`, or NULL if `stat` is NULL or
 * `name` does not match a key in `stat.
 */
DDS_EXPORT const struct dds_stat_keyvalue *dds_lookup_statistic (const struct dds_statistics *stat, const char *name)
  ddsrt_nonnull ((2));

#if defined (__cplusplus)
}
#endif
#endif
