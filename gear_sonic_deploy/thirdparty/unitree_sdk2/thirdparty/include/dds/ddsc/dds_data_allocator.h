/*
 * Copyright(c) 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

#ifndef DDS_DATA_ALLOCATOR_H
#define DDS_DATA_ALLOCATOR_H

/**
 * @defgroup data_allocator (Data Allocator)
 * @ingroup dds
 * A quick-and-dirty provisional interface
 */

#include "dds/dds.h"
#include "dds/ddsrt/attributes.h"
#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

// macOS' mutexes require quite a lot of space, but it is not quite enough
// to make this system-dependent
#define DDS_DATA_ALLOCATOR_MAX_SIZE (12 * sizeof (void *))

/**
 * @ingroup data_allocator
 * @brief Data Allocator structure
 * Contains internal details about the data allocator for a given entity
 */
typedef struct dds_data_allocator {
  dds_entity_t entity;  /**< to which entity this allocator is attached */
  union {
    unsigned char bytes[DDS_DATA_ALLOCATOR_MAX_SIZE]; /**< internal details */
    void *align_ptr; /**< internal details */
    uint64_t align_u64; /**< internal details */
  } opaque; /**< internal details */
} dds_data_allocator_t;

/**
 * @ingroup data_allocator
 * @brief Initialize an object for performing allocations/frees in the context of a reader/writer
 *
 * The operation will fall back to standard heap allocation if nothing better is available.
 *
 * @param[in] entity the handle of the entity
 * @param[out] data_allocator opaque allocator object to initialize
 *
 * @returns success or a generic error indication
 *
 * @retval DDS_RETCODE_OK
 *    the allocator object was successfully initialized
 * @retval DDS_RETCODE_BAD_PARAMETER
 *    entity is invalid, data_allocator is a null pointer
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *    Cyclone DDS is not initialized
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *    operation not supported on this entity
 */
DDS_EXPORT dds_return_t dds_data_allocator_init (dds_entity_t entity, dds_data_allocator_t *data_allocator);

/**
 * @ingroup data_allocator
 * @brief Initialize an object for performing standard allocations/frees on the heap
 *
 * @param[out] data_allocator opaque allocator object to initialize
 *
 * @returns success or a generic error indication
 *
 * @retval DDS_RETCODE_OK
 *    the allocator object was successfully initialized
 * @retval DDS_RETCODE_BAD_PARAMETER
 *    entity is invalid, data_allocator is a null pointer
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *    Cyclone DDS is not initialized
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *    operation not supported on this entity
 */
DDS_EXPORT dds_return_t dds_data_allocator_init_heap (dds_data_allocator_t *data_allocator);

/**
 * @ingroup data_allocator
 * @brief Finalize a previously initialized allocator object
 *
 * @param[in,out] data_allocator object to finalize
 *
 * @returns success or an error indication
 *
 * @retval DDS_RETCODE_OK
 *    the data was successfully finalized
 * @retval DDS_RETCODE_BAD_PARAMETER
 *    data_allocator does not reference a valid entity
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *    Cyclone DDS is not initialized
 */
DDS_EXPORT dds_return_t dds_data_allocator_fini (dds_data_allocator_t *data_allocator);

/**
 * @ingroup data_allocator
 * @brief Allocate memory using the given allocator
 *
 * @param[in,out] data_allocator  initialized allocator object
 * @param[in] size minimum number of bytes to allocate with suitable alignment
 *
 * @returns a pointer to unaliased, uninitialized memory of at least the requested size, or NULL
 */
DDS_EXPORT void *dds_data_allocator_alloc (dds_data_allocator_t *data_allocator, size_t size)
  ddsrt_attribute_warn_unused_result ddsrt_attribute_malloc;

/**
 * @ingroup data_allocator
 * @brief Release memory using the given allocator
 *
 * @param[in,out] data_allocator  initialized allocator object
 * @param[in] ptr memory to free
 *
 * @returns success or an error indication
 *
 * @retval DDS_RETCODE_OK
 *    the memory was successfully released
 * @retval DDS_RETCODE_BAD_PARAMETER
 *    data_allocator does not reference a valid entity
 *  @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *    dds_data_allocator already finalized
 */
DDS_EXPORT dds_return_t dds_data_allocator_free (dds_data_allocator_t *data_allocator, void *ptr);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// @note This function declaration is deprecated here and has been moved to
/// dds_loan_api.h.
DDS_EXPORT bool dds_is_loan_available(const dds_entity_t entity);

/// @note This function declaration is deprecated here and has been moved to
/// dds_loan_api.h.
DDS_EXPORT bool is_loan_available(const dds_entity_t entity);
#endif

#if defined (__cplusplus)
}
#endif
#endif
