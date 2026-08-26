/*
 * Copyright(c) 2021 ZettaScale Technology
 * Copyright(c) 2021 Apex.AI, Inc
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

// API extension
// defines functions needed for loaning and shared memory usage

#ifndef _DDS_LOAN_API_H_
#define _DDS_LOAN_API_H_

#include "dds/ddsc/dds_basic_types.h"
#include "dds/ddsrt/retcode.h"
#include "dds/export.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @ingroup loan
 * @brief Check if a Loan is available to reader/writer
 * The loan is available if the shared memory is enabled and all the constraints
 * to enable shared memory are met and the type is fixed
 * @note dds_loan_sample can be used if and only if
 * dds_is_loan_available returns true.
 *
 * @param[in] entity the handle of the entity
 *
 * @returns loan available or not
 */
DDS_EXPORT bool dds_is_loan_available(const dds_entity_t entity);

/**
 * @ingroup loan
 * @brief Check if a shared memory is available to reader/writer.
 *
 * @note dds_loan_shared_memory_buffer can be used if and only if
 * dds_is_shared_memory_available returns true.
 *
 * @param[in] entity the handle of the entity
 *
 * @returns true if shared memory is available, false otherwise
 */
DDS_EXPORT bool dds_is_shared_memory_available(const dds_entity_t entity);

DDS_DEPRECATED_EXPORT bool is_loan_available(const dds_entity_t entity);

/**
 * @ingroup loan
 * @brief Loan a shared memory buffer of a specific size from the writer.
 *
 * @note Currently this function is to be used with dds_writecdr by adding the
 * loaned buffer to serdata as iox_chunk.
 * @note The function can only be used if dds_is_shared_memory_available is
 *       true for the writer.
 *
 * @param[in] writer the writer to loan the buffer from
 * @param[in] size the requested buffer size
 * @param[out] buffer the loaned buffer
 *
 * @returns DDS_RETCODE_OK if successful, DDS_RETCODE_ERROR otherwise
 */
DDS_EXPORT dds_return_t dds_loan_shared_memory_buffer(dds_entity_t writer,
                                                      size_t size,
                                                      void **buffer);

/**
 * @ingroup loan
 * @brief Loan a sample from the writer.
 *
 * @note This function is to be used with dds_write to publish the loaned
 * sample.
 * @note The function can only be used if dds_is_loan_available is
 *       true for the writer.
 *
 * @param[in] writer the writer to loan the buffer from
 * @param[out] sample the loaned sample
 *
 * @returns DDS_RETCODE_OK if successful, DDS_RETCODE_ERROR otherwise
 */
DDS_EXPORT dds_return_t dds_loan_sample(dds_entity_t writer, void **sample);

#if defined(__cplusplus)
}
#endif
#endif
