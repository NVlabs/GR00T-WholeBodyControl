/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */


/*
 * WARNING This file is only needed for the work around for https://github.com/eclipse-cyclonedds/cyclonedds/issues/74
 * Do not include this file in an application! Once issue #74 is solved this header file should be removed.
 */

#ifndef DDS_INTERNAL_API_H
#define DDS_INTERNAL_API_H

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @anchor DDS_READ_WITHOUT_LOCK
 * @ingroup internal
 * @brief Constant to use with dds_read() or dds_take() when using dds_reader_lock_samples()
 */
#define DDS_READ_WITHOUT_LOCK (0xFFFFFFED)

/**
 * @ingroup internal
 * @brief Returns number of samples in read cache and locks the reader cache,
 * to make sure that the samples content doesn't change.
 *
 * Because the cache is locked, you should be able to read/take without having to
 * lock first. This is done by passing the @ref DDS_READ_WITHOUT_LOCK value to the
 * read/take call as maxs. Then the read/take will not lock but still unlock.
 *
 * CycloneDDS doesn't support a read/take that just returns all
 * available samples issue #74. As a work around to support LENGTH_UNLIMITED in C++.
 * dds_reader_lock_samples() and @ref DDS_READ_WITHOUT_LOCK are needed.
 *
 * @param[in]   reader  Reader to lock the cache of.
 *
 * @returns the number of samples in the reader cache.
 */
DDS_EXPORT uint32_t
dds_reader_lock_samples (dds_entity_t reader);

#if defined (__cplusplus)
}
#endif
#endif
