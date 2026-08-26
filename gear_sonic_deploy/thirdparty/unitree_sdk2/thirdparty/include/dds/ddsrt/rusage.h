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
#ifndef DDSRT_RUSAGE_H
#define DDSRT_RUSAGE_H


#include <stddef.h>

#include "dds/config.h"
#include "dds/ddsrt/time.h"
#include "dds/ddsrt/retcode.h"
#include "dds/ddsrt/threads.h"

#if DDSRT_HAVE_RUSAGE

#if defined (__cplusplus)
extern "C" {
#endif

typedef struct {
  dds_time_t utime; /* User CPU time used. */
  dds_time_t stime; /* System CPU time used. */
  size_t maxrss; /* Maximum resident set size in bytes. */
  size_t idrss; /* Integral unshared data size. Not maintained on (at least)
                   Linux and Windows. */
  size_t nvcsw; /* Voluntary context switches. Not maintained on Windows. */
  size_t nivcsw; /* Involuntary context switches. Not maintained on Windows. */
} ddsrt_rusage_t;

enum ddsrt_getrusage_who {
  DDSRT_RUSAGE_SELF,
  DDSRT_RUSAGE_THREAD
};

/**
 * @brief Get resource usage for the current thread or process.
 *
 * @param[in]  who    DDSRT_RUSAGE_SELF or DDSRT_RUSAGE_THREAD.
 * @param[in]  usage  Structure where resource usage is returned.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Resource usage successfully returned in @usage.
 * @retval DDS_RETCODE_OUT_OF_RESOURCES
 *             There were not enough resources to get resource usage.
 * @retval DDS_RETCODE_ERROR
 *             An unidentified error occurred.
 */
DDS_EXPORT dds_return_t ddsrt_getrusage(enum ddsrt_getrusage_who who, ddsrt_rusage_t *usage);

#if DDSRT_HAVE_THREAD_LIST
/**
 * @brief Get resource usage for some thread.
 *
 * @param[in]  tid    id of the thread of to get the resource usage for
 * @param[in]  usage  Structure where resource usage is returned.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Resource usage successfully returned in @usage.
 * @retval DDS_RETCODE_OUT_OF_RESOURCES
 *             There were not enough resources to get resource usage.
 * @retval DDS_RETCODE_ERROR
 *             An unidentified error occurred.
 */
DDS_EXPORT dds_return_t ddsrt_getrusage_anythread (ddsrt_thread_list_id_t tid, ddsrt_rusage_t * __restrict usage);
#endif

#if defined (__cplusplus)
}
#endif

#endif // DDSRT_HAVE_RUSAGE

#endif // DDSRT_RUSAGE_H
