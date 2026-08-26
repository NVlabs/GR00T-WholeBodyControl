/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSRT_PROCESS_H
#define DDSRT_PROCESS_H

#include "dds/export.h"
#include "dds/ddsrt/time.h"
#include "dds/ddsrt/types.h"
#include "dds/ddsrt/retcode.h"

#if DDSRT_WITH_FREERTOS
#include <FreeRTOS.h>
#include <task.h>
typedef TaskHandle_t ddsrt_pid_t; /* typedef void *TaskHandle_t */
#define PRIdPID "p"
#define DDSRT_HAVE_MULTI_PROCESS 0
/* DDSRT_WITH_FREERTOS */
#elif defined(_WIN32)
typedef uint32_t ddsrt_pid_t;
#define PRIdPID PRIu32
#define DDSRT_HAVE_MULTI_PROCESS 1
/* _WIN32 */
#else
#include <unistd.h>
#if defined(_WRS_KERNEL)
typedef RTP_ID ddsrt_pid_t; /* typedef struct wind_rtp *RTP_ID */
#define PRIdPID PRIuPTR
#define DDSRT_HAVE_MULTI_PROCESS 0
#else
typedef pid_t ddsrt_pid_t;
#define PRIdPID "d"
#define DDSRT_HAVE_MULTI_PROCESS 1
#endif
#endif


#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @brief Return process ID (PID) of the calling process.
 *
 * @returns The process ID of the calling process.
 */
DDS_EXPORT ddsrt_pid_t
ddsrt_getpid(void);

/**
 * @brief Return process name of the calling process.
 *
 * On linux maps to /proc/self/cmdline's first entry (argv[0]),
 * on mac/windows maps to relevant API calls. Falls back to process-{pid}
 * on failure.
 *
 * @returns The process name of the calling process.
 */
DDS_EXPORT char *
ddsrt_getprocessname(void);

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_PROCESS_H */
