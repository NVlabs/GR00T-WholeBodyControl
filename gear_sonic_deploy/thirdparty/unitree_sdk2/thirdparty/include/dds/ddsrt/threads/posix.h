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
#ifndef DDSRT_POSIX_THREAD_H
#define DDSRT_POSIX_THREAD_H

#include <pthread.h>

#if defined(__VXWORKS__)
#define DDSRT_HAVE_THREAD_SETNAME (0)
#else
#define DDSRT_HAVE_THREAD_SETNAME (1)
#endif
#if defined (__linux) || defined (__APPLE__)
#define DDSRT_HAVE_THREAD_LIST (1)
#else
#define DDSRT_HAVE_THREAD_LIST (0)
#endif

#if defined (__cplusplus)
extern "C" {
#endif

#if defined(__linux)
typedef long int ddsrt_tid_t;
#define PRIdTID "ld"
typedef long int ddsrt_thread_list_id_t;
/* __linux */
#elif defined(__FreeBSD__) && (__FreeBSD__ >= 9)
/* FreeBSD >= 9.0 */
typedef int ddsrt_tid_t;
#define PRIdTID "d"
/* __FreeBSD__ */
#elif defined(__APPLE__) && !(defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && \
                                      __MAC_OS_X_VERSION_MIN_REQUIRED < 1060)
/* macOS X >= 10.6 */
typedef uint64_t ddsrt_tid_t;
#define PRIdTID PRIu64
/* ddsrt_thread_list_id_t is actually a mach_port_t */
typedef uint32_t ddsrt_thread_list_id_t;
/* __APPLE__ */
#elif defined(__VXWORKS__)
/* TODO: Verify taskIdSelf is the right function to use on VxWorks */
typedef TASK_ID ddsrt_tid_t;
# if defined(_WRS_CONFIG_LP64)
#   define PRIdPID PRIuPTR /* typedef struct windTcb *TASK_ID */
# else
#   define PRIdPID "d" /* typedef int TASK_ID */
# endif
/* __VXWORKS__ */
#else
typedef uintptr_t ddsrt_tid_t;
#define PRIdTID PRIuPTR
#endif

/* Wrapped in a struct to force conformation to abstraction. */
typedef struct {
    pthread_t v;
} ddsrt_thread_t;

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_POSIX_THREAD_H */
