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
#ifndef DDSRT_THREADS_FREERTOS_H
#define DDSRT_THREADS_FREERTOS_H

#include <FreeRTOS.h>
#include <task.h>

#define DDSRT_HAVE_THREAD_SETNAME (0)
#define DDSRT_HAVE_THREAD_LIST (0)

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct {
  TaskHandle_t task;
} ddsrt_thread_t;

typedef UBaseType_t ddsrt_tid_t;
typedef TaskHandle_t ddsrt_thread_list_id_t;
#define PRIdTID "lu"

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_THREADS_FREERTOS_H */
