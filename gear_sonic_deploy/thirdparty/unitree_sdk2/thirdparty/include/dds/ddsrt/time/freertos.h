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
#ifndef DDSRT_TIME_FREERTOS_H
#define DDSRT_TIME_FREERTOS_H

#include <assert.h>
#include <FreeRTOS.h>

#if defined (__cplusplus)
extern "C" {
#endif

#define DDSRT_NSECS_PER_TICK (DDS_NSECS_IN_SEC / configTICK_RATE_HZ)

inline TickType_t
ddsrt_duration_to_ticks_ceil(
  dds_duration_t reltime)
{
  TickType_t ticks = 0;

  assert(portMAX_DELAY > configTICK_RATE_HZ);

  if (reltime == DDS_INFINITY) {
    ticks = portMAX_DELAY;
  } else if (reltime > 0) {
    dds_duration_t max_nsecs =
      (DDS_INFINITY / DDSRT_NSECS_PER_TICK < portMAX_DELAY
        ? DDS_INFINITY - 1 : portMAX_DELAY * DDSRT_NSECS_PER_TICK);

    if (reltime > max_nsecs - (DDSRT_NSECS_PER_TICK - 1)) {
      ticks = portMAX_DELAY;
    } else {
      ticks = (TickType_t)((reltime + (DDSRT_NSECS_PER_TICK - 1)) / DDSRT_NSECS_PER_TICK);
    }
  }

  return ticks;
}

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_TIME_FREERTOS_H */
