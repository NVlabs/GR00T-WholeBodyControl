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
#ifndef DDSI_TIME_H
#define DDSI_TIME_H

#include <stdint.h>

#include "dds/export.h"
#include "dds/ddsrt/time.h"

#if defined (__cplusplus)
extern "C" {
#endif

typedef struct {
  int32_t seconds;
  uint32_t fraction;
} ddsi_time_t;
#define DDSI_TIME_INFINITE ((ddsi_time_t) { INT32_MAX, UINT32_MAX })
#define DDSI_TIME_INVALID ((ddsi_time_t) { -1, UINT32_MAX })

typedef ddsi_time_t ddsi_duration_t;

bool ddsi_is_valid_timestamp (ddsi_time_t t);

DDS_EXPORT ddsi_time_t ddsi_wctime_to_ddsi_time (ddsrt_wctime_t t);
DDS_EXPORT ddsrt_wctime_t ddsi_wctime_from_ddsi_time (ddsi_time_t x);
DDS_EXPORT ddsi_duration_t ddsi_to_ddsi_duration (dds_duration_t t);
DDS_EXPORT dds_duration_t ddsi_from_ddsi_duration (ddsi_duration_t x);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_TIME_H */
