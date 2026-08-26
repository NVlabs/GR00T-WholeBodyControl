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
#ifndef _DDSI_STATISTICS_H_
#define _DDSI_STATISTICS_H_

#include <stdint.h>

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_reader;
struct ddsi_writer;

void ddsi_get_writer_stats (struct ddsi_writer *wr, uint64_t * __restrict rexmit_bytes, uint32_t * __restrict throttle_count, uint64_t * __restrict time_throttled, uint64_t * __restrict time_retransmit);
void ddsi_get_reader_stats (struct ddsi_reader *rd, uint64_t * __restrict discarded_bytes);

#if defined (__cplusplus)
}
#endif
#endif
