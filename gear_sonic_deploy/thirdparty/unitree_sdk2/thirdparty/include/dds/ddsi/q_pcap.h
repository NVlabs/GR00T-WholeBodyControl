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
#ifndef Q_PCAP_H
#define Q_PCAP_H

#include <stdio.h>
#include "dds/ddsrt/time.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct msghdr;

FILE * new_pcap_file (struct ddsi_domaingv *gv, const char *name);

void write_pcap_received (struct ddsi_domaingv *gv, ddsrt_wctime_t tstamp, const struct sockaddr_storage *src, const struct sockaddr_storage *dst, unsigned char *buf, size_t sz);
void write_pcap_sent (struct ddsi_domaingv *gv, ddsrt_wctime_t tstamp, const struct sockaddr_storage *src,
  const ddsrt_msghdr_t *hdr, size_t sz);

#if defined (__cplusplus)
}
#endif

#endif /* Q_PCAP_H */
