/*
 * Copyright(c) 2019 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSI_GUID_H
#define DDSI_GUID_H

#include <stdint.h>

#if defined (__cplusplus)
extern "C" {
#endif

typedef union ddsi_guid_prefix {
  unsigned char s[12];
  uint32_t u[3];
} ddsi_guid_prefix_t;
typedef union ddsi_entityid {
  uint32_t u;
} ddsi_entityid_t;
typedef struct ddsi_guid {
  ddsi_guid_prefix_t prefix;
  ddsi_entityid_t entityid;
} ddsi_guid_t;

#if defined (__cplusplus)
}
#endif

#endif
