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
#ifndef DDSI_KEYHASH_H
#define DDSI_KEYHASH_H

#if defined (__cplusplus)
extern "C" {
#endif

typedef struct ddsi_keyhash {
  unsigned char value[16];
} ddsi_keyhash_t;

#if defined (__cplusplus)
}
#endif

#endif
