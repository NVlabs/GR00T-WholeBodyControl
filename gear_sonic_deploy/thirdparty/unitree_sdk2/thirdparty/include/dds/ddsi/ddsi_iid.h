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
#ifndef _DDS_IID_H_
#define _DDS_IID_H_

#include "dds/export.h"
#include "dds/ddsrt/atomics.h"
#include "dds/ddsrt/sync.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_iid {
  ddsrt_atomic_uint64_t counter;
  uint32_t key[4];
};

DDS_EXPORT void ddsi_iid_init (void);
DDS_EXPORT void ddsi_iid_fini (void);
DDS_EXPORT uint64_t ddsi_iid_gen (void);

#if defined (__cplusplus)
}
#endif
#endif
