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
#ifndef DDSRT_CDTORS_H
#define DDSRT_CDTORS_H

#include "dds/export.h"
#include "dds/ddsrt/sync.h"

#if defined (__cplusplus)
extern "C" {
#endif

DDS_EXPORT void ddsrt_init(void);

DDS_EXPORT void ddsrt_fini(void);

DDS_EXPORT ddsrt_mutex_t *ddsrt_get_singleton_mutex(void);

DDS_EXPORT ddsrt_cond_t *ddsrt_get_singleton_cond(void);

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_CDTORS_H */
