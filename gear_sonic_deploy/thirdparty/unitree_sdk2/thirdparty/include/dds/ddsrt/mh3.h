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
#ifndef DDSRT_MH3_H
#define DDSRT_MH3_H

#include <stdint.h>
#include <stddef.h>

#include "dds/export.h"

#if defined(__cplusplus)
extern "C" {
#endif

DDS_EXPORT uint32_t
ddsrt_mh3(
  const void *key,
  size_t len,
  uint32_t seed);

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_MH3_H */
