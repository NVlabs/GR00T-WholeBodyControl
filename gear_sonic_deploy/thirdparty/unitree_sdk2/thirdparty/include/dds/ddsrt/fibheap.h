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
#ifndef DDSRT_FIBHEAP_H
#define DDSRT_FIBHEAP_H

#include <stdint.h>

#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

typedef struct ddsrt_fibheap_node {
  struct ddsrt_fibheap_node *parent, *children;
  struct ddsrt_fibheap_node *prev, *next;
  unsigned mark: 1;
  unsigned degree: 31;
} ddsrt_fibheap_node_t;

typedef struct ddsrt_fibheap_def {
    uintptr_t offset;
    int (*cmp) (const void *va, const void *vb);
} ddsrt_fibheap_def_t;

typedef struct ddsrt_fibheap {
  ddsrt_fibheap_node_t *roots; /* points to root with min key value */
} ddsrt_fibheap_t;

#define DDSRT_FIBHEAPDEF_INITIALIZER(offset, cmp) { (offset), (cmp) }

DDS_EXPORT void ddsrt_fibheap_def_init (ddsrt_fibheap_def_t *fhdef, uintptr_t offset, int (*cmp) (const void *va, const void *vb));
DDS_EXPORT void ddsrt_fibheap_init (const ddsrt_fibheap_def_t *fhdef, ddsrt_fibheap_t *fh);
DDS_EXPORT void *ddsrt_fibheap_min (const ddsrt_fibheap_def_t *fhdef, const ddsrt_fibheap_t *fh);
DDS_EXPORT void ddsrt_fibheap_merge (const ddsrt_fibheap_def_t *fhdef, ddsrt_fibheap_t *a, ddsrt_fibheap_t *b);
DDS_EXPORT void ddsrt_fibheap_insert (const ddsrt_fibheap_def_t *fhdef, ddsrt_fibheap_t *fh, const void *vnode);
DDS_EXPORT void ddsrt_fibheap_delete (const ddsrt_fibheap_def_t *fhdef, ddsrt_fibheap_t *fh, const void *vnode);
DDS_EXPORT void *ddsrt_fibheap_extract_min (const ddsrt_fibheap_def_t *fhdef, ddsrt_fibheap_t *fh);
DDS_EXPORT void ddsrt_fibheap_decrease_key (const ddsrt_fibheap_def_t *fhdef, ddsrt_fibheap_t *fh, const void *vnode); /* to be called AFTER decreasing the key */

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_FIBHEAP_H */
