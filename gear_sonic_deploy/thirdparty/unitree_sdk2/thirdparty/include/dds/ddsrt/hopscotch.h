/*
 * Copyright(c) 2006 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSRT_HOPSCOTCH_H
#define DDSRT_HOPSCOTCH_H

#include <stdint.h>

#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

/*
 * The hopscotch hash table is dependent on a proper functioning hash.
 * If the hash function generates a lot of hash collisions, then it will
 * not be able to handle that by design.
 * It is capable of handling some collisions, but not more than 32 per
 * bucket (less, when other hash values are clustered around the
 * collision value).
 * When proper distributed hash values are generated, then hopscotch
 * works nice and quickly.
 */
typedef uint32_t (*ddsrt_hh_hash_fn) (const void *a);

/*
 * Hopscotch needs to be able to compare two elements.
 * Returns 0 when not equal.
 */
typedef int (*ddsrt_hh_equals_fn) (const void *a, const void *b);

/*
 * Hopscotch is will resize its internal buckets list when needed. It will
 * call this garbage collection function with the old buckets list. The
 * caller has to delete the list when it deems it safe to do so.
 */
typedef void (*ddsrt_hh_buckets_gc_fn) (void *bs, void *arg);

/* Sequential version */
struct ddsrt_hh;

struct ddsrt_hh_iter {
    struct ddsrt_hh *hh;
    uint32_t cursor;
};

DDS_EXPORT struct ddsrt_hh *ddsrt_hh_new (uint32_t init_size, ddsrt_hh_hash_fn hash, ddsrt_hh_equals_fn equals) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_hh_free (struct ddsrt_hh * __restrict hh) ddsrt_nonnull_all;
DDS_EXPORT void *ddsrt_hh_lookup (const struct ddsrt_hh * __restrict rt, const void * __restrict keyobject) ddsrt_nonnull_all;
DDS_EXPORT int ddsrt_hh_add (struct ddsrt_hh * __restrict rt, void * __restrict data) ddsrt_nonnull_all;
DDS_EXPORT int ddsrt_hh_remove (struct ddsrt_hh * __restrict rt, const void * __restrict keyobject) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_hh_add_absent (struct ddsrt_hh * __restrict rt, void * __restrict data) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_hh_remove_present (struct ddsrt_hh * __restrict rt, void * __restrict keyobject) ddsrt_nonnull_all;
DDS_EXPORT void ddsrt_hh_enum (struct ddsrt_hh * __restrict rt, void (*f) (void *a, void *f_arg), void *f_arg) ddsrt_nonnull ((1, 2)); /* may delete a */
DDS_EXPORT void *ddsrt_hh_iter_first (struct ddsrt_hh * __restrict rt, struct ddsrt_hh_iter * __restrict iter) ddsrt_nonnull_all; /* may delete nodes */
DDS_EXPORT void *ddsrt_hh_iter_next (struct ddsrt_hh_iter * __restrict iter) ddsrt_nonnull_all;

/* Concurrent version */
struct ddsrt_chh;
struct ddsrt_chh_bucket;

#if ! ddsrt_has_feature_thread_sanitizer
struct ddsrt_chh_iter {
  struct ddsrt_chh_bucket *bs;
  uint32_t size;
  uint32_t cursor;
};
#else
struct ddsrt_chh_iter {
  struct ddsrt_chh *chh;
  struct ddsrt_hh_iter it;
};
#endif

DDS_EXPORT struct ddsrt_chh *ddsrt_chh_new (uint32_t init_size, ddsrt_hh_hash_fn hash, ddsrt_hh_equals_fn equals, ddsrt_hh_buckets_gc_fn gc_buckets, void *gc_buckets_arg);
DDS_EXPORT void ddsrt_chh_free (struct ddsrt_chh * __restrict hh);
DDS_EXPORT void *ddsrt_chh_lookup (struct ddsrt_chh * __restrict rt, const void * __restrict keyobject);
DDS_EXPORT int ddsrt_chh_add (struct ddsrt_chh * __restrict rt, void * __restrict data);
DDS_EXPORT int ddsrt_chh_remove (struct ddsrt_chh * __restrict rt, const void * __restrict keyobject);
DDS_EXPORT void ddsrt_chh_enum_unsafe (struct ddsrt_chh * __restrict rt, void (*f) (void *a, void *f_arg), void *f_arg); /* may delete a */
DDS_EXPORT void *ddsrt_chh_iter_first (struct ddsrt_chh * __restrict rt, struct ddsrt_chh_iter *it);
DDS_EXPORT void *ddsrt_chh_iter_next (struct ddsrt_chh_iter *it);
/* Sequential version, embedded data */
struct ddsrt_ehh;

struct ddsrt_ehh_iter {
    struct ddsrt_ehh *hh;
    uint32_t cursor;
};

DDS_EXPORT struct ddsrt_ehh *ddsrt_ehh_new (size_t elemsz, uint32_t init_size, ddsrt_hh_hash_fn hash, ddsrt_hh_equals_fn equals);
DDS_EXPORT void ddsrt_ehh_free (struct ddsrt_ehh * __restrict hh);
DDS_EXPORT void *ddsrt_ehh_lookup (const struct ddsrt_ehh * __restrict rt, const void * __restrict keyobject);
DDS_EXPORT int ddsrt_ehh_add (struct ddsrt_ehh * __restrict rt, const void * __restrict data);
DDS_EXPORT int ddsrt_ehh_remove (struct ddsrt_ehh * __restrict rt, const void * __restrict keyobject);
DDS_EXPORT void ddsrt_ehh_enum (struct ddsrt_ehh * __restrict rt, void (*f) (void *a, void *f_arg), void *f_arg); /* may delete a */
DDS_EXPORT void *ddsrt_ehh_iter_first (struct ddsrt_ehh * __restrict rt, struct ddsrt_ehh_iter * __restrict iter); /* may delete nodes */
DDS_EXPORT void *ddsrt_ehh_iter_next (struct ddsrt_ehh_iter * __restrict iter);

#if defined (__cplusplus)
}
#endif

#endif
