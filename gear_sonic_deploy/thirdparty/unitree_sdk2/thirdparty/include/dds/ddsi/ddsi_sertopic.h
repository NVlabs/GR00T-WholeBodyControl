/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSI_SERTOPIC_H
#define DDSI_SERTOPIC_H

#include "dds/ddsrt/atomics.h"
#include "dds/ddsrt/avl.h"
#include "dds/ddsc/dds_public_alloc.h"
#include "dds/ddsi/ddsi_sertype.h"
#include "dds/ddsi/ddsi_serdata.h" // for serdata_kind
#include "dds/ddsi/ddsi_keyhash.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct nn_rdata;
struct ddsi_serdata;
struct ddsi_sertopic_serdata_ops;
struct ddsi_sertopic_ops;
struct ddsi_domaingv;

struct ddsi_sertopic {
  const struct ddsi_sertopic_ops *ops;
  const struct ddsi_sertopic_serdata_ops *serdata_ops;
  uint32_t serdata_basehash;
  bool topickind_no_key;
  char *name;
  char *type_name;
  struct ddsi_domaingv *gv;
  ddsrt_atomic_uint32_t refc; /* counts refs from entities (topic, reader, writer), not from data */
};

/* Called to compare two sertopics for equality, if it is already known that name,
   type name, topickind_no_Key, and operations are all the same.  (serdata_basehash
   is computed from the set of operations.) */
typedef bool (*ddsi_sertopic_equal_t) (const struct ddsi_sertopic *a, const struct ddsi_sertopic *b);

/* Hash the custom components of a sertopic (this XOR'd with a hash computed from
   the fields that are defined in struct ddsi_sertopic) */
typedef uint32_t (*ddsi_sertopic_hash_t) (const struct ddsi_sertopic *tp);

/* Called when the refcount dropped to zero */
typedef void (*ddsi_sertopic_free_t) (struct ddsi_sertopic *tp);

/* Zero out a sample, used for generating samples from just a key value and in cleaning up
   after dds_return_loan */
typedef void (*ddsi_sertopic_zero_samples_t) (const struct ddsi_sertopic *d, void *samples, size_t count);

/* (Re)allocate an array of samples, used in growing loaned sample arrays in dds_read */
typedef void (*ddsi_sertopic_realloc_samples_t) (void **ptrs, const struct ddsi_sertopic *d, void *old, size_t oldcount, size_t count);

/* Release any memory allocated by ddsi_sertopic_to_sample (also undo sertopic_alloc_sample if "op" so requests) */
typedef void (*ddsi_sertopic_free_samples_t) (const struct ddsi_sertopic *d, void **ptrs, size_t count, dds_free_op_t op);

struct ddsi_sertopic_ops {
  ddsi_sertopic_free_t free;
  ddsi_sertopic_zero_samples_t zero_samples;
  ddsi_sertopic_realloc_samples_t realloc_samples;
  ddsi_sertopic_free_samples_t free_samples;
  ddsi_sertopic_equal_t equal;
  ddsi_sertopic_hash_t hash;
};

extern const struct ddsi_serdata_ops ddsi_sertopic_serdata_ops_wrap;

DDS_EXPORT void ddsi_sertopic_init (struct ddsi_sertopic *tp, const char *name, const char *type_name, const struct ddsi_sertopic_ops *sertopic_ops, const struct ddsi_sertopic_serdata_ops *serdata_ops, bool topickind_no_key);
DDS_EXPORT void ddsi_sertopic_fini (struct ddsi_sertopic *tp);
DDS_EXPORT struct ddsi_sertopic *ddsi_sertopic_ref (const struct ddsi_sertopic *tp);
DDS_EXPORT void ddsi_sertopic_unref (struct ddsi_sertopic *tp);
DDS_EXPORT uint32_t ddsi_sertopic_compute_serdata_basehash (const struct ddsi_sertopic_serdata_ops *ops);
DDS_EXPORT struct ddsi_sertype *ddsi_sertype_from_sertopic (struct ddsi_sertopic *tp);

DDS_EXPORT bool ddsi_sertopic_equal (const struct ddsi_sertopic *a, const struct ddsi_sertopic *b);
DDS_EXPORT uint32_t ddsi_sertopic_hash (const struct ddsi_sertopic *tp);

DDS_INLINE_EXPORT inline void ddsi_sertopic_free (struct ddsi_sertopic *tp) {
  tp->ops->free (tp);
}
DDS_INLINE_EXPORT inline void ddsi_sertopic_zero_samples (const struct ddsi_sertopic *tp, void *samples, size_t count) {
  tp->ops->zero_samples (tp, samples, count);
}
DDS_INLINE_EXPORT inline void ddsi_sertopic_realloc_samples (void **ptrs, const struct ddsi_sertopic *tp, void *old, size_t oldcount, size_t count) {
  tp->ops->realloc_samples (ptrs, tp, old, oldcount, count);
}
DDS_INLINE_EXPORT inline void ddsi_sertopic_free_samples (const struct ddsi_sertopic *tp, void **ptrs, size_t count, dds_free_op_t op) {
  tp->ops->free_samples (tp, ptrs, count, op);
}
DDS_INLINE_EXPORT inline void ddsi_sertopic_zero_sample (const struct ddsi_sertopic *tp, void *sample) {
  ddsi_sertopic_zero_samples (tp, sample, 1);
}
DDS_INLINE_EXPORT inline void *ddsi_sertopic_alloc_sample (const struct ddsi_sertopic *tp) {
  void *ptr;
  ddsi_sertopic_realloc_samples (&ptr, tp, NULL, 0, 1);
  return ptr;
}
DDS_INLINE_EXPORT inline void ddsi_sertopic_free_sample (const struct ddsi_sertopic *tp, void *sample, dds_free_op_t op) {
  ddsi_sertopic_free_samples (tp, &sample, 1, op);
}

// THINGS USED FOR TESTING BINARY COMPATIBILITY WITH OLD SERTOPIC INTERFACE

struct ddsi_sertopic_serdata {
  const struct ddsi_sertopic_serdata_ops *ops; /* cached from topic->serdata_ops */
  uint32_t hash;
  ddsrt_atomic_uint32_t refc;
  enum ddsi_serdata_kind kind;
  const struct ddsi_sertopic *topic;

  /* these get set by generic code after creating the serdata */
  ddsrt_wctime_t timestamp;
  uint32_t statusinfo;

  /* FIXME: can I get rid of this one? */
  ddsrt_mtime_t twrite; /* write time, not source timestamp, set post-throttling */
};

typedef uint32_t (*ddsi_sertopic_serdata_size_t) (const struct ddsi_sertopic_serdata *d);
typedef void (*ddsi_sertopic_serdata_free_t) (struct ddsi_sertopic_serdata *d);
typedef struct ddsi_sertopic_serdata * (*ddsi_sertopic_serdata_from_ser_t) (const struct ddsi_sertopic *topic, enum ddsi_serdata_kind kind, const struct nn_rdata *fragchain, size_t size);
typedef struct ddsi_sertopic_serdata * (*ddsi_sertopic_serdata_from_ser_iov_t) (const struct ddsi_sertopic *topic, enum ddsi_serdata_kind kind, ddsrt_msg_iovlen_t niov, const ddsrt_iovec_t *iov, size_t size);
typedef struct ddsi_sertopic_serdata * (*ddsi_sertopic_serdata_from_keyhash_t) (const struct ddsi_sertopic *topic, const struct ddsi_keyhash *keyhash);
typedef struct ddsi_sertopic_serdata * (*ddsi_sertopic_serdata_from_sample_t) (const struct ddsi_sertopic *topic, enum ddsi_serdata_kind kind, const void *sample);
typedef struct ddsi_sertopic_serdata * (*ddsi_sertopic_serdata_to_topicless_t) (const struct ddsi_sertopic_serdata *d);
typedef void (*ddsi_sertopic_serdata_to_ser_t) (const struct ddsi_sertopic_serdata *d, size_t off, size_t sz, void *buf);
typedef struct ddsi_sertopic_serdata * (*ddsi_sertopic_serdata_to_ser_ref_t) (const struct ddsi_sertopic_serdata *d, size_t off, size_t sz, ddsrt_iovec_t *ref);
typedef void (*ddsi_sertopic_serdata_to_ser_unref_t) (struct ddsi_sertopic_serdata *d, const ddsrt_iovec_t *ref);
typedef bool (*ddsi_sertopic_serdata_to_sample_t) (const struct ddsi_sertopic_serdata *d, void *sample, void **bufptr, void *buflim);
typedef bool (*ddsi_sertopic_serdata_topicless_to_sample_t) (const struct ddsi_sertopic *topic, const struct ddsi_sertopic_serdata *d, void *sample, void **bufptr, void *buflim);
typedef bool (*ddsi_sertopic_serdata_eqkey_t) (const struct ddsi_sertopic_serdata *a, const struct ddsi_sertopic_serdata *b);
typedef size_t (*ddsi_sertopic_serdata_print_t) (const struct ddsi_sertopic *topic, const struct ddsi_sertopic_serdata *d, char *buf, size_t size);
typedef void (*ddsi_sertopic_serdata_get_keyhash_t) (const struct ddsi_sertopic_serdata *d, struct ddsi_keyhash *buf, bool force_md5);

struct ddsi_sertopic_serdata_ops {
  ddsi_sertopic_serdata_eqkey_t eqkey;
  ddsi_sertopic_serdata_size_t get_size;
  ddsi_sertopic_serdata_from_ser_t from_ser;
  ddsi_sertopic_serdata_from_ser_iov_t from_ser_iov;
  ddsi_sertopic_serdata_from_keyhash_t from_keyhash;
  ddsi_sertopic_serdata_from_sample_t from_sample;
  ddsi_sertopic_serdata_to_ser_t to_ser;
  ddsi_sertopic_serdata_to_ser_ref_t to_ser_ref;
  ddsi_sertopic_serdata_to_ser_unref_t to_ser_unref;
  ddsi_sertopic_serdata_to_sample_t to_sample;
  ddsi_sertopic_serdata_to_topicless_t to_topicless;
  ddsi_sertopic_serdata_topicless_to_sample_t topicless_to_sample;
  ddsi_sertopic_serdata_free_t free;
  ddsi_sertopic_serdata_print_t print;
  ddsi_sertopic_serdata_get_keyhash_t get_keyhash;
};

DDS_EXPORT void ddsi_sertopic_serdata_init (struct ddsi_sertopic_serdata *d, const struct ddsi_sertopic *topic, enum ddsi_serdata_kind kind);
DDS_EXPORT struct ddsi_sertopic_serdata *ddsi_sertopic_serdata_ref (const struct ddsi_sertopic_serdata *serdata_const);
DDS_EXPORT void ddsi_sertopic_serdata_unref (struct ddsi_sertopic_serdata *serdata);
DDS_EXPORT uint32_t ddsi_sertopic_serdata_size (const struct ddsi_sertopic_serdata *d);
DDS_EXPORT struct ddsi_sertopic_serdata *ddsi_sertopic_serdata_from_ser (const struct ddsi_sertopic *topic, enum ddsi_serdata_kind kind, const struct nn_rdata *fragchain, size_t size);
DDS_EXPORT struct ddsi_sertopic_serdata *ddsi_sertopic_serdata_from_ser_iov (const struct ddsi_sertopic *topic, enum ddsi_serdata_kind kind, ddsrt_msg_iovlen_t niov, const ddsrt_iovec_t *iov, size_t size);
DDS_EXPORT struct ddsi_sertopic_serdata *ddsi_sertopic_serdata_from_keyhash (const struct ddsi_sertopic *topic, const struct ddsi_keyhash *keyhash);
DDS_EXPORT struct ddsi_sertopic_serdata *ddsi_sertopic_serdata_from_sample (const struct ddsi_sertopic *topic, enum ddsi_serdata_kind kind, const void *sample);
DDS_EXPORT struct ddsi_sertopic_serdata *ddsi_sertopic_serdata_to_topicless (const struct ddsi_sertopic_serdata *d);
DDS_EXPORT void ddsi_sertopic_serdata_to_ser (const struct ddsi_sertopic_serdata *d, size_t off, size_t sz, void *buf);
DDS_EXPORT struct ddsi_sertopic_serdata *ddsi_sertopic_serdata_to_ser_ref (const struct ddsi_sertopic_serdata *d, size_t off, size_t sz, ddsrt_iovec_t *ref);
DDS_EXPORT void ddsi_sertopic_serdata_to_ser_unref (struct ddsi_sertopic_serdata *d, const ddsrt_iovec_t *ref);
DDS_EXPORT bool ddsi_sertopic_serdata_to_sample (const struct ddsi_sertopic_serdata *d, void *sample, void **bufptr, void *buflim);
DDS_EXPORT bool ddsi_sertopic_serdata_topicless_to_sample (const struct ddsi_sertopic *topic, const struct ddsi_sertopic_serdata *d, void *sample, void **bufptr, void *buflim);
DDS_EXPORT bool ddsi_sertopic_serdata_eqkey (const struct ddsi_sertopic_serdata *a, const struct ddsi_sertopic_serdata *b);
DDS_EXPORT bool ddsi_sertopic_serdata_print (const struct ddsi_sertopic_serdata *d, char *buf, size_t size);
DDS_EXPORT bool ddsi_sertopic_serdata_print_topicless (const struct ddsi_sertopic *topic, const struct ddsi_sertopic_serdata *d, char *buf, size_t size);
DDS_EXPORT void ddsi_sertopic_serdata_get_keyhash (const struct ddsi_sertopic_serdata *d, struct ddsi_keyhash *buf, bool force_md5);

#if defined (__cplusplus)
}
#endif

#endif
