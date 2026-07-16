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
#ifndef DDSI_SERTYPE_H
#define DDSI_SERTYPE_H

#include "dds/features.h"
#include "dds/ddsrt/atomics.h"
#include "dds/ddsrt/avl.h"
#include "dds/ddsi/ddsi_xqos.h"
#include "dds/ddsi/ddsi_typelib.h"
#include "dds/ddsc/dds_public_alloc.h"
#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_serdata;
struct ddsi_serdata_ops;
struct ddsi_sertype_ops;
struct ddsi_domaingv;
struct ddsi_typeid;
struct ddsi_type_pair;

#define DDSI_SERTYPE_REGISTERING 0x40000000u // set prior to setting gv
#define DDSI_SERTYPE_REGISTERED  0x80000000u // set after setting gv
#define DDSI_SERTYPE_REFC_MASK   0x0fffffffu

typedef struct ddsi_sertype_cdr_data {
  uint32_t sz;
  uint8_t *data;
} ddsi_sertype_cdr_data_t;

struct ddsi_sertype {
  const struct ddsi_sertype_ops *ops;
  const struct ddsi_serdata_ops *serdata_ops;
  uint32_t serdata_basehash;
  uint32_t typekind_no_key : 1;
  uint32_t request_keyhash : 1;
  uint32_t fixed_size : 1;
  uint32_t allowed_data_representation; /* Allowed data representations set in IDL for this type, or DDS_DATA_REPRESENTATION_RESTRICT_DEFAULT in case of
                                            no restrictions in the IDL. Unsupported representations for the type are left out when creating the sertype. */
  char *type_name;
  ddsrt_atomic_voidp_t gv; /* set during registration */
  ddsrt_atomic_uint32_t flags_refc; /* counts refs from entities (topic, reader, writer), not from data */
  const struct ddsi_sertype *base_sertype; /* counted ref to sertype used to derive this sertype, used to overwrite the serdata_ops for a specific data representation */
  void *wrapped_sertopic; /* void pointer because that simplifies the required type-punning */
#ifdef DDS_HAS_SHM
  uint32_t iox_size;
#endif
};

/* The old and the new happen to have the same memory layout on a 64-bit machine
   and so any user that memset's the ddsi_sertype to 0 before filling out the
   required fields gets unchanged behaviour.  32-bit machines have a different
   layout and no such luck.

   There are presumably very few users of this type outside Cyclone DDS itself,
   but the ROS2 RMW implementation does use it -- indeed, it prompted the change.
   This define makes it possible to have a single version of the source that is
   compatible with the old and the new definition, even if it is only partially
   binary compatible. */
#define DDSI_SERTOPIC_HAS_TOPICKIND_NO_KEY 1

/* Type changed: name_type_name and ii removed and gv added; and the set of
   operations got extended by the a predicate for testing to sertypes (with the
   same "ops") for equality ("equal") as well as a function for hashing the
   non-generic part of the sertype definition (via "hash").  These two operations
   make it possible to intern sertypes without duplicates, which has become
   relevant now that multiple ddsi_sertypes can be associated with a single topic
   name.

   Testing for DDSI_SERTOPIC_HAS_EQUAL_AND_HASH allows one to have a single source
   that can handle both variants, but there's no binary compatbility. */
#define DDSI_SERTOPIC_HAS_EQUAL_AND_HASH 1

/* It was a bad decision to have a boolean argument in "init" specifying whether
   the entity kind should say "with key" or "without key".  A general "flags"
   argument is much more flexible ... */
#define DDSI_SERTYPE_HAS_SERTYPE_INIT_FLAGS 1

/* Called to compare two sertypes for equality, if it is already known that
   type name, kind_no_Key, and operations are all the same.  (serdata_basehash
   is computed from the set of operations.) */
typedef bool (*ddsi_sertype_equal_t) (const struct ddsi_sertype *a, const struct ddsi_sertype *b);

/* Hash the custom components of a sertype (this XOR'd with a hash computed from
   the fields that are defined in struct ddsi_sertype) */
typedef uint32_t (*ddsi_sertype_hash_t) (const struct ddsi_sertype *tp);

/* Called when the refcount dropped to zero */
typedef void (*ddsi_sertype_free_t) (struct ddsi_sertype *tp);

/* Zero out a sample, used for generating samples from just a key value and in cleaning up
   after dds_return_loan */
typedef void (*ddsi_sertype_zero_samples_t) (const struct ddsi_sertype *d, void *samples, size_t count);

/* (Re)allocate an array of samples, used in growing loaned sample arrays in dds_read */
typedef void (*ddsi_sertype_realloc_samples_t) (void **ptrs, const struct ddsi_sertype *d, void *old, size_t oldcount, size_t count);

/* Release any memory allocated by ddsi_sertype_to_sample (also undo sertype_alloc_sample if "op" so requests) */
typedef void (*ddsi_sertype_free_samples_t) (const struct ddsi_sertype *d, void **ptrs, size_t count, dds_free_op_t op);

/* Gets the type identifier of the requested kind (minimal or complete) for this sertype */
typedef ddsi_typeid_t * (*ddsi_sertype_typeid_t) (const struct ddsi_sertype *tp, ddsi_typeid_kind_t kind);

/* Compute the serialized size based on the sertype information and the sample */
// Note: size_t maximum is reserved as error value
typedef size_t (*ddsi_sertype_get_serialized_size_t)(
    const struct ddsi_sertype *d, const void *sample);

/* Serialize into a destination buffer */
// Note that we assume the destination buffer is large enough (we do not necessarily check)
// The required size can be obtained with ddsi_sertype_get_serialized_size_t
// Returns true if the serialization succeeds, false otherwise.
typedef bool (*ddsi_sertype_serialize_into_t)(const struct ddsi_sertype *d,
                                              const void *sample,
                                              void *dst_buffer,
                                              size_t dst_size);

/* Gets the type map for this sertype */
typedef ddsi_typemap_t * (*ddsi_sertype_typemap_t) (const struct ddsi_sertype *tp);

/* Gets the CDR blob for the type information of this sertype */
typedef ddsi_typeinfo_t * (*ddsi_sertype_typeinfo_t) (const struct ddsi_sertype *tp);

/* Create a new derived sertype (a shallow copy of the provided sertype) with the
   serdata_ops for the provided data representation */
typedef struct ddsi_sertype * (*ddsi_sertype_derive_t) (const struct ddsi_sertype *sertype, dds_data_representation_id_t data_representation, dds_type_consistency_enforcement_qospolicy_t tce_qos);

struct ddsi_sertype_v0;
typedef void (*ddsi_sertype_v0_t) (struct ddsi_sertype_v0 *dummy);

// Because Windows ... just can't get its act together ...
#ifndef _WIN32
DDS_EXPORT void ddsi_sertype_v0 (struct ddsi_sertype_v0 *dummy);
#else
#define ddsi_sertype_v0 ((ddsi_sertype_v0_t) 1)
#endif

struct ddsi_sertype_ops {
  ddsi_sertype_v0_t version;
  void *arg;

  ddsi_sertype_free_t free;
  ddsi_sertype_zero_samples_t zero_samples;
  ddsi_sertype_realloc_samples_t realloc_samples;
  ddsi_sertype_free_samples_t free_samples;
  ddsi_sertype_equal_t equal;
  ddsi_sertype_hash_t hash;
  ddsi_sertype_typeid_t type_id;
  ddsi_sertype_typemap_t type_map;
  ddsi_sertype_typeinfo_t type_info;
  ddsi_sertype_derive_t derive_sertype;
  ddsi_sertype_get_serialized_size_t get_serialized_size;
  ddsi_sertype_serialize_into_t serialize_into;
};

enum ddsi_sertype_extensibility
{
  DDSI_SERTYPE_EXT_FINAL = 0,
  DDSI_SERTYPE_EXT_APPENDABLE = 1,
  DDSI_SERTYPE_EXT_MUTABLE = 2
};

struct ddsi_sertype *ddsi_sertype_lookup_locked (struct ddsi_domaingv *gv, const struct ddsi_sertype *sertype_template);
void ddsi_sertype_register_locked (struct ddsi_domaingv *gv, struct ddsi_sertype *sertype);

#define DDSI_SERTYPE_FLAG_TOPICKIND_NO_KEY (1u)
#define DDSI_SERTYPE_FLAG_REQUEST_KEYHASH  (2u)
#define DDSI_SERTYPE_FLAG_FIXED_SIZE       (4u)

#define DDSI_SERTYPE_FLAG_MASK (0x7u)

DDS_EXPORT void ddsi_sertype_init_flags (struct ddsi_sertype *tp, const char *type_name, const struct ddsi_sertype_ops *sertype_ops, const struct ddsi_serdata_ops *serdata_ops, uint32_t flags);
DDS_EXPORT void ddsi_sertype_init (struct ddsi_sertype *tp, const char *type_name, const struct ddsi_sertype_ops *sertype_ops, const struct ddsi_serdata_ops *serdata_ops, bool topickind_no_key);
DDS_EXPORT void ddsi_sertype_fini (struct ddsi_sertype *tp);
DDS_EXPORT struct ddsi_sertype *ddsi_sertype_ref (const struct ddsi_sertype *tp);

DDS_EXPORT void ddsi_sertype_unref_locked (struct ddsi_domaingv * const gv, struct ddsi_sertype *tp); /* gv->sertypes_lock must be held */
DDS_EXPORT void ddsi_sertype_unref (struct ddsi_sertype *tp); /* tp->gv->sertypes_lock may not be held */

DDS_EXPORT uint32_t ddsi_sertype_compute_serdata_basehash (const struct ddsi_serdata_ops *ops);

DDS_EXPORT bool ddsi_sertype_equal (const struct ddsi_sertype *a, const struct ddsi_sertype *b);
DDS_EXPORT uint32_t ddsi_sertype_hash (const struct ddsi_sertype *tp);

DDS_EXPORT uint16_t ddsi_sertype_extensibility_enc_format (enum ddsi_sertype_extensibility type_extensibility);
DDS_EXPORT uint16_t ddsi_sertype_get_native_enc_identifier (uint32_t enc_version, uint32_t encoding_format);
DDS_EXPORT uint32_t ddsi_sertype_enc_id_xcdr_version (uint16_t cdr_identifier);
DDS_EXPORT uint32_t ddsi_sertype_enc_id_enc_format (uint16_t cdr_identifier);

DDS_INLINE_EXPORT inline void ddsi_sertype_free (struct ddsi_sertype *tp) {
  tp->ops->free (tp);
}
DDS_INLINE_EXPORT inline void ddsi_sertype_zero_samples (const struct ddsi_sertype *tp, void *samples, size_t count) {
  tp->ops->zero_samples (tp, samples, count);
}
DDS_INLINE_EXPORT inline void ddsi_sertype_realloc_samples (void **ptrs, const struct ddsi_sertype *tp, void *old, size_t oldcount, size_t count)
{
  tp->ops->realloc_samples (ptrs, tp, old, oldcount, count);
}
DDS_INLINE_EXPORT inline void ddsi_sertype_free_samples (const struct ddsi_sertype *tp, void **ptrs, size_t count, dds_free_op_t op) {
  tp->ops->free_samples (tp, ptrs, count, op);
}
DDS_INLINE_EXPORT inline void ddsi_sertype_zero_sample (const struct ddsi_sertype *tp, void *sample) {
  ddsi_sertype_zero_samples (tp, sample, 1);
}
DDS_INLINE_EXPORT inline void *ddsi_sertype_alloc_sample (const struct ddsi_sertype *tp) {
  void *ptr;
  ddsi_sertype_realloc_samples (&ptr, tp, NULL, 0, 1);
  return ptr;
}
DDS_INLINE_EXPORT inline void ddsi_sertype_free_sample (const struct ddsi_sertype *tp, void *sample, dds_free_op_t op) {
  ddsi_sertype_free_samples (tp, &sample, 1, op);
}
DDS_INLINE_EXPORT inline ddsi_typeid_t * ddsi_sertype_typeid (const struct ddsi_sertype *tp, ddsi_typeid_kind_t kind)
{
  if (!tp->ops->type_id)
    return NULL;
  return tp->ops->type_id (tp, kind);
}
DDS_INLINE_EXPORT inline ddsi_typemap_t * ddsi_sertype_typemap (const struct ddsi_sertype *tp)
{
  if (!tp->ops->type_map)
    return NULL;
  return tp->ops->type_map (tp);
}
DDS_INLINE_EXPORT inline ddsi_typeinfo_t *ddsi_sertype_typeinfo (const struct ddsi_sertype *tp)
{
  if (!tp->ops->type_info)
    return NULL;
  return tp->ops->type_info (tp);
}
DDS_INLINE_EXPORT inline struct ddsi_sertype * ddsi_sertype_derive_sertype (const struct ddsi_sertype *base_sertype, dds_data_representation_id_t data_representation, dds_type_consistency_enforcement_qospolicy_t tce_qos) {
  if (!base_sertype->ops->derive_sertype)
    return NULL;
  return base_sertype->ops->derive_sertype (base_sertype, data_representation, tce_qos);
}

DDS_INLINE_EXPORT inline size_t
ddsi_sertype_get_serialized_size(const struct ddsi_sertype *tp,
                                 const void *sample) {
  return tp->ops->get_serialized_size(tp, sample);
}

DDS_INLINE_EXPORT inline bool
ddsi_sertype_serialize_into(const struct ddsi_sertype *tp, const void *sample,
                            void *dst_buffer, size_t dst_size) {
  return tp->ops->serialize_into(tp, sample, dst_buffer, dst_size);
}

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_SERTYPE_H */
