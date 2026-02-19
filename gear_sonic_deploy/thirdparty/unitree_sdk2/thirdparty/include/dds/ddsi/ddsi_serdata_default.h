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
#ifndef DDSI_SERDATA_DEFAULT_H
#define DDSI_SERDATA_DEFAULT_H

#include "dds/ddsrt/endian.h"
#include "dds/ddsi/q_protocol.h" /* for nn_parameterid_t */
#include "dds/ddsi/q_freelist.h"
#include "dds/ddsrt/avl.h"
#include "dds/ddsi/ddsi_serdata.h"
#include "dds/ddsi/ddsi_plist_generic.h"
#include "dds/ddsi/ddsi_typelib.h"
#include "dds/ddsi/ddsi_typelookup.h"

#include "dds/dds.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct CDRHeader {
  unsigned short identifier;
  unsigned short options;
};

struct serdatapool {
  struct nn_freelist freelist;
};

#define KEYBUFTYPE_UNSET    0u
#define KEYBUFTYPE_STATIC   1u // uses u.stbuf
#define KEYBUFTYPE_DYNALIAS 2u // points into payload
#define KEYBUFTYPE_DYNALLOC 3u // dynamically allocated

#define SERDATA_DEFAULT_KEYSIZE_MASK        0x3FFFFFFFu

struct ddsi_serdata_default_key {
  unsigned buftype : 2;
  unsigned keysize : 30;
  union {
    unsigned char stbuf[DDS_FIXED_KEY_MAX_SIZE];
    unsigned char *dynbuf;
  } u;
};

/* Debug builds may want to keep some additional state */
#ifndef NDEBUG
#define DDSI_SERDATA_DEFAULT_DEBUG_FIELDS \
  bool fixed;
#else
#define DDSI_SERDATA_DEFAULT_DEBUG_FIELDS
#endif

/* There is an alignment requirement on the raw data (it must be at
   offset mod 8 for the conversion to/from a dds_stream to work).
   So we define two types: one without any additional padding, and
   one where the appropriate amount of padding is inserted */
#define DDSI_SERDATA_DEFAULT_PREPAD   \
  struct ddsi_serdata c;              \
  uint32_t pos;                       \
  uint32_t size;                      \
  DDSI_SERDATA_DEFAULT_DEBUG_FIELDS   \
  struct ddsi_serdata_default_key key;\
  struct serdatapool *serpool;        \
  struct ddsi_serdata_default *next /* in pool->freelist */
/* We suppress the zero-array warning (MSVC C4200) here ONLY for MSVC
   and ONLY if it is being compiled as C++ code, as it only causes
   issues there */
#if defined _MSC_VER && defined __cplusplus
#define DDSI_SERDATA_DEFAULT_POSTPAD  \
  struct CDRHeader hdr;               \
  DDSRT_WARNING_MSVC_OFF(4200)        \
  char data[];                        \
  DDSRT_WARNING_MSVC_ON(4200)
#else
#define DDSI_SERDATA_DEFAULT_POSTPAD  \
  struct CDRHeader hdr;               \
  char data[]
#endif

struct ddsi_serdata_default_unpadded {
  DDSI_SERDATA_DEFAULT_PREPAD;
  DDSI_SERDATA_DEFAULT_POSTPAD;
};

#ifdef __GNUC__
#define DDSI_SERDATA_DEFAULT_PAD(n) ((n) % 8)
#else
#define DDSI_SERDATA_DEFAULT_PAD(n) (n)
#endif

struct ddsi_serdata_default {
  DDSI_SERDATA_DEFAULT_PREPAD;
  char pad[DDSI_SERDATA_DEFAULT_PAD (8 - (offsetof (struct ddsi_serdata_default_unpadded, data) % 8))];
  DDSI_SERDATA_DEFAULT_POSTPAD;
};

#undef DDSI_SERDATA_DEFAULT_PAD
#undef DDSI_SERDATA_DEFAULT_POSTPAD
#undef DDSI_SERDATA_DEFAULT_PREPAD
#undef DDSI_SERDATA_DEFAULT_FIXED_FIELD

#ifndef DDS_TOPIC_INTERN_FILTER_FN_DEFINED
#define DDS_TOPIC_INTERN_FILTER_FN_DEFINED
typedef bool (*dds_topic_intern_filter_fn) (const void * sample, void *ctx);
#endif

typedef struct ddsi_sertype_default_desc_key {
  uint32_t ops_offs;   /* Offset for key ops */
  uint32_t idx;        /* Key index (used for key order) */
} ddsi_sertype_default_desc_key_t;

typedef struct ddsi_sertype_default_desc_key_seq {
  uint32_t nkeys;
  ddsi_sertype_default_desc_key_t *keys;
} ddsi_sertype_default_desc_key_seq_t;

typedef struct ddsi_sertype_default_desc_op_seq {
  uint32_t nops;    /* Number of words in ops (which >= number of ops stored in preproc output) */
  uint32_t *ops;    /* Marshalling meta data */
} ddsi_sertype_default_desc_op_seq_t;

/* Reduced version of dds_topic_descriptor_t */
struct ddsi_sertype_default_desc {
  uint32_t size;    /* Size of topic type */
  uint32_t align;   /* Alignment of topic type */
  uint32_t flagset; /* Flags */
  ddsi_sertype_default_desc_key_seq_t keys;
  ddsi_sertype_default_desc_op_seq_t ops;
  ddsi_sertype_cdr_data_t typeinfo_ser;
  ddsi_sertype_cdr_data_t typemap_ser;
};

struct ddsi_sertype_default {
  struct ddsi_sertype c;
  uint16_t encoding_format; /* CDR_ENC_FORMAT_(PLAIN|DELIMITED|PL) - CDR encoding format for the top-level type in this sertype */
  uint16_t write_encoding_version; /* CDR_ENC_VERSION_(1|2) - CDR encoding version used for writing data using this sertype */
  struct serdatapool *serpool;
  struct ddsi_sertype_default_desc type;
  size_t opt_size_xcdr1;
  size_t opt_size_xcdr2;
};

struct ddsi_plist_sample {
  void *blob;
  size_t size;
  nn_parameterid_t keyparam;
};

struct ddsi_rawcdr_sample {
  void *blob;
  size_t size;
  void *key;
  size_t keysize;
};

extern DDS_EXPORT const struct ddsi_sertype_ops ddsi_sertype_ops_default;

extern DDS_EXPORT const struct ddsi_serdata_ops ddsi_serdata_ops_cdr;
extern DDS_EXPORT const struct ddsi_serdata_ops ddsi_serdata_ops_cdr_nokey;

extern DDS_EXPORT const struct ddsi_serdata_ops ddsi_serdata_ops_xcdr2;
extern DDS_EXPORT const struct ddsi_serdata_ops ddsi_serdata_ops_xcdr2_nokey;

struct serdatapool * ddsi_serdatapool_new (void);
void ddsi_serdatapool_free (struct serdatapool * pool);
dds_return_t ddsi_sertype_default_init (const struct ddsi_domaingv *gv, struct ddsi_sertype_default *st, const dds_topic_descriptor_t *desc, uint16_t min_xcdrv, dds_data_representation_id_t data_representation);

#if defined (__cplusplus)
}
#endif

#endif
