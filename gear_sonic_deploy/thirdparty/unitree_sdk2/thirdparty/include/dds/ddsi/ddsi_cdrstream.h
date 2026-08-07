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
#ifndef DDSI_CDRSTREAM_H
#define DDSI_CDRSTREAM_H

#include "dds/ddsrt/static_assert.h"
#include "dds/ddsi/ddsi_serdata.h"
#include "dds/ddsi/ddsi_serdata_default.h"

#if defined (__cplusplus)
extern "C" {
#endif

#define DDSI_CDRSTREAM_MAX_NESTING_DEPTH 32  /* maximum level of nesting for key extraction */

typedef struct dds_istream {
  const unsigned char *m_buffer;
  uint32_t m_size;          /* Buffer size */
  uint32_t m_index;         /* Read/write offset from start of buffer */
  uint32_t m_xcdr_version;  /* XCDR version of the data */
} dds_istream_t;

typedef struct dds_ostream {
  unsigned char *m_buffer;
  uint32_t m_size;          /* Buffer size */
  uint32_t m_index;         /* Read/write offset from start of buffer */
  uint32_t m_xcdr_version;  /* XCDR version to use for serializing data */
} dds_ostream_t;

typedef struct dds_ostreamBE {
  dds_ostream_t x;
} dds_ostreamBE_t;

typedef struct dds_ostreamLE {
  dds_ostream_t x;
} dds_ostreamLE_t;

DDSRT_STATIC_ASSERT (offsetof (dds_ostreamLE_t, x) == 0);
DDSRT_STATIC_ASSERT (offsetof (dds_ostreamBE_t, x) == 0);

DDS_EXPORT void dds_istream_init (dds_istream_t * __restrict st, uint32_t size, const void * __restrict input, uint32_t xcdr_version);
DDS_EXPORT void dds_istream_fini (dds_istream_t * __restrict st);
DDS_EXPORT void dds_ostream_init (dds_ostream_t * __restrict st, uint32_t size, uint32_t xcdr_version);
DDS_EXPORT void dds_ostream_fini (dds_ostream_t * __restrict st);
DDS_EXPORT void dds_ostreamLE_init (dds_ostreamLE_t * __restrict st, uint32_t size, uint32_t xcdr_version);
DDS_EXPORT void dds_ostreamLE_fini (dds_ostreamLE_t * __restrict st);
DDS_EXPORT void dds_ostreamBE_init (dds_ostreamBE_t * __restrict st, uint32_t size, uint32_t xcdr_version);
DDS_EXPORT void dds_ostreamBE_fini (dds_ostreamBE_t * __restrict st);

// *actual_size is set to the actual size of the data (*actual_size <= size) on successful return
DDS_EXPORT bool dds_stream_normalize (void * __restrict data, uint32_t size, bool bswap, uint32_t xcdr_version, const struct ddsi_sertype_default * __restrict type, bool just_key, uint32_t * __restrict actual_size) ddsrt_attribute_warn_unused_result ddsrt_nonnull_all;
DDS_EXPORT const uint32_t *dds_stream_normalize_data (char * __restrict data, uint32_t * __restrict off, uint32_t size, bool bswap, uint32_t xcdr_version, const uint32_t * __restrict ops) ddsrt_attribute_warn_unused_result ddsrt_nonnull_all;

DDS_EXPORT const uint32_t *dds_stream_write (dds_ostream_t * __restrict os, const char * __restrict data, const uint32_t * __restrict ops);
DDS_EXPORT const uint32_t *dds_stream_writeLE (dds_ostreamLE_t * __restrict os, const char * __restrict data, const uint32_t * __restrict ops);
DDS_EXPORT const uint32_t *dds_stream_writeBE (dds_ostreamBE_t * __restrict os, const char * __restrict data, const uint32_t * __restrict ops);
DDS_EXPORT const uint32_t * dds_stream_write_with_byte_order (dds_ostream_t * __restrict os, const char * __restrict data, const uint32_t * __restrict ops, enum ddsrt_byte_order_selector bo);
DDS_EXPORT bool dds_stream_write_sample (dds_ostream_t * __restrict os, const void * __restrict data, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT bool dds_stream_write_sampleLE (dds_ostreamLE_t * __restrict os, const void * __restrict data, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT bool dds_stream_write_sampleBE (dds_ostreamBE_t * __restrict os, const void * __restrict data, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT void dds_stream_read_sample (dds_istream_t * __restrict is, void * __restrict data, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT void dds_stream_free_sample (void * __restrict data, const uint32_t * __restrict ops);

DDS_EXPORT uint32_t dds_stream_countops (const uint32_t * __restrict ops, uint32_t nkeys, const dds_key_descriptor_t * __restrict keys);
DDS_EXPORT size_t dds_stream_check_optimize (const struct ddsi_sertype_default_desc * __restrict desc, uint32_t xcdr_version);
DDS_EXPORT void dds_istream_from_serdata_default (dds_istream_t * __restrict s, const struct ddsi_serdata_default * __restrict d);
DDS_EXPORT void dds_ostream_from_serdata_default (dds_ostream_t * __restrict s, const struct ddsi_serdata_default * __restrict d);
DDS_EXPORT void dds_ostream_add_to_serdata_default (dds_ostream_t * __restrict s, struct ddsi_serdata_default ** __restrict d);

DDS_EXPORT void dds_stream_write_key (dds_ostream_t * __restrict os, const char * __restrict sample, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT void dds_stream_write_keyBE (dds_ostreamBE_t * __restrict os, const char * __restrict sample, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT bool dds_stream_extract_key_from_data (dds_istream_t * __restrict is, dds_ostream_t * __restrict os, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT void dds_stream_extract_key_from_key (dds_istream_t * __restrict is, dds_ostream_t * __restrict os, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT bool dds_stream_extract_keyBE_from_data (dds_istream_t * __restrict is, dds_ostreamBE_t * __restrict os, const struct ddsi_sertype_default * __restrict type);
DDS_EXPORT void dds_stream_extract_keyBE_from_key (dds_istream_t * __restrict is, dds_ostreamBE_t * __restrict os, const struct ddsi_sertype_default * __restrict type);

DDS_EXPORT const uint32_t *dds_stream_read (dds_istream_t * __restrict is, char * __restrict data, const uint32_t * __restrict ops);
DDS_EXPORT void dds_stream_read_key (dds_istream_t * __restrict is, char * __restrict sample, const struct ddsi_sertype_default * __restrict type);

DDS_EXPORT size_t dds_stream_print_key (dds_istream_t * __restrict is, const struct ddsi_sertype_default * __restrict type, char * __restrict buf, size_t size);

DDS_EXPORT size_t dds_stream_print_sample (dds_istream_t * __restrict is, const struct ddsi_sertype_default * __restrict type, char * __restrict buf, size_t size);

DDS_EXPORT uint16_t dds_stream_minimum_xcdr_version (const uint32_t * __restrict ops);
DDS_EXPORT uint32_t dds_stream_type_nesting_depth (const uint32_t * __restrict ops);
DDS_EXPORT bool dds_stream_extensibility (const uint32_t * __restrict ops, enum ddsi_sertype_extensibility *ext);


#if defined (__cplusplus)
}
#endif
#endif
