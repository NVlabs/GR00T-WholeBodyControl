/*
 * Copyright(c) 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDS_SHM__TRANSPORT_H
#define DDS_SHM__TRANSPORT_H

#include "dds/export.h"
#include "dds/features.h"

#include "dds/ddsi/ddsi_config.h"
#include "dds/ddsi/ddsi_keyhash.h"
#include "dds/ddsi/ddsi_tran.h"
#include "dds/ddsi/q_protocol.h" /* for, e.g., SubmessageKind_t */
#include "dds/ddsrt/sync.h"

#include "iceoryx_binding_c/chunk.h"
#include "iceoryx_binding_c/publisher.h"
#include "iceoryx_binding_c/subscriber.h"
#include "iceoryx_binding_c/config.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  IOX_CHUNK_UNINITIALIZED,
  IOX_CHUNK_CONTAINS_RAW_DATA,
  IOX_CHUNK_CONTAINS_SERIALIZED_DATA
} iox_shm_data_state_t;

struct iceoryx_header {
  struct ddsi_guid guid;
  dds_time_t tstamp;
  uint32_t statusinfo;
  uint32_t data_size;
  unsigned char data_kind;
  ddsi_keyhash_t keyhash;
  iox_shm_data_state_t shm_data_state;
};

typedef struct iceoryx_header iceoryx_header_t;

struct dds_reader;
struct shm_monitor;

typedef struct {  
  ddsrt_mutex_t mutex;
  struct shm_monitor *monitor;
  struct dds_reader *parent_reader;
} iox_sub_context_t;

DDS_EXPORT iox_sub_context_t **iox_sub_context_ptr(iox_sub_t sub);

DDS_EXPORT void iox_sub_context_init(iox_sub_context_t *context);

DDS_EXPORT void
iox_sub_context_fini(iox_sub_context_t *context);

// lock and unlock for individual subscribers
DDS_EXPORT void shm_lock_iox_sub(iox_sub_t sub);

DDS_EXPORT void shm_unlock_iox_sub(iox_sub_t sub);

DDS_EXPORT void free_iox_chunk(iox_sub_t *iox_sub, void **iox_chunk);

DDS_EXPORT iceoryx_header_t *iceoryx_header_from_chunk(const void *iox_chunk);

DDS_EXPORT void shm_set_loglevel(enum ddsi_shm_loglevel);

DDS_EXPORT void *shm_create_chunk(iox_pub_t iox_pub, size_t size);

DDS_EXPORT void shm_set_data_state(void *iox_chunk,
                                   iox_shm_data_state_t data_state);

DDS_EXPORT iox_shm_data_state_t shm_get_data_state(void *iox_chunk);

#if defined(__cplusplus)
}
#endif

#endif // DDS_SHM__TRANSPORT_H
