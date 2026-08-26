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
#ifndef DDSI_ENTITY_H
#define DDSI_ENTITY_H

#include "dds/export.h"
#include "dds/features.h"

#include "dds/ddsrt/atomics.h"
#include "dds/ddsrt/avl.h"
#include "dds/ddsrt/fibheap.h"
#include "dds/ddsrt/sync.h"
#include "dds/ddsi/q_rtps.h"
#include "dds/ddsi/ddsi_plist.h"
#include "dds/ddsi/q_protocol.h"
#include "dds/ddsi/q_lat_estim.h"
#include "dds/ddsi/q_hbcontrol.h"
#include "dds/ddsi/q_feature_check.h"
#include "dds/ddsi/q_inverse_uint32_set.h"
#include "dds/ddsi/ddsi_typelib.h"
#include "dds/ddsi/ddsi_serdata_default.h"
#include "dds/ddsi/ddsi_handshake.h"
#include "dds/ddsi/ddsi_typelookup.h"
#include "dds/ddsi/ddsi_tran.h"
#include "dds/ddsi/ddsi_list_genptr.h"
#include "dds/ddsi/q_gc.h"
#include "dds/ddsi/q_lease.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct whc_node;
struct ddsi_plist;
struct nn_rsample_info;
struct nn_rdata;
struct ddsi_tkmap_instance;
struct ddsi_writer_info;
struct entity_index;

typedef void (*ddsi2direct_directread_cb_t) (const struct nn_rsample_info *sampleinfo, const struct nn_rdata *fragchain, void *arg);

enum ddsi_entity_kind {
  DDSI_EK_PARTICIPANT,
  DDSI_EK_PROXY_PARTICIPANT,
  DDSI_EK_TOPIC,
  DDSI_EK_WRITER,
  DDSI_EK_PROXY_WRITER,
  DDSI_EK_READER,
  DDSI_EK_PROXY_READER
};

typedef struct ddsi_status_cb_data
{
  int raw_status_id;
  uint32_t extra;
  uint64_t handle;
  bool add;
} ddsi_status_cb_data_t;

typedef void (*ddsi_status_cb_t) (void *entity, const ddsi_status_cb_data_t *data);

typedef struct ddsi_type_pair
#ifdef DDS_HAS_TYPE_DISCOVERY
{
  struct ddsi_type *minimal;
  struct ddsi_type *complete;
}
#endif
ddsi_type_pair_t;

struct ddsi_entity_common {
  enum ddsi_entity_kind kind;
  ddsi_guid_t guid;
  ddsrt_wctime_t tupdate; /* timestamp of last update */
  uint64_t iid;
  struct ddsi_tkmap_instance *tk;
  ddsrt_mutex_t lock;
  bool onlylocal;
  struct ddsi_domaingv *gv;
  ddsrt_avl_node_t all_entities_avlnode;

  /* QoS changes always lock the entity itself, and additionally
     (and within the scope of the entity lock) acquire qos_lock
     while manipulating the QoS.  So any thread that needs to read
     the QoS without acquiring the entity's lock can still do so
     (e.g., the materialisation of samples for built-in topics
     when connecting a reader to a writer for a built-in topic).

     qos_lock lock order across entities in is in increasing
     order of entity addresses cast to uintptr_t. */
  ddsrt_mutex_t qos_lock;
};

struct ddsi_local_reader_ary {
  ddsrt_mutex_t rdary_lock;
  unsigned valid: 1; /* always true until (proxy-)writer is being deleted; !valid => !fastpath_ok */
  unsigned fastpath_ok: 1; /* if not ok, fall back to using GUIDs (gives access to the reader-writer match data for handling readers that bumped into resource limits, hence can flip-flop, unlike "valid") */
  uint32_t n_readers;
  struct ddsi_reader **rdary; /* for efficient delivery, null-pointer terminated, grouped by topic */
};

struct ddsi_alive_state {
  bool alive;
  uint32_t vclock;
};

bool ddsi_is_null_guid (const ddsi_guid_t *guid);
int ddsi_is_builtin_entityid (ddsi_entityid_t id, nn_vendorid_t vendorid);
bool ddsi_update_qos_locked (struct ddsi_entity_common *e, dds_qos_t *ent_qos, const dds_qos_t *xqos, ddsrt_wctime_t timestamp);
int ddsi_set_topic_type_name (dds_qos_t *xqos, const char * topic_name, const char * type_name);
int ddsi_compare_entityid (const void *a, const void *b);

DDS_EXPORT int ddsi_compare_guid (const void *va, const void *vb);
DDS_EXPORT ddsi_entityid_t ddsi_to_entityid (unsigned u);
DDS_EXPORT nn_vendorid_t ddsi_get_entity_vendorid (const struct ddsi_entity_common *e);
DDS_EXPORT uint64_t ddsi_get_entity_instanceid (const struct ddsi_domaingv *gv, const struct ddsi_guid *guid);
DDS_EXPORT void ddsi_entity_common_init (struct ddsi_entity_common *e, struct ddsi_domaingv *gv, const struct ddsi_guid *guid, enum ddsi_entity_kind kind, ddsrt_wctime_t tcreate, nn_vendorid_t vendorid, bool onlylocal);
DDS_EXPORT void ddsi_entity_common_fini (struct ddsi_entity_common *e);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_ENTITY_H */
