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
#ifndef DDSI_PROXY_ENDPOINT_H
#define DDSI_PROXY_ENDPOINT_H

#include "dds/export.h"
#include "dds/features.h"

#include "dds/ddsrt/avl.h"
#include "dds/ddsrt/atomics.h"
#include "dds/ddsi/ddsi_guid.h"
#include "dds/ddsi/ddsi_domaingv.h"
#include "dds/ddsi/ddsi_entity.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_proxy_participant;
struct dds_qos;
struct addrset;

struct ddsi_proxy_endpoint_common
{
  struct ddsi_proxy_participant *proxypp; /* counted backref to proxy participant */
  struct ddsi_proxy_endpoint_common *next_ep; /* next \ endpoint belonging to this proxy participant */
  struct ddsi_proxy_endpoint_common *prev_ep; /* prev / -- this is in arbitrary ordering */
  struct dds_qos *xqos; /* proxy endpoint QoS lives here; FIXME: local ones should have it moved to common as well */
  struct addrset *as; /* address set to use for communicating with this endpoint */
  ddsi_guid_t group_guid; /* 0:0:0:0 if not available */
  nn_vendorid_t vendor; /* cached from proxypp->vendor */
  seqno_t seq; /* sequence number of most recent SEDP message */
#ifdef DDS_HAS_TYPE_DISCOVERY
  struct ddsi_type_pair *type_pair;
#endif
#ifdef DDS_HAS_SECURITY
  nn_security_info_t security_info;
#endif
};

struct ddsi_generic_proxy_endpoint {
  struct ddsi_entity_common e;
  struct ddsi_proxy_endpoint_common c;
};

struct ddsi_proxy_writer {
  struct ddsi_entity_common e;
  struct ddsi_proxy_endpoint_common c;
  ddsrt_avl_tree_t readers; /* matching LOCAL readers, see pwr_rd_match */
  int32_t n_reliable_readers; /* number of those that are reliable */
  int32_t n_readers_out_of_sync; /* number of those that require special handling (accepting historical data, waiting for historical data set to become complete) */
  seqno_t last_seq; /* highest known seq published by the writer, not last delivered */
  uint32_t last_fragnum; /* last known frag for last_seq, or UINT32_MAX if last_seq not partial */
  nn_count_t nackfragcount; /* last nackfrag seq number */
  ddsrt_atomic_uint32_t next_deliv_seq_lowword; /* lower 32-bits for next sequence number that will be delivered; for generating acks; 32-bit so atomic reads on all supported platforms */
  unsigned deliver_synchronously: 1; /* iff 1, delivery happens straight from receive thread for non-historical data; else through delivery queue "dqueue" */
  unsigned have_seen_heartbeat: 1; /* iff 1, we have received at least on heartbeat from this proxy writer */
  unsigned local_matching_inprogress: 1; /* iff 1, we are still busy matching local readers; this is so we don't deliver incoming data to some but not all readers initially */
  unsigned alive: 1; /* iff 1, the proxy writer is alive (lease for this proxy writer is not expired); field may be modified only when holding both pwr->e.lock and pwr->c.proxypp->e.lock */
  unsigned filtered: 1; /* iff 1, builtin proxy writer uses content filter, which affects heartbeats and gaps. */
  unsigned redundant_networking: 1; /* 1 iff requests receiving data on all advertised interfaces */
#ifdef DDS_HAS_SSM
  unsigned supports_ssm: 1; /* iff 1, this proxy writer supports SSM */
#endif
#ifdef DDS_HAS_SHM
  unsigned is_iceoryx : 1;
#endif
  uint32_t alive_vclock; /* virtual clock counting transitions between alive/not-alive */
  struct nn_defrag *defrag; /* defragmenter for this proxy writer; FIXME: perhaps shouldn't be for historical data */
  struct nn_reorder *reorder; /* message reordering for this proxy writer, out-of-sync readers can have their own, see pwr_rd_match */
  struct nn_dqueue *dqueue; /* delivery queue for asynchronous delivery (historical data is always delivered asynchronously) */
  struct xeventq *evq; /* timed event queue to be used for ACK generation */
  struct ddsi_local_reader_ary rdary; /* LOCAL readers for fast-pathing; if not fast-pathed, fall back to scanning local_readers */
  ddsi2direct_directread_cb_t ddsi2direct_cb;
  void *ddsi2direct_cbarg;
  struct lease *lease;
};


typedef int (*ddsi_filter_fn_t)(struct ddsi_writer *wr, struct ddsi_proxy_reader *prd, struct ddsi_serdata *serdata);

struct ddsi_proxy_reader {
  struct ddsi_entity_common e;
  struct ddsi_proxy_endpoint_common c;
  unsigned deleting: 1; /* set when being deleted */
  unsigned is_fict_trans_reader: 1; /* only true when it is certain that is a fictitious transient data reader (affects built-in topic generation) */
  unsigned requests_keyhash: 1; /* 1 iff this reader would like to receive keyhashes */
  unsigned redundant_networking: 1; /* 1 iff requests receiving data on all advertised interfaces */
#ifdef DDS_HAS_SSM
  unsigned favours_ssm: 1; /* iff 1, this proxy reader favours SSM when available */
#endif
#ifdef DDS_HAS_SHM
  unsigned is_iceoryx: 1;
#endif
  ddsrt_avl_tree_t writers; /* matching LOCAL writers */
  uint32_t receive_buffer_size; /* assumed receive buffer size inherited from proxypp */
  ddsi_filter_fn_t filter;
};

DDS_EXPORT extern const ddsrt_avl_treedef_t ddsi_pwr_readers_treedef;
DDS_EXPORT extern const ddsrt_avl_treedef_t ddsi_prd_writers_treedef;

void ddsi_proxy_writer_set_alive_may_unlock (struct ddsi_proxy_writer *pwr, bool notify);
int ddsi_proxy_writer_set_notalive (struct ddsi_proxy_writer *pwr, bool notify);
void ddsi_proxy_writer_get_alive_state (struct ddsi_proxy_writer *pwr, struct ddsi_alive_state *st);
struct ddsi_entity_common *ddsi_entity_common_from_proxy_endpoint_common (const struct ddsi_proxy_endpoint_common *c);
bool ddsi_is_proxy_endpoint (const struct ddsi_entity_common *e);


/* To create a new proxy writer or reader; the proxy participant is
   determined from the GUID and must exist. */
DDS_EXPORT int ddsi_new_proxy_writer (struct ddsi_domaingv *gv, const struct ddsi_guid *ppguid, const struct ddsi_guid *guid, struct addrset *as, const struct ddsi_plist *plist, struct nn_dqueue *dqueue, struct xeventq *evq, ddsrt_wctime_t timestamp, seqno_t seq);
DDS_EXPORT int ddsi_new_proxy_reader (struct ddsi_domaingv *gv, const struct ddsi_guid *ppguid, const struct ddsi_guid *guid, struct addrset *as, const struct ddsi_plist *plist, ddsrt_wctime_t timestamp, seqno_t seq
#ifdef DDS_HAS_SSM
, int favours_ssm
#endif
);

/* To delete a proxy writer or reader; these synchronously hide it
   from the outside world, preventing it from being matched to a
   reader or writer. Actual deletion is scheduled in the future, when
   no outstanding references may still exist (determined by checking
   thread progress, &c.). */
DDS_EXPORT int ddsi_delete_proxy_writer (struct ddsi_domaingv *gv, const struct ddsi_guid *guid, ddsrt_wctime_t timestamp, int isimplicit);
DDS_EXPORT int ddsi_delete_proxy_reader (struct ddsi_domaingv *gv, const struct ddsi_guid *guid, ddsrt_wctime_t timestamp, int isimplicit);

DDS_EXPORT void ddsi_update_proxy_reader (struct ddsi_proxy_reader *prd, seqno_t seq, struct addrset *as, const struct dds_qos *xqos, ddsrt_wctime_t timestamp);
DDS_EXPORT void ddsi_update_proxy_writer (struct ddsi_proxy_writer *pwr, seqno_t seq, struct addrset *as, const struct dds_qos *xqos, ddsrt_wctime_t timestamp);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_PROXY_ENDPOINT_H */
