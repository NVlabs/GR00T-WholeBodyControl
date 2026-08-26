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
#ifndef DDSI_PROXY_PARTICIPANT_H
#define DDSI_PROXY_PARTICIPANT_H

#include "dds/export.h"
#include "dds/features.h"

#include "dds/ddsrt/atomics.h"
#include "dds/ddsrt/avl.h"
#include "dds/ddsrt/fibheap.h"
#include "dds/ddsi/ddsi_domaingv.h"
#include "dds/ddsi/ddsi_entity.h"
#include "dds/ddsi/ddsi_topic.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct lease;
struct ddsi_plist;
struct addrset;
struct ddsi_proxy_endpoint_common;

struct ddsi_proxy_participant
{
  struct ddsi_entity_common e;
  uint32_t refc; /* number of proxy endpoints (both user & built-in; not groups, they don't have a life of their own) */
  nn_vendorid_t vendor; /* vendor code from discovery */
  unsigned bes; /* built-in endpoint set */
  ddsi_guid_t privileged_pp_guid; /* if this PP depends on another PP for its SEDP writing */
  struct ddsi_plist *plist; /* settings/QoS for this participant */
  ddsrt_atomic_voidp_t minl_auto; /* clone of min(leaseheap_auto) */
  ddsrt_fibheap_t leaseheap_auto; /* keeps leases for this proxypp and leases for pwrs (with liveliness automatic) */
  ddsrt_atomic_voidp_t minl_man; /* clone of min(leaseheap_man) */
  ddsrt_fibheap_t leaseheap_man; /* keeps leases for this proxypp and leases for pwrs (with liveliness manual-by-participant) */
  struct lease *lease; /* lease for this proxypp */
  struct addrset *as_default; /* default address set to use for user data traffic */
  struct addrset *as_meta; /* default address set to use for discovery traffic */
  struct ddsi_proxy_endpoint_common *endpoints; /* all proxy endpoints can be reached from here */
#ifdef DDS_HAS_TOPIC_DISCOVERY
  ddsrt_avl_tree_t topics;
#endif
  seqno_t seq; /* sequence number of most recent SPDP message */
  uint32_t receive_buffer_size; /* assumed size of receive buffer, used to limit bursts involving this proxypp */
  unsigned implicitly_created : 1; /* participants are implicitly created for Cloud/Fog discovered endpoints */
  unsigned is_ddsi2_pp: 1; /* if this is the federation-leader on the remote node */
  unsigned minimal_bes_mode: 1;
  unsigned lease_expired: 1;
  unsigned deleting: 1;
  unsigned proxypp_have_spdp: 1;
  unsigned owns_lease: 1;
  unsigned redundant_networking: 1; /* 1 iff requests receiving data on all advertised interfaces */
#ifdef DDS_HAS_SECURITY
  nn_security_info_t security_info;
  struct ddsi_proxy_participant_sec_attributes *sec_attr;
#endif
};

#ifdef DDS_HAS_TOPIC_DISCOVERY
extern const ddsrt_avl_treedef_t ddsi_proxypp_proxytp_treedef;
#endif

/* Set when this proxy participant is created implicitly and has to be deleted upon disappearance
   of its last endpoint.  FIXME: Currently there is a potential race with adding a new endpoint
   in parallel to deleting the last remaining one. The endpoint will then be created, added to the
   proxy participant and then both are deleted. With the current single-threaded discovery
   this can only happen when it is all triggered by lease expiry. */
#define CF_IMPLICITLY_CREATED_PROXYPP          (1 << 0)
/* Set when this proxy participant is a DDSI2 participant, to help Cloud figure out whom to send
   discovery data when used in conjunction with the networking bridge */
#define CF_PARTICIPANT_IS_DDSI2                (1 << 1)
/* Set when this proxy participant is not to be announced on the built-in topics yet */
#define CF_PROXYPP_NO_SPDP                     (1 << 2)

int ddsi_update_proxy_participant_plist_locked (struct ddsi_proxy_participant *proxypp, seqno_t seq, const struct ddsi_plist *datap, ddsrt_wctime_t timestamp);
void ddsi_proxy_participant_reassign_lease (struct ddsi_proxy_participant *proxypp, struct lease *newlease);
void ddsi_purge_proxy_participants (struct ddsi_domaingv *gv, const ddsi_xlocator_t *loc, bool delete_from_as_disc);
int ddsi_ref_proxy_participant (struct ddsi_proxy_participant *proxypp, struct ddsi_proxy_endpoint_common *c);
void ddsi_unref_proxy_participant (struct ddsi_proxy_participant *proxypp, struct ddsi_proxy_endpoint_common *c);
void ddsi_proxy_participant_add_pwr_lease_locked (struct ddsi_proxy_participant * proxypp, const struct ddsi_proxy_writer * pwr);
void ddsi_proxy_participant_remove_pwr_lease_locked (struct ddsi_proxy_participant * proxypp, struct ddsi_proxy_writer * pwr);

/* To create or delete a new proxy participant: "guid" MUST have the
   pre-defined participant entity id. Unlike ddsi_delete_participant (),
   deleting a proxy participant will automatically delete all its
   readers & writers. Delete removes the participant from a hash table
   and schedules the actual deletion.

   TODO: what about proxy participants without built-in endpoints?
*/

DDS_EXPORT bool ddsi_new_proxy_participant (struct ddsi_domaingv *gv, const struct ddsi_guid *guid, uint32_t bes, const struct ddsi_guid *privileged_pp_guid, struct addrset *as_default, struct addrset *as_meta, const struct ddsi_plist *plist, dds_duration_t tlease_dur, nn_vendorid_t vendor, unsigned custom_flags, ddsrt_wctime_t timestamp, seqno_t seq);
DDS_EXPORT int ddsi_delete_proxy_participant_by_guid (struct ddsi_domaingv *gv, const struct ddsi_guid *guid, ddsrt_wctime_t timestamp, int isimplicit);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_PROXY_PARTICIPANT_H */
