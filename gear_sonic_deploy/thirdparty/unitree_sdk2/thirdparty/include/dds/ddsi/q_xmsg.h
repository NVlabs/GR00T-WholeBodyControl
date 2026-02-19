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
#ifndef NN_XMSG_H
#define NN_XMSG_H

#include <stddef.h>

#include "dds/ddsrt/bswap.h"
#include "dds/ddsi/q_protocol.h" /* for, e.g., SubmessageKind_t */
//#include "dds/ddsi/ddsi_xqos.h" /* for, e.g., octetseq, stringseq */
#include "dds/ddsi/ddsi_tran.h"
#include "dds/features.h"

#ifdef DDS_HAS_SHM
#include "dds/ddsi/ddsi_shm_transport.h"
#endif

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_serdata;
struct addrset;
struct ddsi_proxy_reader;
struct ddsi_proxy_writer;
struct ddsi_writer;
struct ddsi_participant;

struct nn_adlink_participant_version_info;
struct nn_xmsgpool;
struct nn_xmsg_data;
struct nn_xmsg;
struct nn_xpack;
struct ddsi_plist_sample;

struct nn_xmsg_marker {
  size_t offset;
};

enum nn_xmsg_kind {
  NN_XMSG_KIND_CONTROL,
  NN_XMSG_KIND_DATA,
  NN_XMSG_KIND_DATA_REXMIT,
  NN_XMSG_KIND_DATA_REXMIT_NOMERGE
};

/* XMSGPOOL */

struct nn_xmsgpool *nn_xmsgpool_new (void);
void nn_xmsgpool_free (struct nn_xmsgpool *pool);

/* XMSG */

/* To allocate a new xmsg from the pool; if expected_size is NOT
   exceeded, no reallocs will be performed, else the address of the
   xmsg may change because of reallocing when appending to it. */
struct nn_xmsg *nn_xmsg_new (struct nn_xmsgpool *pool, const ddsi_guid_t *src_guid, struct ddsi_participant *pp, size_t expected_size, enum nn_xmsg_kind kind);

/* For sending to a particular destination (participant) */
void nn_xmsg_setdst1 (struct ddsi_domaingv *gv, struct nn_xmsg *m, const ddsi_guid_prefix_t *gp, const ddsi_xlocator_t *addr);
bool nn_xmsg_getdst1prefix (struct nn_xmsg *m, ddsi_guid_prefix_t *gp);

/* For sending to a particular proxy reader; this is a convenience
   routine that extracts a suitable address from the proxy reader's
   address sets and calls setdst1. */
void nn_xmsg_setdstPRD (struct nn_xmsg *m, const struct ddsi_proxy_reader *prd);
void nn_xmsg_setdstPWR (struct nn_xmsg *m, const struct ddsi_proxy_writer *pwr);

/* For sending to all in the address set AS -- typically, the writer's
   address set to multicast to all matched readers */
void nn_xmsg_setdstN (struct nn_xmsg *msg, struct addrset *as);

int nn_xmsg_setmaxdelay (struct nn_xmsg *msg, int64_t maxdelay);

#ifdef DDS_HAS_NETWORK_PARTITIONS
int nn_xmsg_setencoderid (struct nn_xmsg *msg, uint32_t encoderid);
#endif

/* Sets the location of the destination readerId within the message
   (address changes because of reallocations are handled correctly).
   M must be a rexmit, and for all rexmits this must be called.  It is
   a separate function because the location may only become known at a
   late-ish stage in the construction of the message. */
void nn_xmsg_set_data_readerId (struct nn_xmsg *m, ddsi_entityid_t *readerId);

/* If M and MADD are both xmsg's containing the same retransmit
   message, this will merge the destination embedded in MADD into M.
   Typically, this will cause the readerId of M to be cleared and the
   destination to change to the writer's address set.

   M and MADD *must* contain the same sample/fragment of a sample.

   Returns 1 if merge was successful, else 0.  On failure, neither
   message will have been changed and both should be sent as if there
   had been no merging. */
int nn_xmsg_merge_rexmit_destinations_wrlock_held (struct ddsi_domaingv *gv, struct nn_xmsg *m, const struct nn_xmsg *madd);

/* To set writer ids for updating last transmitted sequence number;
   wrfragid is 0 based, unlike DDSI but like other places where
   fragment numbers are handled internally. */
void nn_xmsg_setwriterseq (struct nn_xmsg *msg, const ddsi_guid_t *wrguid, seqno_t wrseq);
void nn_xmsg_setwriterseq_fragid (struct nn_xmsg *msg, const ddsi_guid_t *wrguid, seqno_t wrseq, nn_fragment_number_t wrfragid);

/* Comparison function for retransmits: orders messages on writer
   guid, sequence number and fragment id */
int nn_xmsg_compare_fragid (const struct nn_xmsg *a, const struct nn_xmsg *b);

void nn_xmsg_free (struct nn_xmsg *msg);
size_t nn_xmsg_size (const struct nn_xmsg *m);
void *nn_xmsg_payload (size_t *sz, struct nn_xmsg *m);
void nn_xmsg_payload_to_plistsample (struct ddsi_plist_sample *dst, nn_parameterid_t keyparam, const struct nn_xmsg *m);
enum nn_xmsg_kind nn_xmsg_kind (const struct nn_xmsg *m);
void nn_xmsg_guid_seq_fragid (const struct nn_xmsg *m, ddsi_guid_t *wrguid, seqno_t *wrseq, nn_fragment_number_t *wrfragid);

void *nn_xmsg_submsg_from_marker (struct nn_xmsg *msg, struct nn_xmsg_marker marker);
void *nn_xmsg_append (struct nn_xmsg *m, struct nn_xmsg_marker *marker, size_t sz);
void nn_xmsg_shrink (struct nn_xmsg *m, struct nn_xmsg_marker marker, size_t sz);
void nn_xmsg_serdata (struct nn_xmsg *m, struct ddsi_serdata *serdata, size_t off, size_t len, struct ddsi_writer *wr);
#ifdef DDS_HAS_SECURITY
size_t nn_xmsg_submsg_size (struct nn_xmsg *msg, struct nn_xmsg_marker marker);
void nn_xmsg_submsg_remove (struct nn_xmsg *msg, struct nn_xmsg_marker sm_marker);
void nn_xmsg_submsg_replace (struct nn_xmsg *msg, struct nn_xmsg_marker sm_marker, unsigned char *new_submsg, size_t new_len);
void nn_xmsg_submsg_append_refd_payload (struct nn_xmsg *msg, struct nn_xmsg_marker sm_marker);
#endif
void nn_xmsg_submsg_setnext (struct nn_xmsg *msg, struct nn_xmsg_marker marker);
void nn_xmsg_submsg_init (struct nn_xmsg *msg, struct nn_xmsg_marker marker, SubmessageKind_t smkind);
void nn_xmsg_add_timestamp (struct nn_xmsg *m, ddsrt_wctime_t t);
void nn_xmsg_add_entityid (struct nn_xmsg * m);
void *nn_xmsg_addpar_bo (struct nn_xmsg *m, nn_parameterid_t pid, size_t len, enum ddsrt_byte_order_selector bo);
void *nn_xmsg_addpar (struct nn_xmsg *m, nn_parameterid_t pid, size_t len);
void nn_xmsg_addpar_keyhash (struct nn_xmsg *m, const struct ddsi_serdata *serdata, bool force_md5);
void nn_xmsg_addpar_statusinfo (struct nn_xmsg *m, unsigned statusinfo);
void nn_xmsg_addpar_sentinel (struct nn_xmsg *m);
void nn_xmsg_addpar_sentinel_bo (struct nn_xmsg * m, enum ddsrt_byte_order_selector bo);
int nn_xmsg_addpar_sentinel_ifparam (struct nn_xmsg *m);

/* XPACK */

struct nn_xpack * nn_xpack_new (struct ddsi_domaingv *gv, uint32_t bw_limit, bool async_mode);
void nn_xpack_free (struct nn_xpack *xp);
void nn_xpack_send (struct nn_xpack *xp, bool immediately /* unused */);
int nn_xpack_addmsg (struct nn_xpack *xp, struct nn_xmsg *m, const uint32_t flags);
int64_t nn_xpack_maxdelay (const struct nn_xpack *xp);
unsigned nn_xpack_packetid (const struct nn_xpack *xp);

/* SENDQ */
void nn_xpack_sendq_init (struct ddsi_domaingv *gv);
void nn_xpack_sendq_start (struct ddsi_domaingv *gv);
void nn_xpack_sendq_stop (struct ddsi_domaingv *gv);
void nn_xpack_sendq_fini (struct ddsi_domaingv *gv);

#if defined (__cplusplus)
}
#endif
#endif /* NN_XMSG_H */
