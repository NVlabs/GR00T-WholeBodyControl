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
#ifndef NN_RADMIN_H
#define NN_RADMIN_H

#include <stddef.h>

#include "dds/ddsrt/time.h"
#include "dds/ddsrt/atomics.h"
#include "dds/ddsrt/threads.h"
#include "dds/ddsrt/align.h"
#include "dds/ddsrt/static_assert.h"
#include "dds/ddsi/ddsi_locator.h"
#include "dds/ddsi/q_rtps.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct nn_rbufpool;
struct nn_rbuf;
struct nn_rmsg;
struct nn_rdata;
struct nn_rsample;
struct nn_rsample_chain;
struct nn_rsample_info;
struct nn_defrag;
struct nn_reorder;
struct nn_dqueue;
struct ddsi_guid;
struct ddsi_tran_conn;

struct ddsi_proxy_writer;

struct nn_fragment_number_set;
struct nn_sequence_number_set;

typedef int (*nn_dqueue_handler_t) (const struct nn_rsample_info *sampleinfo, const struct nn_rdata *fragchain, const struct ddsi_guid *rdguid, void *qarg);

struct nn_rmsg_chunk {
  struct nn_rbuf *rbuf;
  struct nn_rmsg_chunk *next;

  /* Size is 0 after initial allocation, must be set with
     nn_rmsg_setsize after receiving a packet from the kernel and
     before processing it.  */
  union {
    uint32_t size;

    /* to ensure reasonable alignment of payload */
    int64_t l;
    double d;
    void *p;
  } u;

  /* unsigned char payload[] -- disallowed by C99 because of nesting */
};

struct nn_rmsg {
  /* Reference count: all references to rdatas of this message are
     counted. The rdatas themselves do not have a reference count.

     The refcount is biased by RMSG_REFCOUNT_UNCOMMITED_BIAS while
     still being inserted to allow verifying it is still uncommitted
     when allocating memory, increasing refcounts, &c.

     Each rdata adds RMS_REFCOUNT_RDATA_BIAS when it leaves
     defragmentation until it has been rejected by reordering or has
     been scheduled for delivery.  This allows delaying the
     decrementing of refcounts until after a sample has been added to
     all radmins even though be delivery of it may take place in
     concurrently. */
  ddsrt_atomic_uint32_t refcount;

  /* Worst-case memory requirement is gigantic (64kB UDP packet, only
     1-byte final fragments, each of one a new interval, or maybe 1
     byte messages, destined for many readers and in each case
     introducing a new interval, with receiver state changes in
     between, &c.), so we can either:

     - allocate a _lot_ and cover the worst case

     - allocate enough for all "reasonable" cases, discarding data when that limit is hit

     - dynamically add chunks of memory, and even entire receive buffers.

     The latter seems the best approach, especially because it also
     covers the second one.  We treat the other chunks specially,
     which is not strictly required but also not entirely
     unreasonable, considering that the first chunk has the refcount &
     the real packet. */
  struct nn_rmsg_chunk *lastchunk;

  /* whether to log */
  bool trace;

  struct nn_rmsg_chunk chunk;
};
DDSRT_STATIC_ASSERT (sizeof (struct nn_rmsg) == offsetof (struct nn_rmsg, chunk) + sizeof (struct nn_rmsg_chunk));
#define NN_RMSG_PAYLOAD(m) ((unsigned char *) (m + 1))
#define NN_RMSG_PAYLOADOFF(m, o) (NN_RMSG_PAYLOAD (m) + (o))

/* Allocated inside a chunk of memory by a custom allocator and requires >= 8-byte alignment */
#define ALIGNOF_RMSG (dds_alignof(struct nn_rmsg) > 8 ? dds_alignof(struct nn_rmsg) : 8)

struct receiver_state {
  ddsi_guid_prefix_t src_guid_prefix;     /* 12 */
  ddsi_guid_prefix_t dst_guid_prefix;     /* 12 */
  struct addrset *reply_locators;         /* 4/8 */
  uint32_t forme:1;                       /* 4 */
  uint32_t rtps_encoded:1;                /* - */
  nn_vendorid_t vendor;                   /* 2 */
  nn_protocol_version_t protocol_version; /* 2 => 44/48 */
  struct ddsi_tran_conn *conn;            /* Connection for request */
  ddsi_locator_t srcloc;
  struct ddsi_domaingv *gv;
};

struct nn_rsample_info {
  seqno_t seq;
  struct receiver_state *rst;
  struct ddsi_proxy_writer *pwr;
  uint32_t size;
  uint32_t fragsize;
  ddsrt_wctime_t timestamp;
  ddsrt_wctime_t reception_timestamp; /* OpenSplice extension -- but we get it essentially for free, so why not? */
  unsigned statusinfo: 2;       /* just the two defined bits from the status info */
  unsigned bswap: 1;            /* so we can extract well formatted writer info quicker */
  unsigned complex_qos: 1;      /* includes QoS other than keyhash, 2-bit statusinfo, PT writer info */
};

struct nn_rdata {
  struct nn_rmsg *rmsg;         /* received (and refcounted) in rmsg */
  struct nn_rdata *nextfrag;    /* fragment chain */
  uint32_t min, maxp1;          /* fragment as byte offsets */
  uint16_t submsg_zoff;         /* offset to submessage from packet start, or 0 */
  uint16_t payload_zoff;        /* offset to payload from packet start */
  uint16_t keyhash_zoff;        /* offset to keyhash from packet start, or 0 */
#ifndef NDEBUG
  ddsrt_atomic_uint32_t refcount_bias_added;
#endif
};

/* All relative offsets in packets that we care about (submessage
   header, payload, writer info) are at multiples of 4 bytes and
   within 64kB, so technically we can make do with 14 bits instead of
   16, in case we run out of space.

   If we _really_ need to squeeze out every last bit, only the submsg
   offset really requires 14 bits, the for the others we could use an
   offset relative to the submessage header so that it is limited by
   the maximum size of the inline QoS ...  Defining the macros now, so
   we have the option to do wild things. */
#ifndef NDEBUG
#define NN_ZOFF_TO_OFF(zoff) ((unsigned) (zoff))
#define NN_OFF_TO_ZOFF(off) (assert ((off) < 65536), ((unsigned short) (off)))
#else
#define NN_ZOFF_TO_OFF(zoff) ((unsigned) (zoff))
#define NN_OFF_TO_ZOFF(off) ((unsigned short) (off))
#endif
#define NN_RDATA_PAYLOAD_OFF(rdata) NN_ZOFF_TO_OFF ((rdata)->payload_zoff)
#define NN_RDATA_SUBMSG_OFF(rdata) NN_ZOFF_TO_OFF ((rdata)->submsg_zoff)
#define NN_RDATA_KEYHASH_OFF(rdata) NN_ZOFF_TO_OFF ((rdata)->keyhash_zoff)

struct nn_rsample_chain_elem {
  /* FIXME: evidently smaller than a defrag_iv, but maybe better to
     merge it with defrag_iv in a union anyway. */
  struct nn_rdata *fragchain;
  struct nn_rsample_chain_elem *next;
  /* Gaps have sampleinfo = NULL, but nonetheless a fragchain with 1
     rdata with min=maxp1 (length 0) and valid rmsg pointer.  And (see
     DQUEUE) its lsb gets abused so we can queue "bubbles" in addition
     to data). */
  struct nn_rsample_info *sampleinfo;
};

struct nn_rsample_chain {
  struct nn_rsample_chain_elem *first;
  struct nn_rsample_chain_elem *last;
};

enum nn_reorder_mode {
  NN_REORDER_MODE_NORMAL,
  NN_REORDER_MODE_MONOTONICALLY_INCREASING,
  NN_REORDER_MODE_ALWAYS_DELIVER
};

enum nn_defrag_drop_mode {
  NN_DEFRAG_DROP_OLDEST,        /* (believed to be) best for unreliable */
  NN_DEFRAG_DROP_LATEST         /* (...) best for reliable  */
};

typedef int32_t nn_reorder_result_t;
/* typedef of reorder result serves as a warning that it is to be
   interpreted as follows: */
/* REORDER_DELIVER > 0 -- number of samples in sample chain */
#define NN_REORDER_ACCEPT        0 /* accepted/stored (for gap: also adjusted next_expected) */
#define NN_REORDER_TOO_OLD      -1 /* discarded because it was too old */
#define NN_REORDER_REJECT       -2 /* caller may reuse memory ("real" reject for data, "fake" for gap) */

typedef void (*nn_dqueue_callback_t) (void *arg);

struct ddsrt_log_cfg;
struct nn_fragment_number_set_header;
struct nn_sequence_number_set_header;

DDS_EXPORT struct nn_rbufpool *nn_rbufpool_new (const struct ddsrt_log_cfg *logcfg, uint32_t rbuf_size, uint32_t max_rmsg_size);
DDS_EXPORT void nn_rbufpool_setowner (struct nn_rbufpool *rbp, ddsrt_thread_t tid);
DDS_EXPORT void nn_rbufpool_free (struct nn_rbufpool *rbp);

DDS_EXPORT struct nn_rmsg *nn_rmsg_new (struct nn_rbufpool *rbufpool);
DDS_EXPORT void nn_rmsg_setsize (struct nn_rmsg *rmsg, uint32_t size);
DDS_EXPORT void nn_rmsg_commit (struct nn_rmsg *rmsg);
DDS_EXPORT void nn_rmsg_free (struct nn_rmsg *rmsg);
DDS_EXPORT void *nn_rmsg_alloc (struct nn_rmsg *rmsg, uint32_t size);

DDS_EXPORT struct nn_rdata *nn_rdata_new (struct nn_rmsg *rmsg, uint32_t start, uint32_t endp1, uint32_t submsg_offset, uint32_t payload_offset, uint32_t keyhash_offset);
DDS_EXPORT struct nn_rdata *nn_rdata_newgap (struct nn_rmsg *rmsg);
DDS_EXPORT void nn_fragchain_adjust_refcount (struct nn_rdata *frag, int adjust);
DDS_EXPORT void nn_fragchain_unref (struct nn_rdata *frag);

DDS_EXPORT struct nn_defrag *nn_defrag_new (const struct ddsrt_log_cfg *logcfg, enum nn_defrag_drop_mode drop_mode, uint32_t max_samples);
DDS_EXPORT void nn_defrag_free (struct nn_defrag *defrag);
DDS_EXPORT struct nn_rsample *nn_defrag_rsample (struct nn_defrag *defrag, struct nn_rdata *rdata, const struct nn_rsample_info *sampleinfo);
void nn_defrag_notegap (struct nn_defrag *defrag, seqno_t min, seqno_t maxp1);

enum nn_defrag_nackmap_result {
  DEFRAG_NACKMAP_UNKNOWN_SAMPLE,
  DEFRAG_NACKMAP_ALL_ADVERTISED_FRAGMENTS_KNOWN,
  DEFRAG_NACKMAP_FRAGMENTS_MISSING
};

enum nn_defrag_nackmap_result nn_defrag_nackmap (struct nn_defrag *defrag, seqno_t seq, uint32_t maxfragnum, struct nn_fragment_number_set_header *map, uint32_t *mapbits, uint32_t maxsz);

void nn_defrag_prune (struct nn_defrag *defrag, ddsi_guid_prefix_t *dst, seqno_t min);

DDS_EXPORT struct nn_reorder *nn_reorder_new (const struct ddsrt_log_cfg *logcfg, enum nn_reorder_mode mode, uint32_t max_samples, bool late_ack_mode);
DDS_EXPORT void nn_reorder_free (struct nn_reorder *r);
struct nn_rsample *nn_reorder_rsample_dup_first (struct nn_rmsg *rmsg, struct nn_rsample *rsampleiv);
DDS_EXPORT struct nn_rdata *nn_rsample_fragchain (struct nn_rsample *rsample);
DDS_EXPORT nn_reorder_result_t nn_reorder_rsample (struct nn_rsample_chain *sc, struct nn_reorder *reorder, struct nn_rsample *rsampleiv, int *refcount_adjust, int delivery_queue_full_p);
DDS_EXPORT nn_reorder_result_t nn_reorder_gap (struct nn_rsample_chain *sc, struct nn_reorder *reorder, struct nn_rdata *rdata, seqno_t min, seqno_t maxp1, int *refcount_adjust);
void nn_reorder_drop_upto (struct nn_reorder *reorder, seqno_t maxp1); // drops [1,maxp1); next_seq' = maxp1
DDS_EXPORT int nn_reorder_wantsample (const struct nn_reorder *reorder, seqno_t seq);
unsigned nn_reorder_nackmap (const struct nn_reorder *reorder, seqno_t base, seqno_t maxseq, struct nn_sequence_number_set_header *map, uint32_t *mapbits, uint32_t maxsz, int notail);
DDS_EXPORT seqno_t nn_reorder_next_seq (const struct nn_reorder *reorder);
void nn_reorder_set_next_seq (struct nn_reorder *reorder, seqno_t seq);

struct nn_dqueue *nn_dqueue_new (const char *name, const struct ddsi_domaingv *gv, uint32_t max_samples, nn_dqueue_handler_t handler, void *arg);
bool nn_dqueue_start (struct nn_dqueue *q);
void nn_dqueue_free (struct nn_dqueue *q);
bool nn_dqueue_enqueue_deferred_wakeup (struct nn_dqueue *q, struct nn_rsample_chain *sc, nn_reorder_result_t rres);
void dd_dqueue_enqueue_trigger (struct nn_dqueue *q);
void nn_dqueue_enqueue (struct nn_dqueue *q, struct nn_rsample_chain *sc, nn_reorder_result_t rres);
void nn_dqueue_enqueue1 (struct nn_dqueue *q, const ddsi_guid_t *rdguid, struct nn_rsample_chain *sc, nn_reorder_result_t rres);
void nn_dqueue_enqueue_callback (struct nn_dqueue *q, nn_dqueue_callback_t cb, void *arg);
int  nn_dqueue_is_full (struct nn_dqueue *q);
void nn_dqueue_wait_until_empty_if_full (struct nn_dqueue *q);

DDS_EXPORT void nn_defrag_stats (struct nn_defrag *defrag, uint64_t *discarded_bytes);
DDS_EXPORT void nn_reorder_stats (struct nn_reorder *reorder, uint64_t *discarded_bytes);

#if defined (__cplusplus)
}
#endif

#endif /* NN_RADMIN_H */
