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
#ifndef Q_WHC_H
#define Q_WHC_H

#include <stddef.h>
#include "dds/ddsrt/time.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_serdata;
struct ddsi_plist;
struct ddsi_tkmap_instance;
struct whc_node; /* opaque, but currently used for deferred free lists */
struct whc;

struct whc_borrowed_sample {
  seqno_t seq;
  struct ddsi_serdata *serdata;
  struct ddsi_plist *plist;
  bool unacked;
  ddsrt_mtime_t last_rexmit_ts;
  unsigned rexmit_count;
};

struct whc_state {
  seqno_t min_seq; /* 0 if WHC empty, else > 0 */
  seqno_t max_seq; /* 0 if WHC empty, else >= min_seq */
  size_t unacked_bytes;
};
#define WHCST_ISEMPTY(whcst) ((whcst)->max_seq == 0)

/* Adjust SIZE and alignment stuff as needed: they are here simply so we can allocate
   an iter on the stack without specifying an implementation. If future changes or
   implementations require more, these can be adjusted.  An implementation should check
   things fit at compile time. */
#define WHC_SAMPLE_ITER_SIZE (8 * sizeof(void *))
struct whc_sample_iter_base {
  struct whc *whc;
};
struct whc_sample_iter {
  struct whc_sample_iter_base c;
  union {
    char opaque[WHC_SAMPLE_ITER_SIZE];
    /* cover alignment requirements: */
    uint64_t x;
    double y;
    void *p;
  } opaque;
};

typedef seqno_t (*whc_next_seq_t)(const struct whc *whc, seqno_t seq);
typedef void (*whc_get_state_t)(const struct whc *whc, struct whc_state *st);
typedef bool (*whc_borrow_sample_t)(const struct whc *whc, seqno_t seq, struct whc_borrowed_sample *sample);
typedef bool (*whc_borrow_sample_key_t)(const struct whc *whc, const struct ddsi_serdata *serdata_key, struct whc_borrowed_sample *sample);
typedef void (*whc_return_sample_t)(struct whc *whc, struct whc_borrowed_sample *sample, bool update_retransmit_info);
typedef void (*whc_sample_iter_init_t)(const struct whc *whc, struct whc_sample_iter *it);
typedef bool (*whc_sample_iter_borrow_next_t)(struct whc_sample_iter *it, struct whc_borrowed_sample *sample);
typedef void (*whc_free_t)(struct whc *whc);

/* min_seq is lowest sequence number that must be retained because of
   reliable readers that have not acknowledged all data */
/* max_drop_seq must go soon, it's way too ugly. */
/* plist may be NULL or ddsrt_malloc'd, WHC takes ownership of plist */
typedef int (*whc_insert_t)(struct whc *whc, seqno_t max_drop_seq, seqno_t seq, ddsrt_mtime_t exp, struct ddsi_plist *plist, struct ddsi_serdata *serdata, struct ddsi_tkmap_instance *tk);
typedef uint32_t (*whc_downgrade_to_volatile_t)(struct whc *whc, struct whc_state *st);
typedef uint32_t (*whc_remove_acked_messages_t)(struct whc *whc, seqno_t max_drop_seq, struct whc_state *whcst, struct whc_node **deferred_free_list);
typedef void (*whc_free_deferred_free_list_t)(struct whc *whc, struct whc_node *deferred_free_list);

struct whc_ops {
  whc_insert_t insert;
  whc_remove_acked_messages_t remove_acked_messages;
  whc_free_deferred_free_list_t free_deferred_free_list;
  whc_get_state_t get_state;
  whc_next_seq_t next_seq;
  whc_borrow_sample_t borrow_sample;
  whc_borrow_sample_key_t borrow_sample_key;
  whc_return_sample_t return_sample;
  whc_sample_iter_init_t sample_iter_init;
  whc_sample_iter_borrow_next_t sample_iter_borrow_next;
  whc_downgrade_to_volatile_t downgrade_to_volatile;
  whc_free_t free;
};

struct whc {
  const struct whc_ops *ops;
};

DDS_INLINE_EXPORT inline seqno_t whc_next_seq (const struct whc *whc, seqno_t seq) {
  return whc->ops->next_seq (whc, seq);
}
DDS_INLINE_EXPORT inline void whc_get_state (const struct whc *whc, struct whc_state *st) {
  whc->ops->get_state (whc, st);
}
DDS_INLINE_EXPORT inline bool whc_borrow_sample (const struct whc *whc, seqno_t seq, struct whc_borrowed_sample *sample) {
  return whc->ops->borrow_sample (whc, seq, sample);
}
DDS_INLINE_EXPORT inline bool whc_borrow_sample_key (const struct whc *whc, const struct ddsi_serdata *serdata_key, struct whc_borrowed_sample *sample) {
  return whc->ops->borrow_sample_key (whc, serdata_key, sample);
}
DDS_INLINE_EXPORT inline void whc_return_sample (struct whc *whc, struct whc_borrowed_sample *sample, bool update_retransmit_info) {
  whc->ops->return_sample (whc, sample, update_retransmit_info);
}
DDS_INLINE_EXPORT inline void whc_sample_iter_init (const struct whc *whc, struct whc_sample_iter *it) {
  whc->ops->sample_iter_init (whc, it);
}
DDS_INLINE_EXPORT inline bool whc_sample_iter_borrow_next (struct whc_sample_iter *it, struct whc_borrowed_sample *sample) {
  return it->c.whc->ops->sample_iter_borrow_next (it, sample);
}
DDS_INLINE_EXPORT inline void whc_free (struct whc *whc) {
  whc->ops->free (whc);
}
DDS_INLINE_EXPORT inline int whc_insert (struct whc *whc, seqno_t max_drop_seq, seqno_t seq, ddsrt_mtime_t exp, struct ddsi_plist *plist, struct ddsi_serdata *serdata, struct ddsi_tkmap_instance *tk) {
  return whc->ops->insert (whc, max_drop_seq, seq, exp, plist, serdata, tk);
}
DDS_INLINE_EXPORT inline unsigned whc_downgrade_to_volatile (struct whc *whc, struct whc_state *st) {
  return whc->ops->downgrade_to_volatile (whc, st);
}
DDS_INLINE_EXPORT inline unsigned whc_remove_acked_messages (struct whc *whc, seqno_t max_drop_seq, struct whc_state *whcst, struct whc_node **deferred_free_list) {
  return whc->ops->remove_acked_messages (whc, max_drop_seq, whcst, deferred_free_list);
}
DDS_INLINE_EXPORT inline void whc_free_deferred_free_list (struct whc *whc, struct whc_node *deferred_free_list) {
  whc->ops->free_deferred_free_list (whc, deferred_free_list);
}

#if defined (__cplusplus)
}
#endif

#endif /* Q_WHC_H */
