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
#ifndef _DDS_RHC_H_
#define _DDS_RHC_H_

#include "dds/ddsrt/static_assert.h"
#include "dds/ddsi/ddsi_rhc.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS /** DOC_TODO we should document the rhc */

#define NO_STATE_MASK_SET   (DDS_ANY_STATE + 1)

#if defined (__cplusplus)
extern "C" {
#endif

struct dds_rhc;
struct dds_readcond;
struct dds_reader;
struct ddsi_tkmap;

typedef dds_return_t (*dds_rhc_associate_t) (struct dds_rhc *rhc, struct dds_reader *reader, const struct ddsi_sertype *type, struct ddsi_tkmap *tkmap);
typedef int32_t (*dds_rhc_read_take_t) (struct dds_rhc *rhc, bool lock, void **values, dds_sample_info_t *info_seq, uint32_t max_samples, uint32_t mask, dds_instance_handle_t handle, struct dds_readcond *cond);
typedef int32_t (*dds_rhc_read_take_cdr_t) (struct dds_rhc *rhc, bool lock, struct ddsi_serdata **values, dds_sample_info_t *info_seq, uint32_t max_samples, uint32_t sample_states, uint32_t view_states, uint32_t instance_states, dds_instance_handle_t handle);

typedef bool (*dds_rhc_add_readcondition_t) (struct dds_rhc *rhc, struct dds_readcond *cond);
typedef void (*dds_rhc_remove_readcondition_t) (struct dds_rhc *rhc, struct dds_readcond *cond);

typedef uint32_t (*dds_rhc_lock_samples_t) (struct dds_rhc *rhc);

struct dds_rhc_ops {
  /* A copy of DDSI rhc ops comes first so we can use either interface without
     additional indirections */
  struct ddsi_rhc_ops rhc_ops;
  dds_rhc_read_take_t read;
  dds_rhc_read_take_t take;
  dds_rhc_read_take_cdr_t readcdr;
  dds_rhc_read_take_cdr_t takecdr;
  dds_rhc_add_readcondition_t add_readcondition;
  dds_rhc_remove_readcondition_t remove_readcondition;
  dds_rhc_lock_samples_t lock_samples;
  dds_rhc_associate_t associate;
};

struct dds_rhc {
  union {
    const struct dds_rhc_ops *ops;
    struct ddsi_rhc rhc;
  } common;
};

DDSRT_STATIC_ASSERT (offsetof (struct dds_rhc, common.ops) == offsetof (struct ddsi_rhc, ops));

DDS_INLINE_EXPORT inline dds_return_t dds_rhc_associate (struct dds_rhc *rhc, struct dds_reader *reader, const struct ddsi_sertype *type, struct ddsi_tkmap *tkmap) {
  return rhc->common.ops->associate (rhc, reader, type, tkmap);
}
DDS_INLINE_EXPORT inline bool dds_rhc_store (struct dds_rhc * __restrict rhc, const struct ddsi_writer_info * __restrict wrinfo, struct ddsi_serdata * __restrict sample, struct ddsi_tkmap_instance * __restrict tk) {
  return rhc->common.ops->rhc_ops.store (&rhc->common.rhc, wrinfo, sample, tk);
}
DDS_INLINE_EXPORT inline void dds_rhc_unregister_wr (struct dds_rhc * __restrict rhc, const struct ddsi_writer_info * __restrict wrinfo) {
  rhc->common.ops->rhc_ops.unregister_wr (&rhc->common.rhc, wrinfo);
}
DDS_INLINE_EXPORT inline void dds_rhc_relinquish_ownership (struct dds_rhc * __restrict rhc, const uint64_t wr_iid) {
  rhc->common.ops->rhc_ops.relinquish_ownership (&rhc->common.rhc, wr_iid);
}
DDS_INLINE_EXPORT inline void dds_rhc_set_qos (struct dds_rhc *rhc, const struct dds_qos *qos) {
  rhc->common.ops->rhc_ops.set_qos (&rhc->common.rhc, qos);
}
DDS_INLINE_EXPORT inline void dds_rhc_free (struct dds_rhc *rhc) {
  rhc->common.ops->rhc_ops.free (&rhc->common.rhc);
}
DDS_INLINE_EXPORT inline int32_t dds_rhc_read (struct dds_rhc *rhc, bool lock, void **values, dds_sample_info_t *info_seq, uint32_t max_samples, uint32_t mask, dds_instance_handle_t handle, struct dds_readcond *cond) {
  return (rhc->common.ops->read) (rhc, lock, values, info_seq, max_samples, mask, handle, cond);
}
DDS_INLINE_EXPORT inline int32_t dds_rhc_take (struct dds_rhc *rhc, bool lock, void **values, dds_sample_info_t *info_seq, uint32_t max_samples, uint32_t mask, dds_instance_handle_t handle, struct dds_readcond *cond) {
  return rhc->common.ops->take (rhc, lock, values, info_seq, max_samples, mask, handle, cond);
}
DDS_INLINE_EXPORT inline int32_t dds_rhc_readcdr (struct dds_rhc *rhc, bool lock, struct ddsi_serdata **values, dds_sample_info_t *info_seq, uint32_t max_samples, uint32_t sample_states, uint32_t view_states, uint32_t instance_states, dds_instance_handle_t handle) {
  return rhc->common.ops->readcdr (rhc, lock, values, info_seq, max_samples, sample_states, view_states, instance_states, handle);
}
DDS_INLINE_EXPORT inline int32_t dds_rhc_takecdr (struct dds_rhc *rhc, bool lock, struct ddsi_serdata **values, dds_sample_info_t *info_seq, uint32_t max_samples, uint32_t sample_states, uint32_t view_states, uint32_t instance_states, dds_instance_handle_t handle) {
  return rhc->common.ops->takecdr (rhc, lock, values, info_seq, max_samples, sample_states, view_states, instance_states, handle);
}
DDS_INLINE_EXPORT inline bool dds_rhc_add_readcondition (struct dds_rhc *rhc, struct dds_readcond *cond) {
  return rhc->common.ops->add_readcondition (rhc, cond);
}
DDS_INLINE_EXPORT inline void dds_rhc_remove_readcondition (struct dds_rhc *rhc, struct dds_readcond *cond) {
  rhc->common.ops->remove_readcondition (rhc, cond);
}
DDS_INLINE_EXPORT inline uint32_t dds_rhc_lock_samples (struct dds_rhc *rhc) {
  return rhc->common.ops->lock_samples (rhc);
}

DDS_EXPORT void dds_reader_data_available_cb (struct dds_reader *rd);

#if defined (__cplusplus)
}
#endif
#endif
#endif
