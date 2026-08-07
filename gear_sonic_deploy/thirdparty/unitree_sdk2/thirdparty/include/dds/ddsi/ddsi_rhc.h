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
#ifndef DDSI_RHC_H
#define DDSI_RHC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "dds/export.h"
#include "dds/features.h"

/* DDS_EXPORT inline i.c.w. __attributes__((visibility...)) and some compilers: */
#include "dds/ddsrt/attributes.h"
#include "dds/ddsrt/time.h"
#include "dds/ddsi/ddsi_guid.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct dds_qos;
struct ddsi_rhc;
struct ddsi_tkmap_instance;
struct ddsi_serdata;

struct ddsi_writer_info
{
  ddsi_guid_t guid;
  bool auto_dispose;
  int32_t ownership_strength;
  uint64_t iid;
#ifdef DDS_HAS_LIFESPAN
  ddsrt_mtime_t lifespan_exp;
#endif
};

typedef void (*ddsi_rhc_free_t) (struct ddsi_rhc *rhc);
typedef bool (*ddsi_rhc_store_t) (struct ddsi_rhc * __restrict rhc, const struct ddsi_writer_info * __restrict wrinfo, struct ddsi_serdata * __restrict sample, struct ddsi_tkmap_instance * __restrict tk);
typedef void (*ddsi_rhc_unregister_wr_t) (struct ddsi_rhc * __restrict rhc, const struct ddsi_writer_info * __restrict wrinfo);
typedef void (*ddsi_rhc_relinquish_ownership_t) (struct ddsi_rhc * __restrict rhc, const uint64_t wr_iid);
typedef void (*ddsi_rhc_set_qos_t) (struct ddsi_rhc *rhc, const struct dds_qos *qos);

struct ddsi_rhc_ops {
  ddsi_rhc_store_t store;
  ddsi_rhc_unregister_wr_t unregister_wr;
  ddsi_rhc_relinquish_ownership_t relinquish_ownership;
  ddsi_rhc_set_qos_t set_qos;
  ddsi_rhc_free_t free;
};

struct ddsi_rhc {
  const struct ddsi_rhc_ops *ops;
};

DDS_INLINE_EXPORT inline bool ddsi_rhc_store (struct ddsi_rhc * __restrict rhc, const struct ddsi_writer_info * __restrict wrinfo, struct ddsi_serdata * __restrict sample, struct ddsi_tkmap_instance * __restrict tk) {
  return rhc->ops->store (rhc, wrinfo, sample, tk);
}
DDS_INLINE_EXPORT inline void ddsi_rhc_unregister_wr (struct ddsi_rhc * __restrict rhc, const struct ddsi_writer_info * __restrict wrinfo) {
  rhc->ops->unregister_wr (rhc, wrinfo);
}
DDS_INLINE_EXPORT inline void ddsi_rhc_relinquish_ownership (struct ddsi_rhc * __restrict rhc, const uint64_t wr_iid) {
  rhc->ops->relinquish_ownership (rhc, wr_iid);
}
DDS_INLINE_EXPORT inline void ddsi_rhc_set_qos (struct ddsi_rhc *rhc, const struct dds_qos *qos) {
  rhc->ops->set_qos (rhc, qos);
}
DDS_INLINE_EXPORT inline void ddsi_rhc_free (struct ddsi_rhc *rhc) {
  rhc->ops->free (rhc);
}

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_RHC_H */
