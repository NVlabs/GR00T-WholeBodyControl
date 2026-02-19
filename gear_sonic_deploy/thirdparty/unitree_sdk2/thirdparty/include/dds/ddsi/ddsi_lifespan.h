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
#ifndef DDSI_LIFESPAN_H
#define DDSI_LIFESPAN_H

#include "dds/ddsrt/fibheap.h"
#include "dds/ddsrt/time.h"
#include "dds/ddsi/ddsi_domaingv.h"

#if defined (__cplusplus)
extern "C" {
#endif

typedef ddsrt_mtime_t (*sample_expired_cb_t)(void *hc, ddsrt_mtime_t tnow);

struct lifespan_adm {
  ddsrt_fibheap_t ls_exp_heap;              /* heap for sample expiration (lifespan) */
  struct xevent *evt;                       /* xevent that triggers for sample with earliest expiration */
  sample_expired_cb_t sample_expired_cb;    /* callback for expired sample; this cb can use lifespan_next_expired_locked to get next expired sample */
  size_t fh_offset;                         /* offset of lifespan_adm element in whc or rhc */
  size_t fhn_offset;                        /* offset of lifespan_fhnode element in whc or rhc node (sample) */
};

struct lifespan_fhnode {
  ddsrt_fibheap_node_t heapnode;
  ddsrt_mtime_t t_expire;
};

DDS_EXPORT void lifespan_init (const struct ddsi_domaingv *gv, struct lifespan_adm *lifespan_adm, size_t fh_offset, size_t fh_node_offset, sample_expired_cb_t sample_expired_cb);
DDS_EXPORT void lifespan_fini (const struct lifespan_adm *lifespan_adm);
DDS_EXPORT ddsrt_mtime_t lifespan_next_expired_locked (const struct lifespan_adm *lifespan_adm, ddsrt_mtime_t tnow, void **sample);
DDS_EXPORT void lifespan_register_sample_real (struct lifespan_adm *lifespan_adm, struct lifespan_fhnode *node);
DDS_EXPORT void lifespan_unregister_sample_real (struct lifespan_adm *lifespan_adm, struct lifespan_fhnode *node);

DDS_INLINE_EXPORT inline void lifespan_register_sample_locked (struct lifespan_adm *lifespan_adm, struct lifespan_fhnode *node)
{
  if (node->t_expire.v != DDS_NEVER)
    lifespan_register_sample_real (lifespan_adm, node);
}

DDS_INLINE_EXPORT inline void lifespan_unregister_sample_locked (struct lifespan_adm *lifespan_adm, struct lifespan_fhnode *node)
{
  if (node->t_expire.v != DDS_NEVER)
    lifespan_unregister_sample_real (lifespan_adm, node);
}

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_LIFESPAN_H */

