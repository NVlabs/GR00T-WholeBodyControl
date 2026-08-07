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
#ifndef Q_GC_H
#define Q_GC_H

#include "dds/export.h"
#include "dds/ddsi/q_thread.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct gcreq;
struct gcreq_queue;
struct ddsi_domaingv;

struct ddsi_writer;
struct ddsi_reader;
struct ddsi_proxy_writer;
struct ddsi_proxy_reader;

typedef void (*gcreq_cb_t) (struct gcreq *gcreq);

struct idx_vtime {
  struct thread_state *thrst;
  vtime_t vtime;
};

struct gcreq {
  struct gcreq *next;
  struct gcreq_queue *queue;
  gcreq_cb_t cb;
  void *arg;
  uint32_t nvtimes;
  struct idx_vtime vtimes[];
};

DDS_EXPORT struct gcreq_queue *gcreq_queue_new (struct ddsi_domaingv *gv);
DDS_EXPORT void gcreq_queue_drain (struct gcreq_queue *q);
DDS_EXPORT void gcreq_queue_free (struct gcreq_queue *q);

DDS_EXPORT struct gcreq *gcreq_new (struct gcreq_queue *gcreq_queue, gcreq_cb_t cb);
DDS_EXPORT bool gcreq_queue_start (struct gcreq_queue *q);
DDS_EXPORT void gcreq_free (struct gcreq *gcreq);
DDS_EXPORT void gcreq_enqueue (struct gcreq *gcreq);
DDS_EXPORT int gcreq_requeue (struct gcreq *gcreq, gcreq_cb_t cb);

#if defined (__cplusplus)
}
#endif

#endif /* Q_GC_H */
