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
#ifndef NN_XEVENT_H
#define NN_XEVENT_H

#include "dds/ddsrt/retcode.h"
#include "dds/ddsi/ddsi_guid.h"

#if defined (__cplusplus)
extern "C" {
#endif

/* NOTE: xevents scheduled with the same tsched used to always be
   executed in the order of scheduling, but that is no longer true.
   With the messages now via the "untimed" path, that should not
   introduce any issues. */

struct ddsi_writer;
struct ddsi_pwr_rd_match;
struct ddsi_participant;
struct ddsi_proxy_participant;
struct ddsi_tran_conn;
struct xevent;
struct xeventq;
struct ddsi_proxy_writer;
struct ddsi_proxy_reader;
struct ddsi_domaingv;
struct nn_xmsg;

DDS_EXPORT struct xeventq *xeventq_new (struct ddsi_domaingv *gv, size_t max_queued_rexmit_bytes, size_t max_queued_rexmit_msgs, uint32_t auxiliary_bandwidth_limit);

/* xeventq_free calls callback handlers with t = NEVER, at which point they are required to free
   whatever memory is claimed for the argument and call delete_xevent. */
DDS_EXPORT void xeventq_free (struct xeventq *evq);
DDS_EXPORT dds_return_t xeventq_start (struct xeventq *evq, const char *name); /* <0 => error, =0 => ok */
DDS_EXPORT void xeventq_stop (struct xeventq *evq);

DDS_EXPORT void qxev_msg (struct xeventq *evq, struct nn_xmsg *msg);

DDS_EXPORT void qxev_pwr_entityid (struct ddsi_proxy_writer * pwr, const ddsi_guid_t *guid);
DDS_EXPORT void qxev_prd_entityid (struct ddsi_proxy_reader * prd, const ddsi_guid_t *guid);
DDS_EXPORT void qxev_nt_callback (struct xeventq *evq, void (*cb) (void *arg), void *arg);

enum qxev_msg_rexmit_result {
  QXEV_MSG_REXMIT_DROPPED,
  QXEV_MSG_REXMIT_MERGED,
  QXEV_MSG_REXMIT_QUEUED
};

DDS_EXPORT enum qxev_msg_rexmit_result qxev_msg_rexmit_wrlock_held (struct xeventq *evq, struct nn_xmsg *msg, int force);

/* All of the following lock EVQ for the duration of the operation */
DDS_EXPORT void delete_xevent (struct xevent *ev);
DDS_EXPORT void delete_xevent_callback (struct xevent *ev);
DDS_EXPORT int resched_xevent_if_earlier (struct xevent *ev, ddsrt_mtime_t tsched);

DDS_EXPORT struct xevent *qxev_heartbeat (struct xeventq *evq, ddsrt_mtime_t tsched, const ddsi_guid_t *wr_guid);
DDS_EXPORT struct xevent *qxev_acknack (struct xeventq *evq, ddsrt_mtime_t tsched, const ddsi_guid_t *pwr_guid, const ddsi_guid_t *rd_guid);
DDS_EXPORT struct xevent *qxev_spdp (struct xeventq *evq, ddsrt_mtime_t tsched, const ddsi_guid_t *pp_guid, const ddsi_guid_t *proxypp_guid);
DDS_EXPORT struct xevent *qxev_pmd_update (struct xeventq *evq, ddsrt_mtime_t tsched, const ddsi_guid_t *pp_guid);
DDS_EXPORT struct xevent *qxev_delete_writer (struct xeventq *evq, ddsrt_mtime_t tsched, const ddsi_guid_t *guid);

/* cb will be called with now = NEVER if the event is still enqueued when when xeventq_free starts cleaning up */
DDS_EXPORT struct xevent *qxev_callback (struct xeventq *evq, ddsrt_mtime_t tsched, void (*cb) (struct xevent *xev, void *arg, ddsrt_mtime_t now), void *arg);

#if defined (__cplusplus)
}
#endif
#endif /* NN_XEVENT_H */
