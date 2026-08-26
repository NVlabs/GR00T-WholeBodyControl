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
#ifndef Q_RECEIVE_H
#define Q_RECEIVE_H

#if defined (__cplusplus)
extern "C" {
#endif

struct nn_rbufpool;
struct nn_rsample_info;
struct nn_rdata;
struct nn_rmsg;
struct ddsi_tran_listener;
struct recv_thread_arg;
struct ddsi_writer;
struct ddsi_proxy_reader;

struct nn_gap_info {
  seqno_t gapstart; // == 0 on init, indicating no gap recorded yet
  seqno_t gapend;   // >= gapstart
  uint32_t gapnumbits;
  uint32_t gapbits[256 / 32];
};

void nn_gap_info_init(struct nn_gap_info *gi);
void nn_gap_info_update(struct ddsi_domaingv *gv, struct nn_gap_info *gi, seqno_t seqnr);
struct nn_xmsg * nn_gap_info_create_gap(struct ddsi_writer *wr, struct ddsi_proxy_reader *prd, struct nn_gap_info *gi);

void trigger_recv_threads (const struct ddsi_domaingv *gv);
uint32_t recv_thread (void *vrecv_thread_arg);
uint32_t listen_thread (struct ddsi_tran_listener * listener);
int user_dqueue_handler (const struct nn_rsample_info *sampleinfo, const struct nn_rdata *fragchain, const ddsi_guid_t *rdguid, void *qarg);
int add_Gap (struct nn_xmsg *msg, struct ddsi_writer *wr, struct ddsi_proxy_reader *prd, seqno_t start, seqno_t base, uint32_t numbits, const uint32_t *bits);

DDS_EXPORT void ddsi_handle_rtps_message (struct thread_state * const thrst, struct ddsi_domaingv *gv, ddsi_tran_conn_t conn, const ddsi_guid_prefix_t *guidprefix, struct nn_rbufpool *rbpool, struct nn_rmsg *rmsg, size_t sz, unsigned char *msg, const ddsi_locator_t *srcloc);

#if defined (__cplusplus)
}
#endif

#endif /* Q_RECEIVE_H */
