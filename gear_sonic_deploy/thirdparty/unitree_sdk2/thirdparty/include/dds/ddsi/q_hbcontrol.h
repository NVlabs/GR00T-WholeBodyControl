/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef Q_HBCONTROL_H
#define Q_HBCONTROL_H

#include "dds/features.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_writer;
struct whc_state;
struct ddsi_proxy_reader;

struct hbcontrol {
  ddsrt_mtime_t t_of_last_write;
  ddsrt_mtime_t t_of_last_hb;
  ddsrt_mtime_t t_of_last_ackhb;
  ddsrt_mtime_t tsched;
  uint32_t hbs_since_last_write;
  uint32_t last_packetid;
};

void writer_hbcontrol_init (struct hbcontrol *hbc);
int64_t writer_hbcontrol_intv (const struct ddsi_writer *wr, const struct whc_state *whcst, ddsrt_mtime_t tnow);
void writer_hbcontrol_note_asyncwrite (struct ddsi_writer *wr, ddsrt_mtime_t tnow);
int writer_hbcontrol_ack_required (const struct ddsi_writer *wr, const struct whc_state *whcst, ddsrt_mtime_t tnow);
struct nn_xmsg *writer_hbcontrol_piggyback (struct ddsi_writer *wr, const struct whc_state *whcst, ddsrt_mtime_t tnow, uint32_t packetid, int *hbansreq);
int writer_hbcontrol_must_send (const struct ddsi_writer *wr, const struct whc_state *whcst, ddsrt_mtime_t tnow);
struct nn_xmsg *writer_hbcontrol_create_heartbeat (struct ddsi_writer *wr, const struct whc_state *whcst, ddsrt_mtime_t tnow, int hbansreq, int issync);

#ifdef DDS_HAS_SECURITY
struct nn_xmsg *writer_hbcontrol_p2p(struct ddsi_writer *wr, const struct whc_state *whcst, int hbansreq, struct ddsi_proxy_reader *prd);
#endif


#if defined (__cplusplus)
}
#endif

#endif /* Q_HBCONTROL_H */
