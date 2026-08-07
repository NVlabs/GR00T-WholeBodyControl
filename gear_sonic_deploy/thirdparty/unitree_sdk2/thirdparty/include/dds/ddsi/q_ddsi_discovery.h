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
#ifndef NN_DDSI_DISCOVERY_H
#define NN_DDSI_DISCOVERY_H

#include "dds/ddsi/q_unused.h"
#include "dds/ddsi/ddsi_domaingv.h" // FIXME: MAX_XMIT_CONNS

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_participant;
struct ddsi_topic;
struct ddsi_writer;
struct ddsi_reader;
struct nn_rsample_info;
struct nn_rdata;
struct ddsi_plist;

struct participant_builtin_topic_data_locators {
  struct nn_locators_one def_uni[MAX_XMIT_CONNS], meta_uni[MAX_XMIT_CONNS];
  struct nn_locators_one def_multi, meta_multi;
};

void get_participant_builtin_topic_data (const struct ddsi_participant *pp, ddsi_plist_t *dst, struct participant_builtin_topic_data_locators *locs);

int spdp_write (struct ddsi_participant *pp);
int spdp_dispose_unregister (struct ddsi_participant *pp);

int sedp_write_topic (struct ddsi_topic *tp, bool alive);
int sedp_write_writer (struct ddsi_writer *wr);
int sedp_write_reader (struct ddsi_reader *rd);
int sedp_dispose_unregister_writer (struct ddsi_writer *wr);
int sedp_dispose_unregister_reader (struct ddsi_reader *rd);

int builtins_dqueue_handler (const struct nn_rsample_info *sampleinfo, const struct nn_rdata *fragchain, const ddsi_guid_t *rdguid, void *qarg);

#if defined (__cplusplus)
}
#endif

#endif /* NN_DDSI_DISCOVERY_H */
