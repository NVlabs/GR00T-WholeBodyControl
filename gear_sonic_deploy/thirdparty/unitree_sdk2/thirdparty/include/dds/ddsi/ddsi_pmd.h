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
#ifndef DDSI_PMD_H
#define DDSI_PMD_H

#include <stdint.h>
#include "dds/ddsrt/time.h"
#include "dds/ddsi/ddsi_serdata.h"
#include "dds/ddsi/ddsi_plist_generic.h"
#include "dds/ddsi/ddsi_guid.h"
#include "dds/ddsi/ddsi_xqos.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_domaingv;
struct thread_state;
struct ddsi_guid;
struct nn_xpack;
struct ddsi_participant;
struct receiver_state;

typedef struct ParticipantMessageData {
  ddsi_guid_prefix_t participantGuidPrefix;
  uint32_t kind; /* really 4 octets */
  ddsi_octetseq_t value;
} ParticipantMessageData_t;

extern const enum pserop participant_message_data_ops[];
extern size_t participant_message_data_nops;
extern const enum pserop participant_message_data_ops_key[];
extern size_t participant_message_data_nops_key;

void write_pmd_message_guid (struct ddsi_domaingv * const gv, struct ddsi_guid *pp_guid, unsigned pmd_kind);
void write_pmd_message (struct thread_state * const ts1, struct nn_xpack *xp, struct ddsi_participant *pp, unsigned pmd_kind);
void handle_pmd_message (const struct receiver_state *rst, struct ddsi_serdata *sample_common);

#if defined (__cplusplus)
}
#endif
#endif /* DDSI_PMD_H */
