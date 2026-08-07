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
#ifndef NN_RTPS_H
#define NN_RTPS_H

#include "dds/export.h"
#include "dds/ddsi/ddsi_vendor.h"
#include "dds/ddsi/ddsi_guid.h"

#if defined (__cplusplus)
extern "C" {
#endif

typedef struct {
  uint8_t major, minor;
} nn_protocol_version_t;
typedef uint64_t seqno_t;
#define MAX_SEQ_NUMBER INT64_MAX

#define PGUIDPREFIX(gp) (gp).u[0], (gp).u[1], (gp).u[2]
#define PGUID(g) PGUIDPREFIX ((g).prefix), (g).entityid.u
#define PGUIDPREFIXFMT "%" PRIx32 ":%" PRIx32 ":%" PRIx32
#define PGUIDFMT PGUIDPREFIXFMT ":%" PRIx32

/* predefined entity ids; here viewed as an unsigned int, on the
   network as four bytes corresponding to the integer in network byte
   order */
#define NN_ENTITYID_UNKNOWN 0x0
#define NN_ENTITYID_PARTICIPANT 0x1c1
#define NN_ENTITYID_SEDP_BUILTIN_TOPIC_WRITER 0x2c2
#define NN_ENTITYID_SEDP_BUILTIN_TOPIC_READER 0x2c7
#define NN_ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER 0x3c2
#define NN_ENTITYID_SEDP_BUILTIN_PUBLICATIONS_READER 0x3c7
#define NN_ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER 0x4c2
#define NN_ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_READER 0x4c7
#define NN_ENTITYID_SPDP_BUILTIN_PARTICIPANT_WRITER 0x100c2
#define NN_ENTITYID_SPDP_BUILTIN_PARTICIPANT_READER 0x100c7
#define NN_ENTITYID_P2P_BUILTIN_PARTICIPANT_MESSAGE_WRITER 0x200c2
#define NN_ENTITYID_P2P_BUILTIN_PARTICIPANT_MESSAGE_READER 0x200c7

#define NN_ENTITYID_TL_SVC_BUILTIN_REQUEST_WRITER 0x300c3
#define NN_ENTITYID_TL_SVC_BUILTIN_REQUEST_READER 0x300c4
#define NN_ENTITYID_TL_SVC_BUILTIN_REPLY_WRITER 0x301c3
#define NN_ENTITYID_TL_SVC_BUILTIN_REPLY_READER 0x301c4

#define NN_ENTITYID_SEDP_BUILTIN_PUBLICATIONS_SECURE_WRITER 0xff0003c2
#define NN_ENTITYID_SEDP_BUILTIN_PUBLICATIONS_SECURE_READER 0xff0003c7
#define NN_ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_SECURE_WRITER 0xff0004c2
#define NN_ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_SECURE_READER 0xff0004c7
#define NN_ENTITYID_P2P_BUILTIN_PARTICIPANT_STATELESS_MESSAGE_WRITER 0x201c3
#define NN_ENTITYID_P2P_BUILTIN_PARTICIPANT_STATELESS_MESSAGE_READER 0x201c4
#define NN_ENTITYID_P2P_BUILTIN_PARTICIPANT_MESSAGE_SECURE_WRITER 0xff0200c2
#define NN_ENTITYID_P2P_BUILTIN_PARTICIPANT_MESSAGE_SECURE_READER 0xff0200c7
#define NN_ENTITYID_P2P_BUILTIN_PARTICIPANT_VOLATILE_SECURE_WRITER 0xff0202c3
#define NN_ENTITYID_P2P_BUILTIN_PARTICIPANT_VOLATILE_SECURE_READER 0xff0202c4
#define NN_ENTITYID_SPDP_RELIABLE_BUILTIN_PARTICIPANT_SECURE_WRITER 0xff0101c2
#define NN_ENTITYID_SPDP_RELIABLE_BUILTIN_PARTICIPANT_SECURE_READER 0xff0101c7

#define NN_ENTITYID_SOURCE_MASK 0xc0
#define NN_ENTITYID_SOURCE_USER 0x00
#define NN_ENTITYID_SOURCE_BUILTIN 0xc0
#define NN_ENTITYID_SOURCE_VENDOR 0x40
#define NN_ENTITYID_KIND_MASK 0x3f
#define NN_ENTITYID_KIND_WRITER_WITH_KEY 0x02
#define NN_ENTITYID_KIND_WRITER_NO_KEY 0x03
#define NN_ENTITYID_KIND_READER_NO_KEY 0x04
#define NN_ENTITYID_KIND_READER_WITH_KEY 0x07
/* Entity kind topic is not defined in the RTPS spec, so the following two
   should to be used as vendor specific entities using NN_ENTITYID_SOURCE_VENDOR.
   Two entity kinds for built-in and user topics are required, because the
   vendor and built-in flags cannot be combined. */
#define NN_ENTITYID_KIND_CYCLONE_TOPIC_BUILTIN 0x0c
#define NN_ENTITYID_KIND_CYCLONE_TOPIC_USER 0x0d

#define NN_ENTITYID_ALLOCSTEP 0x100

struct cfgst;
struct ddsi_domaingv;
DDS_EXPORT int rtps_config_prep (struct ddsi_domaingv *gv, struct cfgst *cfgst);
int rtps_config_open_trace (struct ddsi_domaingv *gv);
DDS_EXPORT int rtps_init (struct ddsi_domaingv *gv);
int rtps_start (struct ddsi_domaingv *gv);
void rtps_stop (struct ddsi_domaingv *gv);
DDS_EXPORT void rtps_fini (struct ddsi_domaingv *gv);

DDS_EXPORT void ddsi_set_deafmute (struct ddsi_domaingv *gv, bool deaf, bool mute, int64_t reset_after);

#if defined (__cplusplus)
}
#endif

#endif /* NN_RTPS_H */
