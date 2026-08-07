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
#ifndef NN_MISC_H
#define NN_MISC_H

#include "dds/ddsi/q_protocol.h"

#if defined (__cplusplus)
extern "C" {
#endif

DDS_INLINE_EXPORT inline seqno_t fromSN (const nn_sequence_number_t sn) {
  uint64_t sn_high = (uint32_t) sn.high;
  return (sn_high << 32) | sn.low;
}

DDS_INLINE_EXPORT inline bool validating_fromSN (const nn_sequence_number_t sn, seqno_t *res) {
  // fromSN does not checks whatsoever (and shouldn't because it is used quite a lot)
  // Valid sequence numbers are in [1 .. 2**63-1] union { SEQUENCE_NUMBER_UNKNOWN }
  // where SEQUENCE_NUMBER_UNKNOWN is the usual abomination: ((2**32-1) << 32)
  //
  // As far as I can tell, there are no messages where SEQUENCE_NUMBER_UNKNOWN is actually a
  // valid input (it would perhaps have been an elegant way to mark pre-emptive HEARTBEATs and
  // ACKNACKs, but convention is different).  So we reject them as invalid until we are forced
  // to do differently.  That leaves [1 .. 2**63-1]
  //
  // Since we use uint64_t, we can easily test by checking whether (s-1) is in [0 .. 2**63-1)
  const seqno_t tmp = fromSN (sn);
  *res = tmp;
  return (tmp - 1) < MAX_SEQ_NUMBER;
}

DDS_INLINE_EXPORT inline nn_sequence_number_t toSN (seqno_t n) {
  nn_sequence_number_t x;
  x.high = (int32_t) (n >> 32);
  x.low = (uint32_t) n;
  return x;
}

unsigned char normalize_data_datafrag_flags (const SubmessageHeader_t *smhdr);

#ifdef DDS_HAS_NETWORK_PARTITIONS
int WildcardOverlap(char * p1, char * p2);
#endif

extern const ddsi_guid_t nullguid;
DDS_EXPORT bool guid_prefix_zero (const ddsi_guid_prefix_t *a);
DDS_EXPORT int guid_prefix_eq (const ddsi_guid_prefix_t *a, const ddsi_guid_prefix_t *b);
DDS_EXPORT int guid_eq (const struct ddsi_guid *a, const struct ddsi_guid *b);
DDS_EXPORT int ddsi2_patmatch (const char *pat, const char *str);

#ifdef DDS_HAS_NETWORK_PARTITIONS
struct ddsi_config;
struct ddsi_config_partitionmapping_listelem;
struct ddsi_config_partitionmapping_listelem *find_partitionmapping (const struct ddsi_config *cfg, const char *partition, const char *topic);
int is_ignored_partition (const struct ddsi_config *cfg, const char *partition, const char *topic);
#endif
#ifdef DDS_HAS_NETWORK_CHANNELS
struct ddsi_config;
struct ddsi_config_channel_listelem;
struct ddsi_config_channel_listelem *find_channel (const struct config *cfg, nn_transport_priority_qospolicy_t transport_priority);
#endif

#if defined (__cplusplus)
}
#endif

#endif /* NN_MISC_H */
