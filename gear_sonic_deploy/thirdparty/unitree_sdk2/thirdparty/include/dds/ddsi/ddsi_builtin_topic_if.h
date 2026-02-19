/*
 * Copyright(c) 2019 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef _DDSI_BUILTIN_TOPIC_IF_H_
#define _DDSI_BUILTIN_TOPIC_IF_H_

#include "dds/ddsi/ddsi_vendor.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_entity_common;
struct ddsi_tkmap_instance;
struct ddsi_sertype;
struct ddsi_guid;
struct ddsi_topic_definition;

struct ddsi_builtin_topic_interface {
  void *arg;

  bool (*builtintopic_is_builtintopic) (const struct ddsi_sertype *type, void *arg);
  bool (*builtintopic_is_visible) (const struct ddsi_guid *guid, nn_vendorid_t vendorid, void *arg);
  struct ddsi_tkmap_instance * (*builtintopic_get_tkmap_entry) (const struct ddsi_guid *guid, void *arg);
  void (*builtintopic_write_endpoint) (const struct ddsi_entity_common *e, ddsrt_wctime_t timestamp, bool alive, void *arg);
  void (*builtintopic_write_topic) (const struct ddsi_topic_definition *tpd, ddsrt_wctime_t timestamp, bool alive, void *arg);
};

DDS_INLINE_EXPORT inline bool builtintopic_is_visible (const struct ddsi_builtin_topic_interface *btif, const struct ddsi_guid *guid, nn_vendorid_t vendorid) {
  return btif ? btif->builtintopic_is_visible (guid, vendorid, btif->arg) : false;
}
DDS_INLINE_EXPORT inline bool builtintopic_is_builtintopic (const struct ddsi_builtin_topic_interface *btif, const struct ddsi_sertype *type) {
  return btif ? btif->builtintopic_is_builtintopic (type, btif->arg) : false;
}
DDS_INLINE_EXPORT inline struct ddsi_tkmap_instance *builtintopic_get_tkmap_entry (const struct ddsi_builtin_topic_interface *btif, const struct ddsi_guid *guid) {
  return btif ? btif->builtintopic_get_tkmap_entry (guid, btif->arg) : NULL;
}
DDS_INLINE_EXPORT inline void builtintopic_write_endpoint (const struct ddsi_builtin_topic_interface *btif, const struct ddsi_entity_common *e, ddsrt_wctime_t timestamp, bool alive) {
  if (btif) btif->builtintopic_write_endpoint (e, timestamp, alive, btif->arg);
}
DDS_INLINE_EXPORT inline void builtintopic_write_topic (const struct ddsi_builtin_topic_interface *btif, const struct ddsi_topic_definition *tpd, ddsrt_wctime_t timestamp, bool alive) {
  if (btif) btif->builtintopic_write_topic (tpd, timestamp, alive, btif->arg);
}

#if defined (__cplusplus)
}
#endif

#endif
