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
#ifndef DDSI_TYPELOOKUP_H
#define DDSI_TYPELOOKUP_H

#include "dds/features.h"

#ifdef DDS_HAS_TYPE_DISCOVERY

#include <stdint.h>
#include "dds/ddsrt/time.h"
#include "dds/ddsrt/hopscotch.h"
#include "dds/ddsrt/mh3.h"
#include "dds/ddsi/ddsi_serdata.h"
#include "dds/ddsi/ddsi_sertype.h"
#include "dds/ddsi/ddsi_plist_generic.h"
#include "dds/ddsi/ddsi_xqos.h"
#include "dds/ddsi/ddsi_xt_typeinfo.h"
#include "dds/ddsi/ddsi_xt_typelookup.h"
#include "dds/ddsi/ddsi_typelib.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_guid;
struct ddsi_domaingv;
struct thread_state;
struct nn_xpack;
struct ddsi_participant;
struct receiver_state;
struct ddsi_serdata;
struct ddsi_sertype;
struct ddsi_type;

/**
 * Send a type lookup request message in order to request type information for the
 * provided type identifier.
 */
DDS_EXPORT bool ddsi_tl_request_type (struct ddsi_domaingv * const gv, const ddsi_typeid_t *type_id, const ddsi_guid_t *proxypp_guid, ddsi_type_include_deps_t deps);

/**
 * Handle an incoming type lookup request message. For all types requested
 * that are known in this node, the serialized sertype is send in a type
 * lookup reply message. In case none of the requested types is known,
 * an empty reply message will be sent.
 */
DDS_EXPORT void ddsi_tl_handle_request (struct ddsi_domaingv *gv, struct ddsi_serdata *sample_common);

/**
 * Add type information from a type lookup reply to the type library.
 */
DDS_EXPORT void ddsi_tl_add_types (struct ddsi_domaingv *gv, const DDS_Builtin_TypeLookup_Reply *reply, struct ddsi_generic_proxy_endpoint ***gpe_match_upd, uint32_t *n_match_upd);

/**
 * Handle an incoming type lookup reply message. The sertypes from this
 * reply are registered in the local type administation and referenced
 * from the corresponding proxy endpoints.
 */
DDS_EXPORT void ddsi_tl_handle_reply (struct ddsi_domaingv *gv, struct ddsi_serdata *sample_common);

#if defined (__cplusplus)
}
#endif
#endif /* DDS_HAS_TYPE_DISCOVERY */
#endif /* DDSI_TYPELOOKUP_H */
