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
#ifndef DDSI_TYPELIB_H
#define DDSI_TYPELIB_H

#include "dds/features.h"

#include <stdbool.h>
#include <stdint.h>
#include "dds/ddsrt/static_assert.h"
#include "dds/ddsrt/avl.h"
#include "dds/ddsi/ddsi_guid.h"
#include "dds/ddsi/ddsi_list_tmpl.h"
#include "dds/ddsi/ddsi_typewrap.h"
#include "dds/ddsi/ddsi_sertype.h"

#if defined (__cplusplus)
extern "C" {
#endif

#ifdef DDS_HAS_TYPE_DISCOVERY

extern const ddsrt_avl_treedef_t ddsi_typelib_treedef;
extern const ddsrt_avl_treedef_t ddsi_typedeps_treedef;
extern const ddsrt_avl_treedef_t ddsi_typedeps_reverse_treedef;

struct ddsi_generic_proxy_endpoint;
struct ddsi_domaingv;
struct ddsi_sertype;
struct ddsi_sertype_cdr_data;
struct ddsi_type;
struct ddsi_type_pair;

enum ddsi_type_state {
  DDSI_TYPE_UNRESOLVED,
  DDSI_TYPE_REQUESTED,
  DDSI_TYPE_PARTIAL_RESOLVED,
  DDSI_TYPE_RESOLVED,
  DDSI_TYPE_INVALID,
};

typedef enum ddsi_type_request {
  DDSI_TYPE_NO_REQUEST,
  DDSI_TYPE_SEND_REQUEST
} ddsi_type_request_t;

/* Used for converting type-id to strings */
struct ddsi_typeid_str {
  char str[50];
};

DDS_EXPORT char *ddsi_make_typeid_str (struct ddsi_typeid_str *buf, const ddsi_typeid_t *type_id);

DDS_EXPORT bool ddsi_typeinfo_equal (const ddsi_typeinfo_t *a, const ddsi_typeinfo_t *b, ddsi_type_include_deps_t deps);
DDS_EXPORT ddsi_typeid_t *ddsi_typeinfo_typeid (const ddsi_typeinfo_t *type_info, ddsi_typeid_kind_t kind);
DDS_EXPORT ddsi_typeinfo_t *ddsi_typeinfo_deser (const struct ddsi_sertype_cdr_data *ser);
DDS_EXPORT void ddsi_typeinfo_fini (ddsi_typeinfo_t *typeinfo);
DDS_EXPORT ddsi_typeinfo_t * ddsi_typeinfo_dup (const ddsi_typeinfo_t *src);
DDS_EXPORT const ddsi_typeid_t *ddsi_typeinfo_minimal_typeid (const ddsi_typeinfo_t *typeinfo);
DDS_EXPORT const ddsi_typeid_t *ddsi_typeinfo_complete_typeid (const ddsi_typeinfo_t *typeinfo);
DDS_EXPORT bool ddsi_typeinfo_present (const ddsi_typeinfo_t *typeinfo);
DDS_EXPORT bool ddsi_typeinfo_valid (const ddsi_typeinfo_t *typeinfo);

DDS_EXPORT ddsi_typemap_t *ddsi_typemap_deser (const struct ddsi_sertype_cdr_data *ser);
DDS_EXPORT void ddsi_typemap_fini (ddsi_typemap_t *typemap);

DDS_EXPORT void ddsi_type_register_dep (struct ddsi_domaingv *gv, const ddsi_typeid_t *src_type_id, struct ddsi_type **dst_dep_type, const struct DDS_XTypes_TypeIdentifier *dep_type_id);
DDS_EXPORT void ddsi_type_ref_locked (struct ddsi_domaingv *gv, struct ddsi_type **type, const struct ddsi_type *src);
DDS_EXPORT dds_return_t ddsi_type_ref_id_locked (struct ddsi_domaingv *gv, struct ddsi_type **type, const ddsi_typeid_t *type_id);
DDS_EXPORT dds_return_t ddsi_type_ref_local (struct ddsi_domaingv *gv, struct ddsi_type **type, const struct ddsi_sertype *sertype, ddsi_typeid_kind_t kind);
DDS_EXPORT dds_return_t ddsi_type_ref_proxy (struct ddsi_domaingv *gv, struct ddsi_type **type, const ddsi_typeinfo_t *type_info, ddsi_typeid_kind_t kind, const ddsi_guid_t *proxy_guid);
DDS_EXPORT dds_return_t ddsi_type_add_typeobj (struct ddsi_domaingv *gv, struct ddsi_type *type, const struct DDS_XTypes_TypeObject *type_obj);
DDS_EXPORT struct ddsi_typeobj *ddsi_type_get_typeobj (struct ddsi_domaingv *gv, const struct ddsi_type *type);
DDS_EXPORT dds_return_t ddsi_type_get_typeinfo_ser (struct ddsi_domaingv *gv, const struct ddsi_type *type, unsigned char **data, uint32_t *sz);
DDS_EXPORT dds_return_t ddsi_type_get_typeinfo (struct ddsi_domaingv *gv, const struct ddsi_type *type, struct ddsi_typeinfo *type_info);
DDS_EXPORT dds_return_t ddsi_type_get_typemap_ser (struct ddsi_domaingv *gv, const struct ddsi_type *type, unsigned char **data, uint32_t *sz);
DDS_EXPORT void ddsi_type_unreg_proxy (struct ddsi_domaingv *gv, struct ddsi_type *type, const ddsi_guid_t *proxy_guid);
DDS_EXPORT void ddsi_type_unref (struct ddsi_domaingv *gv, struct ddsi_type *type);
DDS_EXPORT void ddsi_type_unref_sertype (struct ddsi_domaingv *gv, const struct ddsi_sertype *sertype);
DDS_EXPORT void ddsi_type_unref_locked (struct ddsi_domaingv *gv, struct ddsi_type *type);
DDS_EXPORT bool ddsi_type_resolved_locked (struct ddsi_domaingv *gv, const struct ddsi_type *type, ddsi_type_include_deps_t resolved_kind);
DDS_EXPORT bool ddsi_type_resolved (struct ddsi_domaingv *gv, const struct ddsi_type *type, ddsi_type_include_deps_t resolved_kind);

/**
 * @brief Waits for the provided type to be resolved
 *
 * In case the type is succesfully resolved (or was already resolved), this
 * function increases the refcount for this type. Caller should do the unref.
 */
DDS_EXPORT dds_return_t ddsi_wait_for_type_resolved (struct ddsi_domaingv *gv, const ddsi_typeid_t *type_id, dds_duration_t timeout, struct ddsi_type **type, ddsi_type_include_deps_t resolved_kind, ddsi_type_request_t request);

DDS_EXPORT bool ddsi_is_assignable_from (struct ddsi_domaingv *gv, const struct ddsi_type_pair *rd_type_pair, uint32_t rd_resolved, const struct ddsi_type_pair *wr_type_pair, uint32_t wr_resolved, const dds_type_consistency_enforcement_qospolicy_t *tce);
DDS_EXPORT const ddsi_typeid_t *ddsi_type_pair_minimal_id (const struct ddsi_type_pair *type_pair);
DDS_EXPORT const ddsi_typeid_t *ddsi_type_pair_complete_id (const struct ddsi_type_pair *type_pair);
DDS_EXPORT ddsi_typeinfo_t *ddsi_type_pair_minimal_info (struct ddsi_domaingv *gv, const struct ddsi_type_pair *type_pair);
DDS_EXPORT ddsi_typeinfo_t *ddsi_type_pair_complete_info (struct ddsi_domaingv *gv, const struct ddsi_type_pair *type_pair);
DDS_EXPORT struct ddsi_type_pair *ddsi_type_pair_init (const ddsi_typeid_t *type_id_minimal, const ddsi_typeid_t *type_id_complete);
DDS_EXPORT void ddsi_type_pair_free (struct ddsi_type_pair *type_pair);

/**
 * Returns the type lookup meta object for the provided type identifier.
 * The caller of this functions needs to have locked gv->typelib_lock
 *
 * @remark The returned object from this function is not refcounted,
 *   its lifetime is at lease the lifetime of the (proxy) endpoints
 *   that are referring to it.
 */
DDS_EXPORT struct ddsi_type * ddsi_type_lookup_locked (struct ddsi_domaingv *gv, const ddsi_typeid_t *type_id);
/* Same as above, but does not require to have the lock */
DDS_EXPORT struct ddsi_type * ddsi_type_lookup (struct ddsi_domaingv *gv, const ddsi_typeid_t *type_id);

/**
 * Gets a list of proxy endpoints that are registered for the provided type
 * or for types that (indirectly) depend on this type. The resulting set of
 * endpoints is stored in the gpe_match_upd parameter. The parameter n_match_upd
 * should contain the actual number of entries in gpe_match_upd and will
 * be updated if new entries are added.
 */
DDS_EXPORT void ddsi_type_get_gpe_matches (struct ddsi_domaingv *gv, const struct ddsi_type *type, struct ddsi_generic_proxy_endpoint ***gpe_match_upd, uint32_t *n_match_upd);

/**
 * Compares the provided type lookup meta objects.
 */
DDS_EXPORT int ddsi_type_compare (const struct ddsi_type *a, const struct ddsi_type *b);

#endif /* DDS_HAS_TYPE_DISCOVERY */

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_TYPELIB_H */
