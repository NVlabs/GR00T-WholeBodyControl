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
#ifndef DDSI_XT_IMPL_H
#define DDSI_XT_IMPL_H

#include "dds/features.h"

#include <stdbool.h>
#include <stdint.h>
#include "dds/ddsrt/avl.h"
#include "dds/ddsrt/misc.h"
#include "dds/ddsi/ddsi_xt_typeinfo.h"
#include "dds/ddsi/ddsi_xt_typemap.h"

#if defined (__cplusplus)
extern "C" {
#endif

#define PTYPEIDFMT "[%s %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x]"
#define PHASH(x, n) ((x)._d == DDS_XTypes_EK_MINIMAL || (x)._d == DDS_XTypes_EK_COMPLETE ? (x)._u.equivalence_hash[(n)] : 0)
#define PTYPEID(x) (ddsi_typekind_descr((x)._d)), PHASH((x), 0), PHASH(x, 1), PHASH((x), 2), PHASH((x), 3), PHASH((x), 4), PHASH((x), 5), PHASH((x), 6), PHASH((x), 7), PHASH((x), 8), PHASH((x), 9), PHASH((x), 10), PHASH((x), 11), PHASH((x), 12), PHASH((x), 13)

#define NOARG
DDSI_LIST_TYPES_TMPL(ddsi_type_proxy_guid_list, ddsi_guid_t, NOARG, 32)
#undef NOARG

struct ddsi_typeid {
  struct DDS_XTypes_TypeIdentifier x;
};

struct ddsi_typeobj {
  struct DDS_XTypes_TypeObject x;
};

struct ddsi_typeinfo {
  struct DDS_XTypes_TypeInformation x;
};

struct ddsi_typemap {
  struct DDS_XTypes_TypeMapping x;
};

struct xt_applied_type_annotations {
  struct DDS_XTypes_AppliedBuiltinTypeAnnotations *ann_builtin;
  struct DDS_XTypes_AppliedAnnotationSeq *ann_custom;
};

struct xt_applied_member_annotations {
  struct DDS_XTypes_AppliedBuiltinMemberAnnotations *ann_builtin;
  struct DDS_XTypes_AppliedAnnotationSeq *ann_custom;
};

struct xt_type_detail {
  DDS_XTypes_QualifiedTypeName type_name;
  struct xt_applied_type_annotations annotations;
};

struct xt_member_detail {
  DDS_XTypes_MemberName name;
  DDS_XTypes_NameHash name_hash;
  struct xt_applied_member_annotations annotations;
};

struct xt_string {
  DDS_XTypes_LBound bound;
};

struct xt_collection_common {
  DDS_XTypes_CollectionTypeFlag flags;
  DDS_XTypes_EquivalenceKind ek;
  struct xt_type_detail detail;
  struct ddsi_type *element_type;
  DDS_XTypes_CollectionElementFlag element_flags;
  struct xt_applied_member_annotations element_annotations;
};

struct xt_seq {
  struct xt_collection_common c;
  DDS_XTypes_LBound bound;
};

struct xt_array {
  struct xt_collection_common c;
  struct DDS_XTypes_LBoundSeq bounds;
};

struct xt_map {
  struct xt_collection_common c;
  DDS_XTypes_LBound bound;
  DDS_XTypes_CollectionElementFlag key_flags;
  struct ddsi_type *key_type;
  struct xt_applied_member_annotations key_annotations;
};

struct xt_alias {
  struct ddsi_type *related_type;
  DDS_XTypes_AliasTypeFlag flags;
  DDS_XTypes_AliasMemberFlag related_flags;
  struct xt_type_detail detail;
};

struct xt_annotation_parameter {
  struct ddsi_type *member_type;
  DDS_XTypes_AnnotationParameterFlag flags;
  DDS_XTypes_MemberName name;
  DDS_XTypes_NameHash name_hash;
  struct DDS_XTypes_AnnotationParameterValue default_value;
};
struct xt_annotation_parameter_seq {
  uint32_t length;
  struct xt_annotation_parameter *seq;
};
struct xt_annotation {
  DDS_XTypes_QualifiedTypeName annotation_name;
  struct xt_annotation_parameter_seq *members;
};

struct xt_struct_member {
  DDS_XTypes_MemberId id;
  DDS_XTypes_StructMemberFlag flags;
  struct ddsi_type *type;
  struct xt_member_detail detail;
};
struct xt_struct_member_seq {
  uint32_t length;
  struct xt_struct_member *seq;
};
struct xt_struct {
  DDS_XTypes_StructTypeFlag flags;
  struct ddsi_type *base_type;
  struct xt_struct_member_seq members;
  struct xt_type_detail detail;
};

struct xt_union_member {
  DDS_XTypes_MemberId id;
  DDS_XTypes_UnionMemberFlag flags;
  struct ddsi_type *type;
  struct DDS_XTypes_UnionCaseLabelSeq label_seq;
  struct xt_member_detail detail;
};
struct xt_union_member_seq {
  uint32_t length;
  struct xt_union_member *seq;
};
struct xt_union {
  DDS_XTypes_UnionTypeFlag flags;
  struct ddsi_type *disc_type;
  DDS_XTypes_UnionDiscriminatorFlag disc_flags;
  struct xt_applied_type_annotations disc_annotations;
  struct xt_union_member_seq members;
  struct xt_type_detail detail;
};

struct xt_bitfield {
  uint16_t position;
  DDS_XTypes_BitsetMemberFlag flags;
  uint8_t bitcount;
  DDS_XTypes_TypeKind holder_type; // Must be primitive integer type
  struct xt_member_detail detail;
};
struct xt_bitfield_seq {
  uint32_t length;
  struct xt_bitfield *seq;
};
struct xt_bitset {
  DDS_XTypes_BitsetTypeFlag flags;
  struct xt_bitfield_seq fields;
  struct xt_type_detail detail;
};

struct xt_enum_literal {
  int32_t value;
  DDS_XTypes_EnumeratedLiteralFlag flags;
  struct xt_member_detail detail;
};
struct xt_enum_literal_seq {
  uint32_t length;
  struct xt_enum_literal *seq;
};
struct xt_enum {
  DDS_XTypes_EnumTypeFlag flags;
  DDS_XTypes_BitBound bit_bound;
  struct xt_enum_literal_seq literals;
  struct xt_type_detail detail;
};

struct xt_bitflag {
  uint16_t position;
  DDS_XTypes_BitflagFlag flags;
  struct xt_member_detail detail;
};
struct xt_bitflag_seq {
  uint32_t length;
  struct xt_bitflag *seq;
};
struct xt_bitmask {
  DDS_XTypes_BitmaskTypeFlag flags;
  DDS_XTypes_BitBound bit_bound;
  struct xt_bitflag_seq bitflags;
  struct xt_type_detail detail;
};

struct xt_type
{
  ddsi_typeid_t id;
  ddsi_typeid_kind_t kind;
  struct DDS_XTypes_StronglyConnectedComponentId sc_component_id;

  uint8_t _d;
  union {
    // case TK_NONE:
    // case TK_BOOLEAN:
    // case TK_BYTE:
    // case TK_INT16:
    // case TK_INT32:
    // case TK_INT64:
    // case TK_UINT8:
    // case TK_UINT16:
    // case TK_UINT32:
    // case TK_UINT64:
    // case TK_FLOAT32:
    // case TK_FLOAT64:
    // case TK_FLOAT128:
    // case TK_CHAR8:
    // case TK_CHAR16:
    //   <empty for primitive types>
    // case TK_STRING8:
    struct xt_string str8;
    // case TK_STRING16:
    struct xt_string str16;
    // case TK_SEQUENCE:
    struct xt_seq seq;
    // case TK_ARRAY:
    struct xt_array array;
    // case TK_MAP:
    struct xt_map map;
    // case TK_ALIAS:
    struct xt_alias alias;
    // case TK_ANNOTATION:
    struct xt_annotation annotation;
    // case TK_STRUCTURE:
    struct xt_struct structure;
    // case TK_UNION:
    struct xt_union union_type;
    // case TK_BITSET:
    struct xt_bitset bitset;
    // case TK_ENUM:
    struct xt_enum enum_type;
    // case TK_BITMASK:
    struct xt_bitmask bitmask;
  } _u;
};

/* Type identifier must at offset 0, see comment for ddsi_type */
DDSRT_STATIC_ASSERT (offsetof (struct xt_type, id) == 0);

struct ddsi_type_dep {
  ddsrt_avl_node_t src_avl_node;
  ddsrt_avl_node_t dep_avl_node;
  ddsi_typeid_t src_type_id;    // type that has the dependency on dep_type_id
  ddsi_typeid_t dep_type_id;    // dependent type, a direct or indirect dependency of src_type_id
  bool from_type_info;          // entry was added based on a dependent type in the type-info, requires unref of the dependent type on deletion
};

struct ddsi_type {
  struct xt_type xt;                            /* wrapper for XTypes type id/obj */
  ddsrt_avl_node_t avl_node;
  enum ddsi_type_state state;
  seqno_t request_seqno;                        /* sequence number of the last type lookup request message */
  struct ddsi_type_proxy_guid_list proxy_guids; /* administration for proxy endpoints (not proxy topics) that are using this type */
  uint32_t refc;                                /* refcount for this record */
};

/* The xt_type member must be at offset 0 so that the type identifier field
   in this type is at offset 0, and a ddsi_type can be used for hash table lookup
   without copying the type identifier in the search template */
DDSRT_STATIC_ASSERT (offsetof (struct ddsi_type, xt) == 0);

DDS_EXPORT char *ddsi_make_typeid_str_impl (struct ddsi_typeid_str *buf, const DDS_XTypes_TypeIdentifier *type_id);
DDS_EXPORT int ddsi_typeid_compare_impl (const struct DDS_XTypes_TypeIdentifier *a, const struct DDS_XTypes_TypeIdentifier *b);
DDS_EXPORT void ddsi_typeid_fini_impl (struct DDS_XTypes_TypeIdentifier *type_id);

DDS_EXPORT void ddsi_typeid_copy_impl (struct DDS_XTypes_TypeIdentifier *dst, const struct DDS_XTypes_TypeIdentifier *src);
DDS_EXPORT void ddsi_typeid_copy_to_impl (struct DDS_XTypes_TypeIdentifier *dst, const ddsi_typeid_t *src);
DDS_EXPORT struct DDS_XTypes_TypeIdentifier * ddsi_typeid_dup_impl (const struct DDS_XTypes_TypeIdentifier *src);
DDS_EXPORT ddsi_typeid_t * ddsi_typeid_dup_from_impl (const struct DDS_XTypes_TypeIdentifier *src);
DDS_EXPORT bool ddsi_typeid_is_none_impl (const struct DDS_XTypes_TypeIdentifier *type_id);

DDS_EXPORT void ddsi_xt_get_typeobject_impl (const struct xt_type *xt, struct DDS_XTypes_TypeObject *to);
DDS_EXPORT dds_return_t ddsi_type_ref_id_locked_impl (struct ddsi_domaingv *gv, struct ddsi_type **type, const struct DDS_XTypes_TypeIdentifier *type_id);
DDS_EXPORT struct ddsi_type * ddsi_type_lookup_locked_impl (struct ddsi_domaingv *gv, const struct DDS_XTypes_TypeIdentifier *type_id);
DDS_EXPORT const struct DDS_XTypes_TypeObject * ddsi_typemap_typeobj (const ddsi_typemap_t *tmap, const struct DDS_XTypes_TypeIdentifier *type_id);

DDS_EXPORT bool ddsi_typeid_is_hash_impl (const struct DDS_XTypes_TypeIdentifier *type_id);
DDS_EXPORT bool ddsi_typeid_is_minimal_impl (const struct DDS_XTypes_TypeIdentifier *type_id);
DDS_EXPORT bool ddsi_typeid_is_complete_impl (const struct DDS_XTypes_TypeIdentifier *type_id);
DDS_EXPORT void ddsi_typeobj_fini_impl (struct DDS_XTypes_TypeObject *typeobj);
DDS_EXPORT dds_return_t ddsi_xt_type_init_impl (struct ddsi_domaingv *gv, struct xt_type *xt, const struct DDS_XTypes_TypeIdentifier *ti, const struct DDS_XTypes_TypeObject *to);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_XT_IMPL_H */
