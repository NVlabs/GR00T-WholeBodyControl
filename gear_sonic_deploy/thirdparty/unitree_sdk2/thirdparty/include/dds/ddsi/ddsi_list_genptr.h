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
#ifndef DDSI_LIST_GENPTR_H
#define DDSI_LIST_GENPTR_H

#include <assert.h>
#include "dds/export.h"
#include "dds/ddsrt/types.h"
#include "dds/ddsi/ddsi_list_tmpl.h"

#define NOARG
DDSI_LIST_TYPES_TMPL(generic_ptr_list, void *, NOARG, 32)
DDSI_LIST_DECLS_TMPL(extern, generic_ptr_list, void *, ddsrt_attribute_unused)
#undef NOARG

#define DDSI_LIST_GENERIC_PTR_TYPES(prefix_) \
typedef struct generic_ptr_list_node prefix_##_node_t; \
typedef struct generic_ptr_list prefix_##_t; \
typedef struct generic_ptr_list_iter prefix_##_iter_t; \
typedef struct generic_ptr_list_iter_d prefix_##_iter_d_t;

#define DDSI_LIST_GENERIC_PTR_DECL(linkage_, prefix_, elemT_, attrs_)                          \
linkage_ void prefix_##_init (prefix_##_t *list) attrs_;                                       \
linkage_ void prefix_##_free (prefix_##_t *list) attrs_;                                       \
linkage_ elemT_ prefix_##_insert (prefix_##_t *list, elemT_ o) attrs_;                         \
linkage_ elemT_ prefix_##_append (prefix_##_t *list, elemT_ o) attrs_;                         \
linkage_ elemT_ prefix_##_iter_first (const prefix_##_t *list, prefix_##_iter_t *iter) attrs_; \
linkage_ elemT_ prefix_##_iter_next (prefix_##_iter_t *iter) attrs_;                           \
linkage_ elemT_ *prefix_##_iter_elem_addr (prefix_##_iter_t *iter) attrs_;                     \
linkage_ elemT_ prefix_##_iter_d_first (prefix_##_t *list, prefix_##_iter_d_t *iter) attrs_;   \
linkage_ elemT_ prefix_##_iter_d_next (prefix_##_iter_d_t *iter) attrs_;                       \
linkage_ void prefix_##_iter_d_remove (prefix_##_iter_d_t *iter) attrs_;                       \
linkage_ elemT_ prefix_##_remove (prefix_##_t *list, elemT_ o) attrs_;                         \
linkage_ elemT_ prefix_##_take_first (prefix_##_t *list) attrs_;                               \
linkage_ elemT_ prefix_##_take_last (prefix_##_t *list) attrs_;                                \
linkage_ uint32_t prefix_##_count (const prefix_##_t *list) attrs_;                            \
linkage_ void prefix_##_append_list (prefix_##_t *list, prefix_##_t *b) attrs_;                \
linkage_ elemT_ *prefix_##_index_addr (prefix_##_t *list, uint32_t index) attrs_;              \
linkage_ elemT_ prefix_##_index (prefix_##_t *list, uint32_t index) attrs_;

#define DDSI_LIST_GENERIC_PTR_CODE(linkage_, prefix_, elemT_, equals_)                   \
linkage_ void prefix_##_init (prefix_##_t *list) {                                       \
  generic_ptr_list_init ((struct generic_ptr_list *) list);                          \
}                                                                                        \
linkage_ void prefix_##_free (prefix_##_t *list) {                                       \
  generic_ptr_list_free ((struct generic_ptr_list *) list);                          \
}                                                                                        \
linkage_ elemT_ prefix_##_insert (prefix_##_t *list, elemT_ o) {                         \
  return (elemT_) generic_ptr_list_insert ((struct generic_ptr_list *) list, (void *) o); \
}                                                                                        \
linkage_ elemT_ prefix_##_append (prefix_##_t *list, elemT_ o) {                         \
  return (elemT_) generic_ptr_list_append ((struct generic_ptr_list *) list, (void *) o); \
}                                                                                        \
linkage_ elemT_ prefix_##_iter_first (const prefix_##_t *list, prefix_##_iter_t *iter) { \
  return (elemT_) generic_ptr_list_iter_first ((struct generic_ptr_list *) list, (struct generic_ptr_list_iter *) iter); \
}                                                                                        \
linkage_ elemT_ prefix_##_iter_next (prefix_##_iter_t *iter) {                           \
  return (elemT_) generic_ptr_list_iter_next ((struct generic_ptr_list_iter *) iter);    \
}                                                                                        \
linkage_ elemT_ *prefix_##_iter_elem_addr (prefix_##_iter_t *iter) {                     \
  return (elemT_ *) generic_ptr_list_iter_elem_addr ((struct generic_ptr_list_iter *) iter); \
}                                                                                        \
linkage_ elemT_ prefix_##_iter_d_first (prefix_##_t *list, prefix_##_iter_d_t *iter) {   \
  return (elemT_) generic_ptr_list_iter_d_first ((struct generic_ptr_list *) list, (struct generic_ptr_list_iter_d *) iter); \
}                                                                                        \
linkage_ elemT_ prefix_##_iter_d_next (prefix_##_iter_d_t *iter) {                       \
  return (elemT_) generic_ptr_list_iter_d_next ((struct generic_ptr_list_iter_d *) iter);\
}                                                                                        \
linkage_ void prefix_##_iter_d_remove (prefix_##_iter_d_t *iter) {                       \
  generic_ptr_list_iter_d_remove ((struct generic_ptr_list_iter_d *) iter);              \
}                                                                                        \
linkage_ elemT_ prefix_##_remove (prefix_##_t *list, elemT_ o) {                         \
  return (elemT_) generic_ptr_list_remove ((struct generic_ptr_list *) list, (void *) o, (generic_ptr_list_eq_fn) equals_); \
}                                                                                        \
linkage_ elemT_ prefix_##_take_first (prefix_##_t *list) {                               \
  return (elemT_) generic_ptr_list_take_first ((struct generic_ptr_list *) list);    \
}                                                                                        \
linkage_ elemT_ prefix_##_take_last (prefix_##_t *list) {                                \
  return (elemT_) generic_ptr_list_take_last ((struct generic_ptr_list *) list);     \
}                                                                                        \
linkage_ uint32_t prefix_##_count (const prefix_##_t *list) {                            \
  return generic_ptr_list_count ((struct generic_ptr_list *) list);                  \
}                                                                                        \
linkage_ void prefix_##_append_list (prefix_##_t *list, prefix_##_t *b) {                \
  generic_ptr_list_append_list ((struct generic_ptr_list *) list, (struct generic_ptr_list *) b); \
}                                                                                        \
linkage_ elemT_ *prefix_##_index_addr (prefix_##_t *list, uint32_t index) {              \
  return (elemT_ *) generic_ptr_list_index_addr ((struct generic_ptr_list *) list, index); \
}                                                                                        \
linkage_ elemT_ prefix_##_index (prefix_##_t *list, uint32_t index) {                    \
  return (elemT_) generic_ptr_list_index ((struct generic_ptr_list *) list, index);  \
}

#endif /* DDSI_LIST_GENPTR_H */
