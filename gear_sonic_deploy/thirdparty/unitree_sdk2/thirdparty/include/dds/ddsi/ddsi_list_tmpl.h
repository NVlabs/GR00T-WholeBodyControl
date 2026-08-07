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
#ifndef DDSI_LIST_TMPL_H
#define DDSI_LIST_TMPL_H

#define DDSI_LIST_TYPES_TMPL(prefix_, elemT_, extension_, batch_) \
struct prefix_##_node {                                           \
  struct prefix_##_node *next;                                    \
  uint32_t first, lastp1;                                         \
  elemT_ ary[(batch_)];                                           \
};                                                                \
                                                                  \
struct prefix_ {                                                  \
  struct prefix_##_node *head;                                    \
  struct prefix_##_node *tail;                                    \
  uint32_t count;                                                 \
  extension_                                                      \
};                                                                \
                                                                  \
struct prefix_##_iter {                                           \
  struct prefix_##_node *node;                                    \
  uint32_t idx;                                                   \
};                                                                \
                                                                  \
struct prefix_##_iter_d {                                         \
  struct prefix_ *list;                                           \
  struct prefix_##_node *node;                                    \
  struct prefix_##_node *prev;                                    \
  uint32_t idx;                                                   \
};                                                                \
                                                                  \
typedef int (*prefix_##_eq_fn)(const elemT_, const elemT_);

#ifndef NDEBUG
#define DDSI_LIST_TMPL_POISON(x) do { x = (void *)1; } while (0)
#else
#define DDSI_LIST_TMPL_POISON(x) do {} while (0)
#endif

#define DDSI_LIST_DECLS_TMPL(linkage_, prefix_, elemT_, attrs_) \
linkage_ void prefix_##_init (struct prefix_ *list) attrs_; \
linkage_ void prefix_##_free (struct prefix_ *list) attrs_; \
linkage_ elemT_ prefix_##_insert (struct prefix_ *list, elemT_ o) attrs_; \
linkage_ elemT_ prefix_##_append (struct prefix_ *list, elemT_ o) attrs_; \
linkage_ elemT_ prefix_##_iter_first (const struct prefix_ *list, struct prefix_##_iter *iter) attrs_; \
linkage_ elemT_ prefix_##_iter_next (struct prefix_##_iter *iter) attrs_; \
linkage_ elemT_ *prefix_##_iter_elem_addr (struct prefix_##_iter *iter) attrs_; \
linkage_ elemT_ prefix_##_iter_d_first (struct prefix_ *list, struct prefix_##_iter_d *iter) attrs_; \
linkage_ elemT_ prefix_##_iter_d_next (struct prefix_##_iter_d *iter) attrs_; \
linkage_ void prefix_##_iter_d_remove (struct prefix_##_iter_d *iter) attrs_; \
linkage_ elemT_ prefix_##_remove (struct prefix_ *list, elemT_ o, prefix_##_eq_fn) attrs_; \
linkage_ elemT_ prefix_##_take_first (struct prefix_ *list) attrs_; \
linkage_ elemT_ prefix_##_take_last (struct prefix_ *list) attrs_; \
linkage_ uint32_t prefix_##_count (const struct prefix_ *list) attrs_; \
linkage_ void prefix_##_append_list (struct prefix_ *list, struct prefix_ *b) attrs_; \
linkage_ elemT_ *prefix_##_index_addr (struct prefix_ *list, uint32_t index) attrs_; \
linkage_ elemT_ prefix_##_index (struct prefix_ *list, uint32_t index) attrs_;

#define DDSI_LIST_CODE_TMPL(linkage_, prefix_, elemT_, null_, malloc_, free_) \
linkage_ void prefix_##_init (struct prefix_ *list)                   \
{                                                                     \
  list->head = NULL;                                                  \
  list->tail = NULL;                                                  \
  list->count = 0;                                                    \
}                                                                     \
                                                                      \
linkage_ void prefix_##_free (struct prefix_ *list)                   \
{                                                                     \
  /* Note: just free, not re-init */                                  \
  struct prefix_##_node *n;                                           \
  while ((n = list->head) != NULL)                                    \
  {                                                                   \
    list->head = n->next;                                             \
    free_ (n);                                                        \
  }                                                                   \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_insert (struct prefix_ *list, elemT_ o)     \
{                                                                     \
  struct prefix_##_node *n;                                           \
  const uint32_t bs = (uint32_t) (sizeof (n->ary) / sizeof (n->ary[0])); \
  if (list->head != NULL && list->head->first > 0)                    \
    n = list->head;                                                   \
  else                                                                \
  {                                                                   \
    if ((n = (malloc_ (sizeof (struct prefix_##_node)))) == NULL)     \
      return null_;                                                   \
    n->next = list->head;                                             \
    n->first = n->lastp1 = bs;                                        \
    if (list->head == NULL)                                           \
      list->tail = n;                                                 \
    list->head = n;                                                   \
  }                                                                   \
  n->ary[--n->first] = o;                                             \
  list->count++;                                                      \
  return o;                                                           \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_append (struct prefix_ *list, elemT_ o)     \
{                                                                     \
  struct prefix_##_node *n;                                           \
  const uint32_t bs = (uint32_t) (sizeof (n->ary) / sizeof (n->ary[0])); \
  if (list->head != NULL && list->tail->lastp1 < bs)                  \
    n = list->tail;                                                   \
  else                                                                \
  {                                                                   \
    if ((n = (malloc_ (sizeof (struct prefix_##_node)))) == NULL)     \
      return null_;                                                   \
    n->next = NULL;                                                   \
    n->first = n->lastp1 = 0;                                         \
    if (list->head == NULL)                                           \
      list->head = n;                                                 \
    else                                                              \
      list->tail->next = n;                                           \
    list->tail = n;                                                   \
  }                                                                   \
  n->ary[n->lastp1++] = o;                                            \
  list->count++;                                                      \
  return o;                                                           \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_iter_first (const struct prefix_ *list, struct prefix_##_iter *iter) \
{                                                                     \
  iter->node = list->head;                                            \
  if (iter->node == NULL)                                             \
  {                                                                   \
    iter->idx = 0;                                                    \
    return null_;                                                     \
  }                                                                   \
  iter->idx = iter->node->first;                                      \
  if (iter->node->first < iter->node->lastp1)                         \
    return iter->node->ary[iter->idx];                                \
  return null_;                                                       \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_iter_next (struct prefix_##_iter *iter)     \
{                                                                     \
  /* You MAY NOT call _list_iter_next after having received a null    \
    * pointer from _iter_first or _iter_next */                       \
  assert (iter->node != NULL);                                        \
  if (iter->idx+1 < iter->node->lastp1)                               \
    return iter->node->ary[++iter->idx];                              \
  if (iter->node->next == NULL)                                       \
    return null_;                                                     \
  iter->node = iter->node->next;                                      \
  iter->idx = iter->node->first;                                      \
  return iter->node->ary[iter->idx];                                  \
}                                                                     \
                                                                      \
linkage_ elemT_ *prefix_##_iter_elem_addr (struct prefix_##_iter *iter) \
{                                                                     \
  assert (iter->node != NULL);                                        \
  return &iter->node->ary[iter->idx];                                 \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_iter_d_first (struct prefix_ *list, struct prefix_##_iter_d *iter) \
{                                                                     \
  iter->list = list;                                                  \
  iter->node = list->head;                                            \
  iter->prev = NULL;                                                  \
  if (iter->node == NULL)                                             \
  {                                                                   \
    iter->idx = 0;                                                    \
    return null_;                                                     \
  }                                                                   \
  iter->idx = iter->node->first;                                      \
  if (iter->node->first < iter->node->lastp1)                         \
    return iter->node->ary[iter->idx];                                \
  return null_;                                                       \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_iter_d_next (struct prefix_##_iter_d *iter) \
{                                                                     \
  /* You MAY NOT call _list_iter_d_next after having received a null  \
    * pointer from _iter_d_first or _iter_d_next */                   \
  if (iter->node == NULL)                                             \
    return prefix_##_iter_d_first (iter->list, iter);                 \
  if (iter->idx+1 < iter->node->lastp1)                               \
    return iter->node->ary[++iter->idx];                              \
  if (iter->node->next == NULL)                                       \
    return null_;                                                     \
  iter->prev = iter->node;                                            \
  iter->node = iter->node->next;                                      \
  iter->idx = iter->node->first;                                      \
  return iter->node->ary[iter->idx];                                  \
}                                                                     \
                                                                      \
linkage_ void prefix_##_iter_d_remove (struct prefix_##_iter_d *iter) \
{                                                                     \
  struct prefix_ * const list = iter->list;                           \
  struct prefix_##_node * const n = iter->node;                       \
  uint32_t j;                                                         \
  assert(iter->node);                                                 \
  list->count--;                                                      \
  for (j = iter->idx; j > n->first; j--)                              \
    n->ary[j] = n->ary[j-1];                                          \
  n->first++;                                                         \
  if (n->first == n->lastp1)                                          \
  {                                                                   \
    if (n == list->tail)                                              \
      list->tail = iter->prev;                                        \
    if (iter->prev)                                                   \
    {                                                                 \
      iter->prev->next = n->next;                                     \
      iter->node = iter->prev;                                        \
      iter->idx = iter->prev->lastp1;                                 \
      DDSI_LIST_TMPL_POISON(iter->prev);                              \
    }                                                                 \
    else                                                              \
    {                                                                 \
      list->head = n->next;                                           \
      iter->node = NULL; /* removed first entry, restart */           \
    }                                                                 \
    free_ (n);                                                        \
  }                                                                   \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_remove (struct prefix_ *list, elemT_ o, prefix_##_eq_fn equals) \
{                                                                     \
  struct prefix_##_iter_d iter;                                       \
  elemT_ obj;                                                         \
  for (obj = prefix_##_iter_d_first (list, &iter); !(equals (obj, null_)); obj = prefix_##_iter_d_next (&iter)) \
  {                                                                   \
    if (equals (obj, o))                                              \
    {                                                                 \
      prefix_##_iter_d_remove (&iter);                                \
      return obj;                                                     \
    }                                                                 \
  }                                                                   \
  return null_;                                                       \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_take_first (struct prefix_ *list)           \
{                                                                     \
  if (list->count == 0)                                               \
    return null_;                                                     \
  struct prefix_##_iter_d iter;                                       \
  elemT_ obj = prefix_##_iter_d_first (list, &iter);                  \
  prefix_##_iter_d_remove (&iter);                                    \
  return obj;                                                         \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_take_last (struct prefix_ *list)            \
{                                                                     \
  if (list->count == 0)                                               \
    return null_;                                                     \
  struct prefix_##_iter_d iter;                                       \
  uint32_t i;                                                         \
  elemT_ obj;                                                         \
  obj = prefix_##_iter_d_first (list, &iter);                         \
  for (i = 0; i < list->count - 1; i++)                               \
    obj = prefix_##_iter_d_next (&iter);                              \
  prefix_##_iter_d_remove (&iter);                                    \
  return obj;                                                         \
}                                                                     \
                                                                      \
linkage_ uint32_t prefix_##_count (const struct prefix_ *list)        \
{                                                                     \
  return list->count;                                                 \
}                                                                     \
                                                                      \
linkage_ void prefix_##_append_list (struct prefix_ *list, struct prefix_ *b) \
{                                                                     \
  if (list->head == NULL)                                             \
    *list = *b;                                                       \
  else if (b->head != NULL)                                           \
  {                                                                   \
    list->tail->next = b->head;                                       \
    list->tail = b->tail;                                             \
    list->count += b->count;                                          \
  }                                                                   \
}                                                                     \
                                                                      \
linkage_ elemT_ *prefix_##_index_addr (struct prefix_ *list, uint32_t index) \
{                                                                     \
  struct prefix_##_node *n;                                           \
  uint32_t pos = 0;                                                   \
  if (index >= list->count)                                           \
    return NULL;                                                      \
  if (index == list->count - 1)                                       \
  {                                                                   \
    n = list->tail;                                                   \
    pos = list->count - (n->lastp1 - n->first);                       \
  }                                                                   \
  else                                                                \
  {                                                                   \
    for (n = list->head; n; n = n->next)                              \
    {                                                                 \
      const uint32_t c = n->lastp1 - n->first;                        \
      if (pos + c > index)                                            \
        break;                                                        \
      pos += c;                                                       \
    }                                                                 \
  }                                                                   \
  if (n == NULL)                                                      \
    return NULL;                                                      \
  assert (pos <= index && index < pos + n->lastp1 - n->first);        \
  return &n->ary[n->first + (index - pos)];                           \
}                                                                     \
                                                                      \
linkage_ elemT_ prefix_##_index (struct prefix_ *list, uint32_t index) \
{                                                                     \
  elemT_ *p = prefix_##_index_addr (list, index);                     \
  if (p == NULL)                                                      \
    return null_;                                                     \
  return *p;                                                          \
}

#endif /* DDSI_LIST_TMPL_H */
