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
#ifndef DDSRT_CIRCLIST_H
#define DDSRT_CIRCLIST_H

/* Circular doubly linked list implementation */

#include <stdbool.h>
#include <stdint.h>
#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

#define DDSRT_FROM_CIRCLIST(typ_, member_, cle_) ((typ_ *) ((char *) (cle_) - offsetof (typ_, member_)))

struct ddsrt_circlist {
  struct ddsrt_circlist_elem *latest; /* pointer to latest inserted element */
};

struct ddsrt_circlist_elem {
  struct ddsrt_circlist_elem *next;
  struct ddsrt_circlist_elem *prev;
};

DDS_EXPORT void ddsrt_circlist_init (struct ddsrt_circlist *list);
DDS_EXPORT bool ddsrt_circlist_isempty (const struct ddsrt_circlist *list);
DDS_EXPORT void ddsrt_circlist_append (struct ddsrt_circlist *list, struct ddsrt_circlist_elem *elem);
DDS_EXPORT void ddsrt_circlist_remove (struct ddsrt_circlist *list, struct ddsrt_circlist_elem *elem);
DDS_EXPORT struct ddsrt_circlist_elem *ddsrt_circlist_oldest (const struct ddsrt_circlist *list);
DDS_EXPORT struct ddsrt_circlist_elem *ddsrt_circlist_latest (const struct ddsrt_circlist *list);

#endif /* DDSRT_CIRCLIST_H */
