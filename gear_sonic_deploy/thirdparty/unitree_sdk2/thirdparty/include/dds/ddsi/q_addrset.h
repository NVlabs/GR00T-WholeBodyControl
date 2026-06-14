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
#ifndef NN_ADDRSET_H
#define NN_ADDRSET_H

#include "dds/ddsrt/sync.h"
#include "dds/ddsrt/avl.h"
#include "dds/ddsi/q_log.h"
#include "dds/ddsi/q_thread.h"
#include "dds/ddsi/q_protocol.h"
#include "dds/ddsi/q_feature_check.h"

#if defined (__cplusplus)
extern "C" {
#endif

typedef struct addrset_node {
  ddsrt_avl_node_t avlnode;
  ddsi_xlocator_t loc;
} * addrset_node_t;

struct addrset {
  ddsrt_mutex_t lock;
  ddsrt_atomic_uint32_t refc;
  ddsrt_avl_ctree_t ucaddrs, mcaddrs;
};

typedef void (*addrset_forall_fun_t) (const ddsi_xlocator_t *loc, void *arg);
typedef ssize_t (*addrset_forone_fun_t) (const ddsi_xlocator_t *loc, void *arg);

DDS_EXPORT struct addrset *new_addrset (void);
DDS_EXPORT struct addrset *ref_addrset (struct addrset *as);
DDS_EXPORT void unref_addrset (struct addrset *as);
DDS_EXPORT void add_locator_to_addrset (const struct ddsi_domaingv *gv, struct addrset *as, const ddsi_locator_t *loc);
DDS_EXPORT void add_xlocator_to_addrset (const struct ddsi_domaingv *gv, struct addrset *as, const ddsi_xlocator_t *loc);
DDS_EXPORT void remove_from_addrset (const struct ddsi_domaingv *gv, struct addrset *as, const ddsi_xlocator_t *loc);
DDS_EXPORT int addrset_purge (struct addrset *as);
int compare_locators (const ddsi_locator_t *a, const ddsi_locator_t *b);
int compare_xlocators (const ddsi_xlocator_t *a, const ddsi_xlocator_t *b);

/* These lock ASADD, then lock/unlock AS any number of times, then
   unlock ASADD */
void copy_addrset_into_addrset_uc (const struct ddsi_domaingv *gv, struct addrset *as, const struct addrset *asadd);
void copy_addrset_into_addrset_mc (const struct ddsi_domaingv *gv, struct addrset *as, const struct addrset *asadd);
void copy_addrset_into_addrset (const struct ddsi_domaingv *gv, struct addrset *as, const struct addrset *asadd);

size_t addrset_count (const struct addrset *as);
size_t addrset_count_uc (const struct addrset *as);
size_t addrset_count_mc (const struct addrset *as);
int addrset_empty_uc (const struct addrset *as);
int addrset_empty_mc (const struct addrset *as);
int addrset_empty (const struct addrset *as);
int addrset_any_uc (const struct addrset *as, ddsi_xlocator_t *dst);
int addrset_any_mc (const struct addrset *as, ddsi_xlocator_t *dst);
void addrset_any_uc_else_mc_nofail (const struct addrset *as, ddsi_xlocator_t *dst);

/* Keeps AS locked */
int addrset_forone (struct addrset *as, addrset_forone_fun_t f, void *arg);
DDS_EXPORT void addrset_forall (struct addrset *as, addrset_forall_fun_t f, void *arg);
size_t addrset_forall_count (struct addrset *as, addrset_forall_fun_t f, void *arg);
size_t addrset_forall_uc_else_mc_count (struct addrset *as, addrset_forall_fun_t f, void *arg);
size_t addrset_forall_mc_count (struct addrset *as, addrset_forall_fun_t f, void *arg);
void nn_log_addrset (struct ddsi_domaingv *gv, uint32_t tf, const char *prefix, const struct addrset *as);

/* Tries to lock A then B for a decent check, returning false if
   trylock B fails */
int addrset_eq_onesidederr (const struct addrset *a, const struct addrset *b);

int is_unspec_locator (const ddsi_locator_t *loc);
int is_unspec_xlocator (const ddsi_xlocator_t *loc);
void set_unspec_locator (ddsi_locator_t *loc);
void set_unspec_xlocator (ddsi_xlocator_t *loc);

struct ddsi_domaingv;
int add_addresses_to_addrset (const struct ddsi_domaingv *gv, struct addrset *as, const char *addrs, int port_mode, const char *msgtag, int req_mc);

#ifdef DDS_HAS_SSM
int addrset_contains_ssm (const struct ddsi_domaingv *gv, const struct addrset *as);
int addrset_any_ssm (const struct ddsi_domaingv *gv, const struct addrset *as, ddsi_xlocator_t *dst);
int addrset_any_non_ssm_mc (const struct ddsi_domaingv *gv, const struct addrset *as, ddsi_xlocator_t *dst);
void copy_addrset_into_addrset_no_ssm_mc (const struct ddsi_domaingv *gv, struct addrset *as, const struct addrset *asadd);
void copy_addrset_into_addrset_no_ssm (const struct ddsi_domaingv *gv, struct addrset *as, const struct addrset *asadd);
#endif

#if defined (__cplusplus)
}
#endif
#endif /* NN_ADDRSET_H */
