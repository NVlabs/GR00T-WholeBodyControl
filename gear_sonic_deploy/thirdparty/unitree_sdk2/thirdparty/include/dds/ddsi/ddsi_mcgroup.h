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
#ifndef DDSI_MCGROUP_H
#define DDSI_MCGROUP_H

#include "dds/ddsi/ddsi_tran.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct nn_group_membership;

struct nn_group_membership *new_group_membership (void);
void free_group_membership (struct nn_group_membership *mship);
int ddsi_join_mc (const struct ddsi_domaingv *gv, struct nn_group_membership *mship, ddsi_tran_conn_t conn, const ddsi_locator_t *srcip, const ddsi_locator_t *mcip);
int ddsi_leave_mc (const struct ddsi_domaingv *gv, struct nn_group_membership *mship, ddsi_tran_conn_t conn, const ddsi_locator_t *srcip, const ddsi_locator_t *mcip);
void ddsi_transfer_group_membership (struct nn_group_membership *mship, ddsi_tran_conn_t conn, ddsi_tran_conn_t newconn);
int ddsi_rejoin_transferred_mcgroups (const struct ddsi_domaingv *gv, struct nn_group_membership *mship, ddsi_tran_conn_t conn);

#if defined (__cplusplus)
}
#endif

#endif
