/*
 * Copyright(c) 2020 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSI_WRADDRSET_H
#define DDSI_WRADDRSET_H

#include <stddef.h>
#include <stdbool.h>

#if defined (__cplusplus)
extern "C" {
#endif

struct addrset;
struct ddsi_writer;

struct addrset *compute_writer_addrset (const struct ddsi_writer *wr);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_WRADDRSET_H */
