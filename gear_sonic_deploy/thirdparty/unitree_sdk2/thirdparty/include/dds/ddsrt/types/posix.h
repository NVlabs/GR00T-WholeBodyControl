/*
 * Copyright(c) 2006 to 2019 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSRT_TYPES_POSIX_H
#define DDSRT_TYPES_POSIX_H

#include <stdint.h>
#include <inttypes.h>
#if defined(__IAR_SYSTEMS_ICC__)
typedef long int ssize_t;
#else
#include <unistd.h>
#endif

#endif /* DDSRT_TYPES_POSIX_H */
