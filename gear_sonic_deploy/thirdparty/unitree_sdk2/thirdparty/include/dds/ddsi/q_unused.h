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
#ifndef NN_UNUSED_H
#define NN_UNUSED_H

#ifdef __GNUC__
#define UNUSED_ARG(x) x __attribute__ ((unused))
#else
#define UNUSED_ARG(x) x
#endif

#ifndef NDEBUG
#define UNUSED_ARG_NDEBUG(x) x
#else
#define UNUSED_ARG_NDEBUG(x) UNUSED_ARG (x)
#endif

#endif /* NN_UNUSED_H */
