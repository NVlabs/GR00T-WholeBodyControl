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
#ifndef DDSRT_ARCH_H
#define DDSRT_ARCH_H

#if _WIN32
# if _WIN64
#   define DDSRT_64BIT 1
# else
#   define DDSRT_64BIT 0
# endif
#else
# if defined(_LP64)
#   define DDSRT_64BIT 1
# else
#   define DDSRT_64BIT 0
# endif
#endif

#endif /* DDSRT_ARCH_H */
