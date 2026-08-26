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
#ifndef DDSRT_TYPES_H
#define DDSRT_TYPES_H

#include <stdbool.h>
#include <stdint.h>

#if _WIN32
# include "dds/ddsrt/types/windows.h"
#elif __VXWORKS__
# include "dds/ddsrt/types/vxworks.h"
#else
# include "dds/ddsrt/types/posix.h"
#endif

#define PRIdSIZE "zd"
#define PRIuSIZE "zu"
#define PRIxSIZE "zx"

#endif /* DDSRT_TYPES_H */
