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
#ifndef DDSRT_FILESYSTEM_POSIX_H
#define DDSRT_FILESYSTEM_POSIX_H

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <limits.h>

typedef DIR *ddsrt_dir_handle_t;
typedef mode_t ddsrt_mode_t;

#define DDSRT_PATH_MAX PATH_MAX
#define DDSRT_FILESEPCHAR '/'

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_FILESYSTEM_POSIX_H */
