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
#ifndef DDSRT_IO_H
#define DDSRT_IO_H

#include <stdarg.h>
#include <stdio.h>

#include "dds/export.h"
#include "dds/ddsrt/attributes.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Write a formatted string to a newly allocated buffer.
 */
DDS_EXPORT int
ddsrt_vasprintf(
  char **strp,
  const char *fmt,
  va_list ap);

/**
 * @brief Write a formatted string to a newly allocated buffer.
 */
DDS_EXPORT int
ddsrt_asprintf(
  char **strp,
  const char *fmt,
  ...) ddsrt_attribute_format ((printf, 2, 3));

#if defined(_MSC_VER) && (_MSC_VER < 1900)
extern int snprintf(char *s, size_t n, const char *format, ...);
#endif

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_IO_H */
