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
#ifndef DDSRT_EXPAND_VARS_H
#define DDSRT_EXPAND_VARS_H

#include "dds/export.h"
#include "dds/ddsrt/attributes.h"
#include "dds/ddsrt/retcode.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef const char * (*expand_lookup_fn)(const char *name, void *data);

/**
 * @brief Expand variables within string.
 *
 * Expands ${X}, ${X:-Y}, ${X:+Y}, ${X:?Y} forms, but not $X.
 *
 * The result string should be freed with ddsrt_free().
 *
 * @param[in]  string  String to expand.
 * @param[in]  lookup  Lookup function to retrieve replacement value
 * @param[in]  data    Data passed to lookup function
 *
 * @returns Allocated char*.
 *
 * @retval NULL
 *             Expansion failed.
 * @retval Pointer
 *             Copy of the string argument with the variables expanded.
 */
DDS_EXPORT char*
ddsrt_expand_vars(
  const char *string,
  expand_lookup_fn lookup,
  void * data);

/**
 * @brief Expand variables within string.
 *
 * Expands $X, ${X}, ${X:-Y}, ${X:+Y}, ${X:?Y} forms, $ and \
 * can be escaped with \.
 *
 * The result string should be freed with ddsrt_free().
 *
 * @param[in]  string  String to expand.
 * @param[in]  lookup  Lookup function to retrieve replacement value
 * @param[in]  data    Data passed to lookup function
 *
 * @returns Allocated char*.
 *
 * @retval NULL
 *             Expansion failed.
 * @retval Pointer
 *             Copy of the string argument with the variables expanded.
 */
DDS_EXPORT char*
ddsrt_expand_vars_sh(
  const char *string,
  expand_lookup_fn lookup,
  void * data);

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_EXPAND_VARS_H */
