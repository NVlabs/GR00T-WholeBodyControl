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
#ifndef DDSRT_ENVIRON_H
#define DDSRT_ENVIRON_H

#include "dds/export.h"
#include "dds/ddsrt/attributes.h"
#include "dds/ddsrt/expand_vars.h"
#include "dds/ddsrt/retcode.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Get value for environment variable.
 *
 * @param[in]  name   Environment variable name.
 * @param[out] value  Alias to value of environment variable - must not be modified
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Environment variable written to @buf.
 * @retval DDS_RETCODE_NOT_FOUND
 *             Environment variable not found.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             FIXME: document
 * @retval DDS_RETCODE_OUT_OF_RESOURCES
 *             FIXME: document
 * @retval DDS_RETCODE_ERROR
 *             Unspecified error.
 */
DDS_EXPORT dds_return_t
ddsrt_getenv(
  const char *name,
  const char **value)
ddsrt_nonnull_all;

/**
 * @brief Set environment variable value.
 *
 * Sets the environment variable to the value specified in value, or
 * alternatively, unsets the environment variable if value is an empty string.
 *
 * @param[in]  name   Environment variable name.
 * @param[in]  value  Value to set environment variable to.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Environment variable successfully set to @value.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Invalid environment variable name.
 * @retval DDS_RETCODE_OUT_OF_RESOURCES
 *             Not enough system resources to set environment variable.
 * @retval DDS_RETCODE_ERROR
 *             Unspecified system error.
 */
DDS_EXPORT dds_return_t
ddsrt_setenv(
  const char *name,
  const char *value)
ddsrt_nonnull_all;

/**
 * @brief Unset environment variable value.
 *
 * @param[in]  name  Environment variable name.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Environment variable successfully unset.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Invalid environment variable name.
 * @retval DDS_RETCODE_OUT_OF_RESOURCES
 *             Not enough system resources to unset environment variable.
 * @retval DDS_RETCODE_ERROR
 *             Unspecified system error.
 */
DDS_EXPORT dds_return_t
ddsrt_unsetenv(
  const char *name)
ddsrt_nonnull_all;

/**
 * @brief Expand environment variables within string.
 *
 * Expands ${X}, ${X:-Y}, ${X:+Y}, ${X:?Y} forms, but not $X.
 *
 * The result string should be freed with ddsrt_free().
 *
 * @param[in]  string  String to expand.
 * @param[in]  domid   Domain id that this is relevant to
 *                     UINT32_MAX means none (see logging)
 *                     also made available as
 *                        ${CYCLONEDDS_DOMAIN_ID}
 *
 * @returns Allocated char*.
 *
 * @retval NULL
 *             Expansion failed.
 * @retval Pointer
 *             Copy of the string argument with the environment
 *             variables expanded.
 */
DDS_EXPORT char*
ddsrt_expand_envvars(
  const char *string,
  uint32_t domid);

/**
 * @brief Expand environment variables within string.
 *
 * Expands $X, ${X}, ${X:-Y}, ${X:+Y}, ${X:?Y} forms, $ and \
 * can be escaped with \.
 *
 * The result string should be freed with ddsrt_free().
 *
 * @param[in]  string  String to expand.
 *
 * @returns Allocated char*.
 *
 * @retval NULL
 *             Expansion failed.
 * @retval Pointer
 *             Copy of the string argument with the environment
 *             variables expanded.
 */
DDS_EXPORT char*
ddsrt_expand_envvars_sh(
  const char *string,
  uint32_t domid);


#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_ENVIRON_H */
