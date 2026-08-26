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
#ifndef DDSRT_STRTOL_H
#define DDSRT_STRTOL_H

#include <stdint.h>

#include "dds/export.h"
#include "dds/ddsrt/retcode.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Convert a character to an integer value
 *
 * Translates the numeric value of the provided character. For characters in range
 * '0' to '9' the returned integer value is 0-9. For the range 'a' to 'z' and 'A'
 * to 'Z', the numeric return value is 10-36.
 *
 * @param[in] chr   The character
 *
 * @returns The integer value for the character, or -1 in case @chr cannot be translated to a numeric value
 */
DDS_EXPORT int32_t ddsrt_todigit(const int chr);

/**
 * @brief Convert a string to a long long integer.
 *
 * Translate @str to a long long integer considering base, and sign. If @base
 * is 0, base is determined from @str. A prefix of "0x" or "0X" will cause the
 * number be read in base 16 (hexadecimal), otherwise base 10 (decimal) is
 * used, unless the first character is '0', in which case the number will be
 * read in base 8 (octal).
 *
 * @param[in]   str     String to convert into a number.
 * @param[out]  endptr  If not NULL, a char* where the address of first invalid
 *                      character is stored.
 * @param[in]   base    Base to use. Must be a base between 2 and 36, or 0 to
 *                      determine from @str.
 * @param[out]  llng    A long long integer where the number is stored.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             String successfully converted to an integer.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Base is invalid.
 * @retval DDS_RETCODE_OUT_OF_RANGE
 *             String converted to an integer, but was out of range.
 */
DDS_EXPORT dds_return_t
ddsrt_strtoll(
  const char *str,
  char **endptr,
  int32_t base,
  long long *llng);

/**
 * @brief Convert a string to an unsigned long long integer.
 *
 * Translate @str to an unsigned long long integer considering base, and sign.
 * If @base is 0, base is determined from @str. A prefix of "0x" or "0X" will
 * cause the number be read in base 16 (hexadecimal), otherwise base 10
 * (decimal) is used, unless the first character is '0', in which case the
 * number will be read in base 8 (octal).
 *
 * @param[in]   str     String to convert into a number.
 * @param[out]  endptr  If not NULL, a char* where the address of first invalid
 *                      character is stored.
 * @param[in]   base    Base to use. Must be a base between 2 and 36, or 0 to
 *                      determine from @str.
 * @param[out]  ullng   A long long integer where the number is stored.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             String successfully converted to an integer.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Base is invalid.
 * @retval DDS_RETCODE_OUT_OF_RANGE
 *             String converted to an integer, but was out of range.
 */
DDS_EXPORT dds_return_t
ddsrt_strtoull(
  const char *str,
  char **endptr,
  int32_t base,
  unsigned long long *ullng);

/**
 * @brief Convert a string to a long long integer.
 *
 * @param[in]  str   String to convert into a long long integer.
 * @param[in]  llng  A long long integer where the number is stored.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             String successfully converted to an integer.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Base is invalid.
 * @retval DDS_RETCODE_OUT_OF_RANGE
 *             String converted to an integer, but was out of range.
 */
DDS_EXPORT dds_return_t
ddsrt_atoll(
  const char *str,
  long long *llng);

/**
 * @brief Convert a string to an unsigned long long integer.
 *
 * @param[in]   str    String to conver into an unsigned long long integer.
 * @param[out]  ullng  An unsigned long long integer where the number is stored.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             String successfully converted to an integer.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Base is invalid.
 * @retval DDS_RETCODE_OUT_OF_RANGE
 *             String converted to an integer, but was out of range.
 */
DDS_EXPORT dds_return_t
ddsrt_atoull(
  const char *str,
  unsigned long long *ullng);

/**
 * @brief Convert a long long integer into a string.
 *
 * @param[in]   num     Long long integer to convert into a string.
 * @param[in]   str     Buffer where string representation is written.
 * @param[in]   len     Number of bytes available in buffer.
 * @param[out]  endptr  A char* where the address of the null terminating byte
 *                      is stored.
 *
 * @returns The value of @str on success, otherwise NULL.
 */
DDS_EXPORT char *
ddsrt_lltostr(
  long long num,
  char *str,
  size_t len,
  char **endptr);

/**
 * @brief Convert an unsigned long long integer into a string.
 *
 * @param[in]   num     Unsigned long long integer to covert into a string.
 * @param[in]   str     Buffer where string representation is stored.
 * @param[in]   len     Number of bytes available in buffer.
 * @param[out]  endptr  A char* where the adress of the null terminating byte
 *                      is stored.
 *
 * @returns The value of @str on success, otherwise NULL.
 */
DDS_EXPORT char *
ddsrt_ulltostr(
  unsigned long long num,
  char *str,
  size_t len,
  char **endptr);

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_STRTOL_H */
