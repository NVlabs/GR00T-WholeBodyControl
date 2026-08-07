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
#ifndef DDSRT_MISC_H
#define DDSRT_MISC_H

#include <limits.h>

#if defined (__cplusplus)
extern "C" {
#endif

#define DDSRT_STRINGIFY(x) DDSRT_STRINGIFY1(x)
#define DDSRT_STRINGIFY1(x) #x

#if defined(__clang__) || \
    defined(__GNUC__) && ((__GNUC__ * 100) + __GNUC_MINOR__) >= 402
# define DDSRT_STR(s) #s
# define DDSRT_JOINSTR(x,y) DDSRT_STR(x ## y)
# define DDSRT_DO_PRAGMA(x) _Pragma(#x)
# define DDSRT_PRAGMA(x) DDSRT_DO_PRAGMA(GCC diagnostic x)

# if defined(__clang__)
#   define DDSRT_WARNING_CLANG_OFF(x) \
      DDSRT_PRAGMA(push) \
      DDSRT_PRAGMA(ignored DDSRT_JOINSTR(-W,x))
#   define DDSRT_WARNING_CLANG_ON(x) \
      DDSRT_PRAGMA(pop)
# elif ((__GNUC__ * 100) + __GNUC_MINOR__) >= 406
#   define DDSRT_WARNING_GNUC_OFF(x) \
      DDSRT_PRAGMA(push) \
      DDSRT_PRAGMA(ignored DDSRT_JOINSTR(-W,x))
#   define DDSRT_WARNING_GNUC_ON(x) \
      DDSRT_PRAGMA(pop)
# else
#   define DDSRT_WARNING_GNUC_OFF(x) \
      DDSRT_PRAGMA(ignored DDSRT_JOINSTR(-W,x))
#   define DDSRT_WARNING_GNUC_ON(x) \
      DDSRT_PRAGMA(warning DDSRT_JOINSTR(-W,x))
# endif
#endif

#if !defined(DDSRT_WARNING_CLANG_OFF) && \
    !defined(DDSRT_WARNING_CLANG_ON)
# define DDSRT_WARNING_CLANG_OFF(x)
# define DDSRT_WARNING_CLANG_ON(x)
#endif

#if !defined(DDSRT_WARNING_GNUC_OFF) && \
    !defined(DDSRT_WARNING_GNUC_ON)
# define DDSRT_WARNING_GNUC_OFF(x)
# define DDSRT_WARNING_GNUC_ON(x)
#endif

#if defined(_MSC_VER)
# define DDSRT_WARNING_MSVC_OFF(x) \
    __pragma (warning(push)) \
    __pragma (warning(disable: ## x))
# define DDSRT_WARNING_MSVC_ON(x) \
    __pragma (warning(pop))
#else
# define DDSRT_WARNING_MSVC_OFF(x)
# define DDSRT_WARNING_MSVC_ON(x)
#endif

/**
 * @brief Macro to disable unused argument warnings
 */
#define DDSRT_UNUSED_ARG(a) (void)(a)

/**
 * @brief Macro to disable warnings for calling deprecated interfaces
 */
#define DDSRT_WARNING_DEPRECATED_OFF \
  DDSRT_WARNING_CLANG_OFF(deprecated-declarations) \
  DDSRT_WARNING_GNUC_OFF(deprecated-declarations) \
  DDSRT_WARNING_MSVC_OFF(4996)

/**
 * @brief Macro to enable warnings for calling deprecated interfaces
 */
#define DDSRT_WARNING_DEPRECATED_ON \
  DDSRT_WARNING_CLANG_ON(deprecated-declarations) \
  DDSRT_WARNING_GNUC_ON(deprecated-declarations) \
  DDSRT_WARNING_MSVC_ON(4996)

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_MISC_H */
