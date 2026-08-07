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
#ifndef DDSRT_ATTRIBUTES_H
#define DDSRT_ATTRIBUTES_H

#if __GNUC__
# define ddsrt_gnuc (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#else
# define ddsrt_gnuc (0)
#endif

#if __clang__
# define ddsrt_clang (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#else
# define ddsrt_clang (0)
#endif

#ifdef __SUNPRO_C
# define __attribute__(x)
#endif

#if defined(__has_attribute)
# define ddsrt_has_attribute(params) __has_attribute(params)
#elif ddsrt_gnuc
# define ddsrt_has_attribute(params) (1) /* GCC < 5 */
#else
# define ddsrt_has_attribute(params) (0)
#endif

#if ddsrt_has_attribute(malloc)
# define ddsrt_attribute_malloc __attribute__ ((__malloc__))
#else
# define ddsrt_attribute_malloc
#endif

#if ddsrt_has_attribute(unused)
# define ddsrt_attribute_unused __attribute__((__unused__))
#else
# define ddsrt_attribute_unused
#endif

#if ddsrt_has_attribute(noreturn)
# define ddsrt_attribute_noreturn __attribute__ ((__noreturn__))
#else
# define ddsrt_attribute_noreturn
#endif

#if ddsrt_has_attribute(nonnull)
# define ddsrt_nonnull(params) __attribute__ ((__nonnull__ params))
# define ddsrt_nonnull_all __attribute__ ((__nonnull__))
#else
# define ddsrt_nonnull(params)
# define ddsrt_nonnull_all
#endif

#if ddsrt_has_attribute(returns_nonnull) && (ddsrt_clang || ddsrt_gnuc >= 40900)
# define ddsrt_attribute_returns_nonnull __attribute__ ((__returns_nonnull__))
#else
# define ddsrt_attribute_returns_nonnull
#endif

/* GCC <= 4.2.4 has the attribute, but warns that it ignores it. */
#if !ddsrt_has_attribute(alloc_size) || (ddsrt_gnuc <= 40204)
# define ddsrt_attribute_alloc_size(params)
#else
# define ddsrt_attribute_alloc_size(params) __attribute__ ((__alloc_size__ params))
#endif

#if ddsrt_has_attribute(const)
# define ddsrt_attribute_const __attribute__ ((__const__))
#else
# define ddsrt_attribute_const
#endif

#if ddsrt_has_attribute(pure)
# define ddsrt_attribute_pure __attribute__ ((__pure__))
#else
# define ddsrt_attribute_pure
#endif

#if ddsrt_has_attribute(format)
# define ddsrt_attribute_format(params) __attribute__ ((__format__ params))
# if __MINGW32__
#   if !defined(__MINGW_PRINTF_FORMAT)
#     define __MINGW_PRINTF_FORMAT gnu_printf
#   endif
    /* GCC assumes printf MS style arguments on Windows */
#   define ddsrt_attribute_format_printf(string_index, first_to_check) \
      ddsrt_attribute_format((__MINGW_PRINTF_FORMAT, string_index, first_to_check))
# else
#   define ddsrt_attribute_format_printf(string_index, first_to_check) \
      ddsrt_attribute_format((printf, string_index, first_to_check))
# endif
#else
# define ddsrt_attribute_format(params)
# define ddsrt_attribute_format_printf(string_index, first_to_check)
#endif

#if ddsrt_has_attribute(warn_unused_result)
# define ddsrt_attribute_warn_unused_result __attribute__ ((__warn_unused_result__))
#else
# define ddsrt_attribute_warn_unused_result
#endif

#if ddsrt_has_attribute(assume_aligned)
# define ddsrt_attribute_assume_aligned(params) __attribute__ ((__assume_aligned__ params))
#else
# define ddsrt_attribute_assume_aligned(params)
#endif

#if ddsrt_has_attribute(packed)
# define ddsrt_attribute_packed __attribute__ ((__packed__))
#else
# define ddsrt_attribute_packed
#endif

#if ddsrt_has_attribute(no_sanitize)
# define ddsrt_attribute_no_sanitize(params) __attribute__ ((__no_sanitize__ params))
#else
# define ddsrt_attribute_no_sanitize(params)
#endif

#if defined(__has_feature)
# define ddsrt_has_feature_thread_sanitizer __has_feature(thread_sanitizer)
#else
# define ddsrt_has_feature_thread_sanitizer 0
#endif

#endif /* DDSRT_ATTRIBUTES_H */
