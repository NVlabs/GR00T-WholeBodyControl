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

/** @file
 *
 * @brief DDS C Logging API
 *
 * This header file defines the public API for logging and controlling logging
 * in the DDS C language binding.
 */
#ifndef DDS_LOG_H
#define DDS_LOG_H

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include "dds/export.h"
#include "dds/ddsrt/attributes.h"

#if defined (__cplusplus)
extern "C" {
#endif

/** @defgroup log_categories Convenience log category definitions.
 *
 * These defines expand into numeric values that can be ORed together to
 * specify messages of which categories must be passed to the respective sinks.
 *
 * Every category other than DDS_LC_FATAL, DDS_LC_ERROR, DDS_LC_WARNING and
 * DDS_LC_INFO automatically falls into the trace category.
 *
 * @{
 */
/** Fatal error condition. Immediate abort on sink return. */
#define DDS_LC_FATAL (1u)
/** Error condition. */
#define DDS_LC_ERROR (2u)
/** Warning condition. */
#define DDS_LC_WARNING (4u)
/** Informational message. */
#define DDS_LC_INFO (8u)
/** Debug/trace messages related to configuration settings. */
#define DDS_LC_CONFIG (16u)
/** Debug/trace messages related to node discovery. */
#define DDS_LC_DISCOVERY (32u)
/** Currently unused. */
#define DDS_LC_DATA (64u)
/** Debug/trace messages for which no specialized category exists (yet). */
#define DDS_LC_TRACE (128u)
/** Debug/trace messages related to receive administration. */
#define DDS_LC_RADMIN (256u)
/** Debug/trace messages related to timing. */
#define DDS_LC_TIMING (512u)
/** Debug/trace messages related to send administration. */
#define DDS_LC_TRAFFIC (1024u)
/** Currently unused. */
#define DDS_LC_TOPIC (2048u)
/** Debug/trace messages related to TCP communication. */
#define DDS_LC_TCP (4096u)
/** Debug/trace messages related to parameter list processing. */
#define DDS_LC_PLIST (8192u)
/** Debug/trace messages related to the writer history cache. */
#define DDS_LC_WHC (16384u)
/** Debug/trace messages related to throttling. */
#define DDS_LC_THROTTLE (32768u)
/** Reader history cache. */
#define DDS_LC_RHC (65536u)
/** Include content in traces. */
#define DDS_LC_CONTENT (131072u)
/** Debug/trace messages related to SHMEM */
#define DDS_LC_SHM (262144u)
/** All common trace categories. */
#define DDS_LC_ALL \
    (DDS_LC_FATAL | DDS_LC_ERROR | DDS_LC_WARNING | DDS_LC_INFO | \
     DDS_LC_CONFIG | DDS_LC_DISCOVERY | DDS_LC_DATA | DDS_LC_TRACE | \
     DDS_LC_TIMING | DDS_LC_TRAFFIC | DDS_LC_TCP | DDS_LC_THROTTLE | \
     DDS_LC_CONTENT | DDS_LC_SHM)
/** @}*/

#define DDS_LOG_MASK \
    (DDS_LC_FATAL | DDS_LC_ERROR | DDS_LC_WARNING | DDS_LC_INFO)

#define DDS_TRACE_MASK \
    (~DDS_LOG_MASK)

/** Structure with log message and meta data passed to callbacks. */
typedef struct {
  /** Log category the message falls into. */
  uint32_t priority;
  /** Log domain id, UINT32_MAX is global. */
  uint32_t domid;
  /** Filename where message was generated. */
  const char *file;
  /** Line number in file where message was generated. */
  uint32_t line;
  /** Name of function message where message was generated. */
  const char *function;
  /** Log message. */
  const char *message;
  /** Size of log message. */
  size_t size;
  /** Default log message header length */
  size_t hdrsize;
} dds_log_data_t;

/** Function signature that log and trace callbacks must adhere too. */
typedef void (*dds_log_write_fn_t) (void *, const dds_log_data_t *);

/** Semi-opaque type for log/trace configuration. */
struct ddsrt_log_cfg_common {
  /** Mask for testing whether the xLOG macro should forward to the
      function (and so incur the cost of constructing the parameters).
      Messages in DDS_LOG_MASK are rare, so the overhead of calling
      the function and then dropping the message is not an issue, unlike
      for messages in DDS_TRACE_MASK. */
  uint32_t mask;

  /** The actual configured trace mask */
  uint32_t tracemask;

  /** Domain id for reporting; UINT32_MAX = no domain */
  uint32_t domid;
};

typedef struct ddsrt_log_cfg {
  struct ddsrt_log_cfg_common c;
  union {
    dds_log_write_fn_t fnptr;
    void *ptr;
    uint32_t u32;
    unsigned char pad[72];
  } u;
} ddsrt_log_cfg_t;

DDS_EXPORT extern uint32_t *const dds_log_mask;

/**
 * @brief Get currently enabled log and trace categories.
 *
 * @returns A uint32_t with enabled categories set.
 */
DDS_INLINE_EXPORT inline uint32_t
dds_get_log_mask(void)
{
    return *dds_log_mask;
}

/**
 * @brief Set enabled log and trace categories.
 *
 * @param[in]  cats  Log and trace categories to enable.
 */
DDS_EXPORT void
dds_set_log_mask(
    uint32_t cats);

/**
 * @private
 */
DDS_EXPORT void
dds_set_log_file(
    FILE *file);

/**
 * @private
 */
DDS_EXPORT void
dds_set_trace_file(
    FILE *file);

/**
 * @brief Register callback to receive log messages
 *
 * Callbacks registered to handle log messages will receive messages of type
 * info, warning, error and fatal. Messages that fall into the trace category
 * will never be delivered to the callback.
 *
 * This operation is synchronous and only returns once the operation is
 * registered with all threads. Meaning that neither callback or userdata will
 * be referenced by the DDS stack on return.
 *
 * @param[in]  callback  Function pointer matching dds_log_write_fn signature
 *                       or a null pointer to restore the default sink.
 * @param[in]  userdata  User specified data passed along with each invocation
 *                       of callback.
 */
DDS_EXPORT void
dds_set_log_sink(
    dds_log_write_fn_t callback,
    void *userdata);

/**
 * @brief Register callback to receive trace messages
 *
 * Callbacks registered to handle trace messages will receive messages of type
 * info, warning, error and fatal as well as all message types that fall into
 * the trace category depending on the log mask.
 *
 * This operation is synchronous and only returns once the operation is
 * registered with all threads. Meaning that neither callback or
 * userdata will be referenced by the DDS stack on return.
 *
 * @param[in]  callback  Function pointer matching dds_log_write_fn_t signature
 *                       or a null pointer to restore the default sink.
 * @param[in]  userdata  User specified data passed along with each invocation
 *                       of callback.
 */
DDS_EXPORT void
dds_set_trace_sink(
    dds_log_write_fn_t callback,
    void *userdata);

/**
 * @brief Initialize a struct ddsrt_log_cfg for use with dds_log_cfg
 *
 * Callbacks registered to handle log messages will receive messages of type
 * info, warning, error and fatal. Messages that fall into the trace category
 * will never be delivered to the callback.
 *
 * Callbacks registered to handle trace messages will receive messages of type
 * info, warning, error and fatal as well as all message types that fall into
 * the trace category depending on the log mask.
 *
 * This operation is synchronous and only returns once the operation is
 * registered with all threads. Meaning that neither callback or
 * userdata will be referenced by the DDS stack on return.
 *
 * @param[out] cfg            On return, initialised to make dds_log_cfg invoked
 *                            with this config object behave as specified by the
 *                            other parameters.
 * @param[in]  domid          Numerical identifier in log/trace, UINT32_MAX is
 *                            reserved for global logging.
 * @param[in]  tracemask      Mask determining which traces should be written.
 * @param[in]  log_fp         File for default sink.
 * @param[in]  trace_fp       File for default sink.
 */
DDS_EXPORT void
dds_log_cfg_init(
    struct ddsrt_log_cfg *cfg,
    uint32_t domid,
    uint32_t tracemask,
    FILE *log_fp,
    FILE *trace_fp);

/**
 * @brief Write a log or trace message for a specific logging configuraiton
 * (categories, id, sinks).
 *
 * Direct use of #dds_log is discouraged. Use #DDS_CINFO, #DDS_CWARNING,
 * #DDS_CERROR, #DDS_CTRACE or #DDS_CLOG instead.
 */
DDS_EXPORT void
dds_log_cfg(
    const struct ddsrt_log_cfg *cfg,
    uint32_t prio,
    const char *file,
    uint32_t line,
    const char *func,
    const char *fmt,
    ...)
  ddsrt_attribute_format_printf(6, 7);

/**
 * @brief Write a log or trace message to the global configuration but with
 * specific domain (intended solely for use during domain start-up, while
 * the domain-specific logging/tracing hasn't been set yet).
 *
 * Write a log or trace message to one (or both) of the currently active sinks.
 *
 * Direct use of #dds_log_id is discouraged. Use #DDS_ILOG instead.
 */
DDS_EXPORT void
dds_log_id(
    uint32_t prio,
    uint32_t domid,
    const char *file,
    uint32_t line,
    const char *func,
    const char *fmt,
    ...)
  ddsrt_attribute_format_printf(6, 7);

/**
 * @brief Write a log or trace message to the global log/trace.
 *
 * Write a log or trace message to one (or both) of the currently active sinks.
 *
 * Direct use of #dds_log is discouraged. Use #DDS_INFO, #DDS_WARNING,
 * #DDS_ERROR, #DDS_FATAL or #DDS_LOG instead.
 */
DDS_EXPORT void
dds_log(
    uint32_t prio,
    const char *file,
    uint32_t line,
    const char *func,
    const char *fmt,
    ...)
  ddsrt_attribute_format_printf(5, 6);

/**
 * @brief Undecorated function name of the current function.
 *
 * Behavior of DDS_FUNCTION outside a function is undefined. Note that
 * implementations differ across compilers and compiler versions. It might be
 * implemented as either a string literal or a constant variable.
 */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#   define DDS_FUNCTION __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
#   define DDS_FUNCTION __func__
#elif defined(__GNUC__)
#   define DDS_FUNCTION __FUNCTION__
#elif defined(__clang__)
#   define DDS_FUNCTION __FUNCTION__
#elif defined(__ghs__)
#   define DDS_FUNCTION __FUNCTION__
#elif (defined(__SUNPRO_C) || defined(__SUNPRO_CC))
/* Solaris Studio had support for __func__ before it supported __FUNCTION__.
   Compiler flag -features=extensions is required on older versions. */
#   define DDS_FUNCTION __func__
#elif defined(__FUNCTION__)
/* Visual Studio */
#   define DDS_FUNCTION __FUNCTION__
#elif defined(__vxworks)
/* At least versions 2.9.6 and 3.3.4 of the GNU C Preprocessor only define
   __GNUC__ if the entire GNU C compiler is in use. VxWorks 5.5 targets invoke
   the preprocessor separately resulting in __GNUC__ not being defined. */
#   define DDS_FUNCTION __FUNCTION__
#else
#   warning "DDS_FUNCTION is not supported"
#   define DDS_FUNCTION ""
#endif

/**
 * @brief Function signature of the current function.
 *
 * See comments on DDS_FUNCTION for details.
 */
#if defined(__GNUC__)
#   define DDS_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif defined(__clang__)
#   define DDS_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif defined(__ghs__)
#   define DDS_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif (defined(__SUNPRO_C) && __SUNPRO_C >= 0x5100)
/* Solaris Studio supports __PRETTY_FUNCTION__ in C since version 12.1 */
#   define DDS_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif (defined(__SUNPRO_CC) && __SUNPRO_CC >= 0x5120)
/* Solaris Studio supports __PRETTY_FUNCTION__ in C++ since version 12.3 */
#   define DDS_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
/* Visual Studio */
#   define DDS_PRETTY_FUNCTION __FUNCSIG__
#elif defined(__vxworks)
/* See comments on __vxworks macro above. */
#   define DDS_PRETTY_FUNCTION __PRETTY_FUNCTION__
#else
/* Fall back to DDS_FUNCTION. */
#   define DDS_PRETTY_FUNCTION DDS_FUNCTION
#endif

/**
 * @brief Write a log message.
 *
 * Write a log or trace message to the currently active log and/or trace sinks
 * if the log category is enabled. Whether or not the category is enabled is
 * checked before any dds_log-related activities to save a couple of % CPU.
 *
 * Only messages that fall into one of the log categories are passed onto
 * dds_log. While messages that fall into a trace category could have been
 * passed just as easily, they are rejected so that tracing is kept entirely
 * separate from logging, if only cosmetic.
 */
#define DDS_LOG(cat, ...) \
    ((dds_get_log_mask() & (cat)) ? \
      dds_log((cat), __FILE__, __LINE__, DDS_FUNCTION, __VA_ARGS__) : 0)

/**
 * @brief Write a log message with a domain id override.
 *
 * Write a log or trace message to the currently active log and/or trace sinks
 * if the log category is enabled. Whether or not the category is enabled is
 * checked before any dds_log-related activities to save a couple of % CPU.
 *
 * Only messages that fall into one of the log categories are passed onto
 * dds_log. While messages that fall into a trace category could have been
 * passed just as easily, they are rejected so that tracing is kept entirely
 * separate from logging, if only cosmetic.
 */
#define DDS_ILOG(cat, domid, ...) \
    ((dds_get_log_mask() & (cat)) ? \
      dds_log_id((cat), (domid), __FILE__, __LINE__, DDS_FUNCTION, __VA_ARGS__) : 0)

/**
 * @brief Write a log message using a specific config.
 *
 * Write a log or trace message to the currently active log and/or trace sinks
 * if the log category is enabled. Whether or not the category is enabled is
 * checked before any dds_log-related activities to save a couple of % CPU.
 *
 * Only messages that fall into one of the log categories are passed onto
 * dds_log. While messages that fall into a trace category could have been
 * passed just as easily, they are rejected so that tracing is kept entirely
 * separate from logging, if only cosmetic.
 */
#define DDS_CLOG(cat, cfg, ...) \
    (((cfg)->c.mask & (cat)) ? \
      dds_log_cfg((cfg), (cat), __FILE__, __LINE__, DDS_FUNCTION, __VA_ARGS__) : 0)

/** Write a log message of type #DDS_LC_INFO into global log. */
#define DDS_INFO(...) \
  DDS_LOG(DDS_LC_INFO, __VA_ARGS__)
/** Write a log message of type #DDS_LC_WARNING into global log. */
#define DDS_WARNING(...) \
  DDS_LOG(DDS_LC_WARNING, __VA_ARGS__)
/** Write a log message of type #DDS_LC_ERROR into global log. */
#define DDS_ERROR(...) \
  DDS_LOG(DDS_LC_ERROR, __VA_ARGS__)
/** Write a log message of type #DDS_LC_ERROR into global log and abort. */
#define DDS_FATAL(...) \
  dds_log(DDS_LC_FATAL, __FILE__, __LINE__, DDS_FUNCTION, __VA_ARGS__)

/* MSVC mishandles __VA_ARGS__ while claiming to be conforming -- and even
   if they have a defensible implement, they still differ from every other
   compiler out there.  An extra layer of macro expansion works around it. */
#define DDS_CLOG_MSVC_WORKAROUND(x) x

/** Write a log message of type #DDS_LC_INFO using specific logging config. */
#define DDS_CINFO(...) \
  DDS_CLOG_MSVC_WORKAROUND(DDS_CLOG(DDS_LC_INFO, __VA_ARGS__))
/** Write a log message of type #DDS_LC_WARNING using specific logging config. */
#define DDS_CWARNING(...) \
  DDS_CLOG_MSVC_WORKAROUND(DDS_CLOG(DDS_LC_WARNING, __VA_ARGS__))
/** Write a log message of type #DDS_LC_ERROR using specific logging config. */
#define DDS_CERROR(...) \
  DDS_CLOG_MSVC_WORKAROUND(DDS_CLOG(DDS_LC_ERROR, __VA_ARGS__))
/** Write a #DDS_LC_TRACE message using specific logging config. */
#define DDS_CTRACE(...) \
  DDS_CLOG_MSVC_WORKAROUND(DDS_CLOG(DDS_LC_TRACE, __VA_ARGS__))

#if defined (__cplusplus)
}
#endif

#endif /* DDS_LOG_H */
