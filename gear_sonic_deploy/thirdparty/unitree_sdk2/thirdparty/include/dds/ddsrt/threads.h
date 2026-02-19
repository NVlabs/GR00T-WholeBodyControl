/*
 * Copyright(c) 2006 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
/**
 * @file threads.h
 * @brief Thread management and creation.
 *
 * Platform independent interface for managing and creating execution threads.
 */
#ifndef DDSRT_THREADS_H
#define DDSRT_THREADS_H

#include <stdbool.h>
#include <stdint.h>

#include "dds/config.h"
#include "dds/export.h"
#include "dds/ddsrt/attributes.h"
#include "dds/ddsrt/retcode.h"
#include "dds/ddsrt/sched.h"

#if DDSRT_WITH_FREERTOS
#include "dds/ddsrt/threads/freertos.h"
#elif _WIN32
#include "dds/ddsrt/threads/windows.h"
#else
#include "dds/ddsrt/threads/posix.h"
#endif

#if defined (__cplusplus)
extern "C" {
#endif

#if defined(_MSC_VER) || __MINGW__
  /* Thread-local storage using __declspec(thread) on Windows versions before
     Vista and Server 2008 works in DLLs if they are bound to the executable,
     it does not work if the library is loaded using LoadLibrary. */
#define ddsrt_thread_local __declspec(thread)
#elif defined(__GNUC__) || (defined(__clang__) && __clang_major__ >= 2)
  /* GCC supports Thread-local storage for x86 since version 3.3. Clang
     supports Thread-local storage since version 2.0. */
  /* VxWorks 7 supports __thread for both GCC and DIAB, older versions may
     support it as well, but that is not verified. */
#define ddsrt_thread_local __thread
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#define ddsrt_thread_local __thread
#else
#error "Thread-local storage is not supported"
#endif

/**
 * @brief Definition for a thread routine invoked on thread create.
 */
typedef uint32_t (*ddsrt_thread_routine_t)(void*);

/**
 * @brief Definition of the thread attributes
 */
typedef struct {
  /** Specifies the scheduling class */
  ddsrt_sched_t schedClass;
  /** Specifies the thread priority */
  int32_t schedPriority;
  /** Specifies the thread stack size */
  uint32_t stackSize;
} ddsrt_threadattr_t;

/**
 * @brief Initialize thread attributes to platform defaults.
 */
DDS_EXPORT void
ddsrt_threadattr_init(
   ddsrt_threadattr_t *attr)
ddsrt_nonnull_all;

/**
 * @brief Create a new thread.
 *
 * Creates a new thread of control that executes concurrently with
 * the calling thread. The new thread applies the function start_routine
 * passing it arg as first argument.
 *
 * The new thread terminates by returning from the start_routine function.
 * The created thread is identified by the returned threadId.
 *
 * @param[out]  thread         Location where thread id is stored.
 * @param[in]   name           Name assigned to created thread.
 * @param[in]   attr           Attributes to create thread with.
 * @param[in]   start_routine  Function to execute in created thread.
 * @param[in]   arg            Argument passed to @start_routine.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Thread successfully created.
 * @retval DDS_RETCODE_ERROR
 *             Thread could not be created.
 */
DDS_EXPORT dds_return_t
ddsrt_thread_create(
  ddsrt_thread_t *thread,
  const char *name,
  const ddsrt_threadattr_t *attr,
  ddsrt_thread_routine_t start_routine,
  void *arg)
ddsrt_nonnull((1,2,3,4));

/**
 * @brief Retrieve integer representation of the given thread id.
 *
 * @returns The integer representation of the current thread.
 */
DDS_EXPORT ddsrt_tid_t
ddsrt_gettid(void);

/**
 * @brief Retrieve integer representation of the given thread id.
 *
 * @returns The integer representation of the given thread.
 */
DDS_EXPORT ddsrt_tid_t
ddsrt_gettid_for_thread( ddsrt_thread_t thread);


/**
 * @brief Return thread ID of the calling thread.
 *
 * @returns Thread ID of the calling thread.
 */
DDS_EXPORT ddsrt_thread_t
ddsrt_thread_self(void);

/**
 * @brief Compare thread identifiers.
 *
 * @returns true if thread ids match, otherwise false.
 */
DDS_EXPORT bool
ddsrt_thread_equal(ddsrt_thread_t t1, ddsrt_thread_t t2);

/**
 * @brief Wait for termination of the specified thread.
 *
 * If the specified thread is still running, wait for its termination
 * else return immediately. In thread_result it returns the exit status
 * of the thread. If NULL is passed for @thread_result, no exit status is
 * returned, but ddsrt_thread_join still waits for the thread to terminate.
 *
 * @param[in]   thread         Id of thread to wait for.
 * @param[out]  thread_result  Location where thread result is stored.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Target thread terminated.
 * @retval DDS_RETCODE_ERROR
 *             An error occurred while waiting for the thread to terminate.
 */
DDS_EXPORT dds_return_t
ddsrt_thread_join(
  ddsrt_thread_t thread,
  uint32_t *thread_result);

/**
 * @brief Get name of current thread.
 *
 * @param[in]  name  Buffer where the name is copied to.
 * @param[in]  size  Number of bytes available in the buffer.
 *
 * @returns The number of bytes (excluding the null terminating bytes) that
 *          are written. If the buffer is not sufficiently large enough, the
 *          name is truncated and the number of bytes that would have been
 *          written is returned.
 */
DDS_EXPORT size_t
ddsrt_thread_getname(
  char *__restrict name,
  size_t size);

/**
 * @brief Set name of current thread.
 *
 * Set name of the current thread to @name. If the name is longer than the
 * platform maximum, it is silently truncated.
 *
 * @param[in]  name  Name for current thread.
 */
#if DDSRT_HAVE_THREAD_SETNAME
DDS_EXPORT void
ddsrt_thread_setname(
  const char *__restrict name);
#endif

#if DDSRT_HAVE_THREAD_LIST
/**
 * @brief Get a list of threads in the calling process
 *
 * @param[out]  tids    Array of size elements to be filled with thread
 *                      identifiers, may be NULL if size is 0
 * @param[in]   size    The size of the tids array; 0 is allowed
 *
 * @returns A dds_return_t indicating the number of threads in the process
 * or an error code on failure.
 *
 * @retval > 0
 *             Number of threads in the process, may be larger than size
 *             tids[0 .. (return - 1)] are valid
 * @retval DDS_RETCODE_ERROR
 *             Something went wrong, contents of tids is undefined
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Not supported on the platform
 */
DDS_EXPORT dds_return_t ddsrt_thread_list (ddsrt_thread_list_id_t * __restrict tids, size_t size);

/**
 * @brief Get the name of the specified thread (in the calling process)
 *
 * @param[in]   tid     Thread identifier for which the name is sought
 * @param[out]  name    Filled with the thread name (or a synthesized one)
 *                      on successful return; name is silently truncated
 *                      if the actual name is longer than name can hold;
 *                      always 0-terminated if size > 0
 * @param[in]   size    Number of bytes of name that may be assigned, size
 *                      is 0 is allowed, though somewhat useless
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Possibly truncated name is returned as a null-terminated
 *             string in name (provided size > 0).
 * @retval DDS_RETCODE_NOT_FOUND
 *             Thread not found; the contents of name is unchanged
 * @retval DDS_RETCODE_ERROR
 *             Unspecified failure, the contents of name is undefined
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Not supported on the platform
 */
DDS_EXPORT dds_return_t ddsrt_thread_getname_anythread (ddsrt_thread_list_id_t tid, char *__restrict name, size_t size);
#endif

/**
 * @brief Push cleanup handler onto the cleanup stack
 *
 * Push a cleanup handler onto the top of the calling thread's cleanup
 * stack. The cleanup handler will be popped of the thread's cleanup stack
 * and invoked with the specified argument when the thread exits.
 *
 * @param[in]  routine  Cleanup handler to push onto the thread cleanup stack.
 * @param[in]  arg      Argument that will be passed to the cleanup handler.
 */
DDS_EXPORT dds_return_t
ddsrt_thread_cleanup_push(
  void (*routine)(void*),
  void *arg);

/**
 * @brief Pop cleanup handler from the top of the cleanup stack
 *
 * Remove routine at the top of the calling thread's cleanup stack and
 * optionally invoke it (if execute is non-zero).
 */
DDS_EXPORT dds_return_t
ddsrt_thread_cleanup_pop(
  int execute);

/**
 * @brief Initialize thread internals.
 *
 * Initialize internals for threads not created with @ddsrt_create_thread. By
 * default initialization is done automatically.
 */
DDS_EXPORT void
ddsrt_thread_init(uint32_t reason);

/**
 * @brief Free thread resources and execute cleanup handlers.
 *
 * Platforms that support it, automatically free resources claimed by the
 * current thread and call any registered cleanup routines. This function only
 * needs to be called on platforms that do not support thread destructors and
 * only for threads that were not created with @ddsrt_thread_create.
 */
DDS_EXPORT void
ddsrt_thread_fini(uint32_t reason);

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_THREADS_H */
