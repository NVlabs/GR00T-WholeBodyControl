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
#ifndef DDSRT_SYNC_H
#define DDSRT_SYNC_H

#include <stdbool.h>

#include "dds/ddsrt/time.h"
#include "dds/ddsrt/retcode.h"
#include "dds/ddsrt/attributes.h"

#if DDSRT_WITH_FREERTOS
#include "dds/ddsrt/sync/freertos.h"
#elif _WIN32
#include "dds/ddsrt/sync/windows.h"
#elif __SunOS_5_6
#include "dds/ddsrt/sync/solaris2.6.h"
#else
#include "dds/ddsrt/sync/posix.h"
#endif

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @brief Initialize a mutex.
 *
 * @param[in]  mutex  Mutex to itialize.
 */
DDS_EXPORT void
ddsrt_mutex_init(
  ddsrt_mutex_t *mutex)
ddsrt_nonnull_all;

/**
 * @brief Destroy a mutex.
 *
 * @param[in]  mutex  Mutex to destroy.
 */
DDS_EXPORT void
ddsrt_mutex_destroy(
  ddsrt_mutex_t *mutex)
ddsrt_nonnull_all;

/**
 * @brief Acquire a mutex.
 *
 * @param[in]  mutex  Mutex to acquire.
 */
DDS_EXPORT void
ddsrt_mutex_lock(
  ddsrt_mutex_t *mutex)
ddsrt_nonnull_all;

/**
 * @brief Acquire a mutex if it is not already acquired.
 *
 * @param[in]  mutex  Mutex to acquire.
 *
 * @returns true if the mutex was acquired, false otherwise.
 */
DDS_EXPORT bool
ddsrt_mutex_trylock(
  ddsrt_mutex_t *mutex)
ddsrt_nonnull_all
ddsrt_attribute_warn_unused_result;

/**
 * @brief Release an acquired mutex.
 *
 * @param[in]  mutex  Mutex to release.
 */
DDS_EXPORT void
ddsrt_mutex_unlock (
  ddsrt_mutex_t *mutex)
ddsrt_nonnull_all;

/**
 * @brief Initialize a condition variable.
 *
 * @param[in]  cond  Condition variable to initialize.
 */
DDS_EXPORT void
ddsrt_cond_init(
  ddsrt_cond_t *cond)
ddsrt_nonnull_all;

/**
 * @brief Destroy a condition variable.
 *
 * @param[in]  cond  Condition variable to destroy.
 */
DDS_EXPORT void
ddsrt_cond_destroy(
  ddsrt_cond_t *cond)
ddsrt_nonnull_all;

/**
 * @brief Wait for a condition variable to be signalled.
 *
 * @param[in]  cond   Condition variable to block on.
 * @param[in]  mutex  Mutex to associate with condition variable.
 *
 * @pre The calling thread must hold the mutex specified by @mutex.
 *
 * @post The calling thread will hold the mutex specified by @mutex.
 */
DDS_EXPORT void
ddsrt_cond_wait(
  ddsrt_cond_t *cond,
  ddsrt_mutex_t *mutex)
ddsrt_nonnull_all;

/**
 * @brief Wait until @abstime for a condition variable to be signalled.
 *
 * @param[in]  cond     Condition variable to block on.
 * @param[in]  mutex    Mutex to associate with condition variable.
 * @param[in]  abstime  Time in nanoseconds since UNIX Epoch.
 *
 * @pre The calling thread must hold the mutex specified by @mutex.
 *
 * @post The calling thread will hold the mutex specified by @mutex.
 *
 * @returns false if the condition variable was not signalled before the
 *          absolute time specified by @abstime passed, otherwise true.
 */
DDS_EXPORT bool
ddsrt_cond_waituntil(
  ddsrt_cond_t *cond,
  ddsrt_mutex_t *mutex,
  dds_time_t abstime)
ddsrt_nonnull((1,2));

/**
 * @brief Wait for @reltime for a condition variable to be signalled.
 *
 * @param[in]  cond     Condition variable to block on.
 * @param[in]  mutex    Mutex to associate with condition variable.
 * @param[in]  reltime  Time in nanoseconds since UNIX Epoch.
 *
 * @pre The calling thread must hold the mutex specified by @mutex.
 *
 * @post The calling thread will hold the mutex specified by @mutex.
 *
 * @returns false if the condition variable was not signalled before the
 *          relative time specified by @reltime passed, otherwise true.
 */
DDS_EXPORT bool
ddsrt_cond_waitfor(
  ddsrt_cond_t *cond,
  ddsrt_mutex_t *mutex,
  dds_duration_t reltime)
ddsrt_nonnull((1,2));

/**
 * @brief Signal a condition variable and unblock at least one thread.
 *
 * @param[in]  cond  Condition variable to signal.
 *
 * @pre The mutex associated with the condition in general should be acquired
 *      by the calling thread before setting the condition state and
 *      signalling.
 */
DDS_EXPORT void
ddsrt_cond_signal(
  ddsrt_cond_t *cond)
ddsrt_nonnull_all;

/**
 * @brief Signal a condition variable and unblock all threads.
 *
 * @param[in]  cond  Condition variable to signal.
 *
 * @pre The mutex associated with the condition in general should be acquired
 *      by the calling thread before setting the condition state and
 *      signalling
 */
DDS_EXPORT void
ddsrt_cond_broadcast(
  ddsrt_cond_t *cond)
ddsrt_nonnull_all;

/**
 * @brief Initialize a read-write lock.
 *
 * @param[in]  rwlock  Read-write lock to initialize.
 */
DDS_EXPORT void
ddsrt_rwlock_init(
  ddsrt_rwlock_t *rwlock)
ddsrt_nonnull_all;

/**
 * @brief Destroy a read-write lock.
 *
 * @param[in]  rwlock  Read-write lock to destroy.
 */
DDS_EXPORT void
ddsrt_rwlock_destroy(
  ddsrt_rwlock_t *rwlock);

/**
 * @brief Acquire a read-write lock for reading.
 *
 * @param[in]  rwlock  Read-write lock to acquire.
 *
 * @post Data related to the critical section must not be changed by the
 *       calling thread.
 */
DDS_EXPORT void
ddsrt_rwlock_read(
  ddsrt_rwlock_t *rwlock)
ddsrt_nonnull_all;

/**
 * @brief Acquire a read-write lock for writing.
 *
 * @param[in]  rwlock  Read-write lock to acquire.
 */
DDS_EXPORT void
ddsrt_rwlock_write(
  ddsrt_rwlock_t *rwlock)
ddsrt_nonnull_all;

/**
 * @brief Try to acquire a read-write lock for reading.
 *
 * Try to acquire a read-write lock while for reading, immediately return if
 * the lock is already exclusively acquired by another thread.
 *
 * @param[in]  rwlock  Read-write lock to aqcuire.
 *
 * @post Data related to the critical section must not changed by the
 *       calling thread.
 *
 * @returns true if the lock was acquired, otherwise false.
 */
DDS_EXPORT bool
ddsrt_rwlock_tryread(
  ddsrt_rwlock_t *rwlock)
ddsrt_nonnull_all
ddsrt_attribute_warn_unused_result;

/**
 * @brief Try to acquire a read-write lock for writing.
 *
 * Try to acquire a read-write lock for writing, immediately return if the
 * lock is already acquired, either for reading or writing, by another thread.
 *
 * @param[in]  rwlock  Read-write lock to acquire.
 *
 * @returns true if the lock was acquired, otherwise false.
 */
DDS_EXPORT bool
ddsrt_rwlock_trywrite(
  ddsrt_rwlock_t *rwlock)
ddsrt_nonnull_all
ddsrt_attribute_warn_unused_result;

/**
 * @brief Release a previously acquired read-write lock.
 *
 * @param[in]  rwlock  Read-write lock to release.
 */
DDS_EXPORT void
ddsrt_rwlock_unlock(
  ddsrt_rwlock_t *rwlock)
ddsrt_nonnull_all;

/* Initialization callback used by ddsrt_once */
typedef void (*ddsrt_once_fn)(void);

/**
 * @brief Invoke init_fn exactly once for a given control.
 *
 * The first thread to call this function with a given control will call the
 * function specified by @init_fn with no arguments. All following calls with
 * the same control will not call the specified function.
 *
 * @pre The control parameter is properly initialized with DDSRT_ONCE_INIT.
 */
DDS_EXPORT void
ddsrt_once(
  ddsrt_once_t *control,
  ddsrt_once_fn init_fn);

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_SYNC_H */
