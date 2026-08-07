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
/**
 * @file heap.h
 * @brief Heap memory management.
 *
 * Platform independent interface to heap memory management.
 */
#ifndef DDSRT_HEAP_H
#define DDSRT_HEAP_H

#include <stddef.h>

#include "dds/export.h"
#include "dds/ddsrt/attributes.h"

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @brief Allocate memory from heap.
 *
 * The allocated block of memory must be freed by calling @ddsrt_free when no
 * longer used.
 *
 * @param[in]  size  The size, in bytes, of the block of memory to allocate.
 *
 * @returns A pointer to the allocated block of memory. abort() is called if
 *          not enough free memory was available.
 */
DDS_EXPORT void *
ddsrt_malloc(
  size_t size)
ddsrt_attribute_malloc
ddsrt_attribute_alloc_size((1));

/**
 * @brief Allocate memory from heap.
 *
 * Allocate a block of memory from heap with the given size. The allocated
 * block of memory must be freed by calling @ddsrt_free when no longer used.
 *
 * @param[in]  size  The size, in bytes, of memory to allocate.
 *
 * @returns A pointer to the allocated block of memory, NULL if not enough
 *          memory was available.
 */
DDS_EXPORT void *
ddsrt_malloc_s(
  size_t size)
ddsrt_attribute_malloc
ddsrt_attribute_alloc_size((1));

/**
 * @brief Allocate memory from heap for an array of @count elements of @size
 *        bytes.
 *
 * The allocated memory is initialized to zero. The allocated memory must be
 * freed by calling @ddsrt_free when no longer used.
 *
 * A non-NULL pointer, that must be freed is always returned, even if the sum
 * @count and @size equals zero.
 *
 * @returns A pointer to the allocated memory. abort() is called if not enough
 *          free memory was available.
 */
DDS_EXPORT void *
ddsrt_calloc(
  size_t count,
  size_t size)
ddsrt_attribute_malloc
ddsrt_attribute_alloc_size((1,2));

/**
 * @brief Allocate memory from heap for an array of @count elements of @size
 *        bytes.
 *
 * The allocated memory is initialized to zero. The allocated memory must be
 * freed by calling @ddsrt_free when no longer used.
 *
 * A non-NULL pointer, that must be freed is always returned, even if the sum
 * @count and @size equals zero.
 *
 * @returns A pointer to the allocated memory, or NULL if not enough memory was
 *          available.
 */
DDS_EXPORT void *
ddsrt_calloc_s(
  size_t count,
  size_t size)
ddsrt_attribute_malloc
ddsrt_attribute_alloc_size((1,2));

/**
 * @brief Reallocate memory from heap.
 *
 * Reallocate memory from heap. If memblk is NULL the function returns
 * ddsrt_malloc_s(size). If size is 0, ddsrt_realloc_s free's the memory
 * pointed to by memblk and returns a pointer as if ddsrt_malloc_s(0) was
 * invoked. The returned pointer must be free'd with ddsrt_free.
 *
 * @returns A pointer to reallocated memory. Calls abort() if not enough free
 *          memory was available.
 */
DDS_EXPORT void *
ddsrt_realloc(
  void *memblk,
  size_t size)
ddsrt_attribute_malloc
ddsrt_attribute_alloc_size((2));

/**
 * @brief Reallocate memory from heap.
 *
 * Reallocate memory from heap. If memblk is NULL the function returns
 * ddsrt_malloc_s(size). If size is 0, ddsrt_realloc_s free's the memory
 * pointed to by memblk and returns a pointer as if ddsrt_malloc_s(0) was
 * invoked. The returned pointer must be free'd with ddsrt_free.
 *
 * @returns A pointer to reallocated memory, or NULL if not enough free memory
 *          was available.
 */
DDS_EXPORT void *
ddsrt_realloc_s(
  void *memblk,
  size_t size)
ddsrt_attribute_malloc
ddsrt_attribute_alloc_size((2));

/**
 * @brief Free a previously allocated block of memory and return it to heap.
 *
 * Free the allocated memory pointed to by @ptr and release it to the heap. No
 * action will be taken if @ptr is NULL.
 *
 * @param[in]  ptr  Pointer to previously allocated block of memory.
 */
DDS_EXPORT void
ddsrt_free(void *ptr);

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_HEAP_H */
