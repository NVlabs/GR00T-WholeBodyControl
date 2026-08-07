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
/* TODO: do we really need to expose this as an API? */

/** @file
 *
 * @brief DDS C Allocation API
 *
 * This header file defines the public API of allocation convenience functions
 * in the Eclipse Cyclone DDS C language binding.
 */
#ifndef DDS_ALLOC_H
#define DDS_ALLOC_H

#include <stddef.h>

#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct dds_topic_descriptor;
struct dds_sequence;

/**
 * @anchor DDS_FREE_KEY_BIT
 * @ingroup alloc
 * @brief Instruction to free all keyfields in sample
 */
#define DDS_FREE_KEY_BIT 0x01

/**
 * @anchor DDS_FREE_CONTENTS_BIT
 * @ingroup alloc
 * @brief Instruction to free all non-keyfields in sample
 */
#define DDS_FREE_CONTENTS_BIT 0x02

/**
 * @anchor DDS_FREE_ALL_BIT
 * @ingroup alloc
 * @brief Instruction to free outer sample
 */
#define DDS_FREE_ALL_BIT 0x04

/**
 * @brief Freeing operation type
 * @ingroup alloc
 * What part of a sample to free
 */
typedef enum
{
  DDS_FREE_ALL = DDS_FREE_KEY_BIT | DDS_FREE_CONTENTS_BIT | DDS_FREE_ALL_BIT, /**< free full sample */
  DDS_FREE_CONTENTS = DDS_FREE_KEY_BIT | DDS_FREE_CONTENTS_BIT, /**< free all sample contents, but leave sample pointer intact */
  DDS_FREE_KEY = DDS_FREE_KEY_BIT /**< free only the keyfields in a sample */
}
dds_free_op_t;

/**
 * @brief DDS Allocator
 * @ingroup alloc
 * C-Style allocator API
 */
typedef struct dds_allocator
{
  void * (*malloc) (size_t size); /**< behave like C malloc */
  void * (*realloc) (void *ptr, size_t size); /**< behave like C realloc, may be null */
  void (*free) (void *ptr); /**< behave like C free */
}
dds_allocator_t;

/**
 * @brief Perform an alloc() with the default allocator.
 *
 * @param[in] size number of bytes
 * @returns new pointer or NULL if out of memory
 */
DDS_EXPORT void * dds_alloc (size_t size);

/**
 * @brief Perform a realloc() with the default allocator.
 *
 * @param[in] ptr previously alloc()'ed pointer
 * @param[in] size new size
 * @return new pointer or NULL if out of memory
 */
DDS_EXPORT void * dds_realloc (void * ptr, size_t size);

/**
 * @brief Perform a realloc() with the default allocator. Zero out memory.
 *
 * @param[in] ptr previously alloc()'ed pointer
 * @param[in] size new size
 * @return new pointer or NULL if out of memory
 */
DDS_EXPORT void * dds_realloc_zero (void * ptr, size_t size);

/**
 * @brief Perform a free() on a memory fragment allocated with the default allocator.
 *
 * @param[in] ptr previously alloc()'ed pointer
 */
DDS_EXPORT void dds_free (void * ptr);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef void * (*dds_alloc_fn_t) (size_t);
typedef void * (*dds_realloc_fn_t) (void *, size_t);
typedef void (*dds_free_fn_t) (void *);
#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * @brief Allocated a string with size, accounting for the null terminator.
 *
 * @param[in] size number of characters
 * @returns newly allocated string or NULL if out of memory
 */
DDS_EXPORT char * dds_string_alloc (size_t size);

/**
 * @brief Duplicate a null-terminated string
 *
 * @param[in] str string to duplicate
 * @returns newly allocated duplicate string, or NULL if out of memory
 */
DDS_EXPORT char * dds_string_dup (const char * str);

/**
 * @brief Free a string, equivalent to dds_free
 *
 * @param[in] str string to free
 */
DDS_EXPORT void dds_string_free (char * str);

/**
 * @brief Free (parts of) a sample according to the \ref dds_free_op_t
 *
 * @param[in] sample sample to free
 * @param[in] desc topic descriptor of the type this sample was created from.
 * @param[in] op Which parts of the sample to free.
 */
DDS_EXPORT void dds_sample_free (void * sample, const struct dds_topic_descriptor * desc, dds_free_op_t op);

#if defined (__cplusplus)
}
#endif
#endif
