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
/* TODO: do we really need to expose all of this as an API? maybe some, but all? */

/** @file
 *
 * @brief DDS C Implementation API
 *
 * This header file defines the public API for all kinds of things in the
 * Eclipse Cyclone DDS C language binding.
 */
#ifndef DDS_IMPL_H
#define DDS_IMPL_H

#include <stdint.h>
#include <stdbool.h>
#include "dds/export.h"
#include "dds/features.h"
#include "dds/ddsrt/align.h"
#include "dds/ddsc/dds_public_alloc.h"
#include "dds/ddsc/dds_opcodes.h"

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @defgroup implementation (Public Implementation Details)
 * @ingroup dds
 * Miscellaneous types and functions that are required to be public, since they are
 * in the output of the IDL compiler, but are not intended for direct use.
 */

/**
 * @ingroup implementation
 * @brief Datastructure of a Sequence type
 * Container for a sequence of bytes. The general model of this type is also used in IDL output,
 * where the uint8_t * _buffer is replaced by the appropriate subtype of what is contained.
 */
typedef struct dds_sequence
{
  uint32_t _maximum; /**< Allocated space in _buffer */
  uint32_t _length; /**< Used space in _buffer */
  uint8_t * _buffer; /**< Sequence of bytes */
  bool _release; /**< Whether a CycloneDDS _free method should free the contained buffer.
                      if you put in your own allocated _buffer set this to false to avoid
                      CycloneDDS calling free() on it. */
}
dds_sequence_t;

/**
 * @ingroup implementation
 * @brief Key Descriptor
 * Used to describe a named key field in a type with the offset from the start of a struct.
 */
typedef struct dds_key_descriptor
{
  const char * m_name; /**< name of keyfield */
  uint32_t m_offset; /**< offset from pointer */
  uint32_t m_idx; /**< m_idx'th key of type */
}
dds_key_descriptor_t;

/**
 * @defgroup topic_definition (Topic Definition)
 * @ingroup implementation
 * Topic definitions are output by the IDL compiler and have an
 * implementation-private definition. The only thing exposed on the
 * API is a pointer to the "dds_topic_descriptor_t" struct type.
 */

/**
 * @ingroup topic_definition
 * @brief Simple sized byte container to hold serialized type info
 * Holds XTypes information (TypeInformation, TypeMapping) for a
 * type
 */
struct dds_type_meta_ser
{
  unsigned char * data;  /**< data pointer */
  uint32_t sz;  /**< data size */
};

/**
 * @anchor DDS_DATA_REPRESENTATION_XCDR1
 * @ingroup topic_definition
 * @brief Data representation XCDR1
 * Type can be represented using XCDR1
 */
#define DDS_DATA_REPRESENTATION_XCDR1    0

/**
 * @anchor DDS_DATA_REPRESENTATION_XML
 * @ingroup topic_definition
 * @brief Data representation XML
 * Type can be represented using XML
 */
#define DDS_DATA_REPRESENTATION_XML      1

/**
 * @anchor DDS_DATA_REPRESENTATION_XCDR2
 * @ingroup topic_definition
 * @brief Data representation XCDR2
 * Type can be represented using XCDR2
 */
#define DDS_DATA_REPRESENTATION_XCDR2    2


/**
 * @anchor DDS_DATA_REPRESENTATION_FLAG_XCDR1
 * @ingroup topic_definition
 * @brief Data representation XCDR1 flag
 * Type can be represented using XCDR1, preshifted
 */
#define DDS_DATA_REPRESENTATION_FLAG_XCDR1  (1u << DDS_DATA_REPRESENTATION_XCDR1)

/**
 * @anchor DDS_DATA_REPRESENTATION_FLAG_XML
 * @ingroup topic_definition
 * @brief Data representation XML flag
 * Type can be represented using XML, preshifted
 */
#define DDS_DATA_REPRESENTATION_FLAG_XML    (1u << DDS_DATA_REPRESENTATION_XML)

/**
 * @anchor DDS_DATA_REPRESENTATION_FLAG_XCDR2
 * @ingroup topic_definition
 * @brief Data representation XCDR2 flag
 * Type can be represented using XCDR2, preshifted
 */
#define DDS_DATA_REPRESENTATION_FLAG_XCDR2  (1u << DDS_DATA_REPRESENTATION_XCDR2)

/**
 * @anchor DDS_DATA_REPRESENTATION_RESTRICT_DEFAULT
 * @ingroup topic_definition
 * @brief Default datarepresentation flag, XCDR1 and XCDR2 flags
 */
#define DDS_DATA_REPRESENTATION_RESTRICT_DEFAULT  (DDS_DATA_REPRESENTATION_FLAG_XCDR1 | DDS_DATA_REPRESENTATION_FLAG_XCDR2)

/**
 * @brief Topic Descriptor
 * @ingroup topic_definition
 * @warning Unstable/Private API
 * Contains all meta information about a type, usually produced by the IDL compiler
 * Since this type is not intended for public consumption it can change without warning.
 */
typedef struct dds_topic_descriptor
{
  const uint32_t m_size;               /**< Size of topic type */
  const uint32_t m_align;              /**< Alignment of topic type */
  const uint32_t m_flagset;            /**< Flags */
  const uint32_t m_nkeys;              /**< Number of keys (can be 0) */
  const char * m_typename;             /**< Type name */
  const dds_key_descriptor_t * m_keys; /**< Key descriptors (NULL iff m_nkeys 0) */
  const uint32_t m_nops;               /**< Number of ops in m_ops */
  const uint32_t * m_ops;              /**< Marshalling meta data */
  const char * m_meta;                 /**< XML topic description meta data */
  struct dds_type_meta_ser type_information;  /**< XCDR2 serialized TypeInformation, only present if flag DDS_TOPIC_XTYPES_METADATA is set */
  struct dds_type_meta_ser type_mapping;      /**< XCDR2 serialized TypeMapping: maps type-id to type object and minimal to complete type id,
                                                   only present if flag DDS_TOPIC_XTYPES_METADATA is set */
  const uint32_t restrict_data_representation; /**< restrictions on the data representations allowed for the top-level type for this topic,
                                           only present if flag DDS_TOPIC_RESTRICT_DATA_REPRESENTATION */
}
dds_topic_descriptor_t;

/**
 * @defgroup reading_masks (Reading Masks)
 * @ingroup conditions
 * Masks for read condition, read, take: there is only one mask here,
 * which combines the sample, view and instance states.
*/

/**
 * @anchor DDS_READ_SAMPLE_STATE
 * @ingroup reading_masks
 * @brief Samples that were already returned once by a read/take operation
 */
#define DDS_READ_SAMPLE_STATE 1u

/**
 * @anchor DDS_NOT_READ_SAMPLE_STATE
 * @ingroup reading_masks
 * @brief Samples that have not been returned by a read/take operation yet
 */
#define DDS_NOT_READ_SAMPLE_STATE 2u

/**
 * @anchor DDS_ANY_SAMPLE_STATE
 * @ingroup reading_masks
 * @brief Samples \ref DDS_READ_SAMPLE_STATE or \ref DDS_NOT_READ_SAMPLE_STATE
 */
#define DDS_ANY_SAMPLE_STATE (1u | 2u)


/**
 * @anchor DDS_NEW_VIEW_STATE
 * @ingroup reading_masks
 * @brief Samples that belong to a new instance (unique key value)
 */
#define DDS_NEW_VIEW_STATE 4u

/**
 * @anchor DDS_NOT_NEW_VIEW_STATE
 * @ingroup reading_masks
 * @brief Samples that belong to an existing instance (previously received key value)
 */
#define DDS_NOT_NEW_VIEW_STATE 8u

/**
 * @anchor DDS_ANY_VIEW_STATE
 * @ingroup reading_masks
 * @brief Samples \ref DDS_NEW_VIEW_STATE or \ref DDS_NOT_NEW_VIEW_STATE
 */
#define DDS_ANY_VIEW_STATE (4u | 8u)


/**
 * @anchor DDS_ALIVE_INSTANCE_STATE
 * @ingroup reading_masks
 * @brief Samples that belong to a write
 */
#define DDS_ALIVE_INSTANCE_STATE 16u

/**
 * @anchor DDS_NOT_ALIVE_DISPOSED_INSTANCE_STATE
 * @ingroup reading_masks
 * @brief Samples that belong to a (write)dispose
 */
#define DDS_NOT_ALIVE_DISPOSED_INSTANCE_STATE 32u

/**
 * @anchor DDS_NOT_ALIVE_NO_WRITERS_INSTANCE_STATE
 * @ingroup reading_masks
 * @brief Samples that belong a writer that is gone
 */
#define DDS_NOT_ALIVE_NO_WRITERS_INSTANCE_STATE 64u

/**
 * @anchor DDS_ANY_INSTANCE_STATE
 * @ingroup reading_masks
 * @brief Samples \ref DDS_ALIVE_INSTANCE_STATE, \ref DDS_NOT_ALIVE_DISPOSED_INSTANCE_STATE or \ref DDS_NOT_ALIVE_NO_WRITERS_INSTANCE_STATE
 */
#define DDS_ANY_INSTANCE_STATE (16u | 32u | 64u)

/**
 * @anchor DDS_ANY_STATE
 * @ingroup reading_masks
 * @brief Any and all samples
 * Equivalen to \ref DDS_ANY_SAMPLE_STATE | \ref DDS_ANY_VIEW_STATE | \ref DDS_ANY_INSTANCE_STATE
 */
#define DDS_ANY_STATE (DDS_ANY_SAMPLE_STATE | DDS_ANY_VIEW_STATE | DDS_ANY_INSTANCE_STATE)

/**
 * @anchor DDS_DOMAIN_DEFAULT
 * @ingroup domain
 * @brief Select the default domain
 */
#define DDS_DOMAIN_DEFAULT ((uint32_t) 0xffffffffu)

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#define DDS_HANDLE_NIL 0
#define DDS_ENTITY_NIL 0
#endif

/**
 * @brief DDS Entity Kind constants
 * @ingroup internal
 * @warning Unstable/Private API
 * Used throughout the library to indicate what entity is what.
 */
typedef enum dds_entity_kind
{
  DDS_KIND_DONTCARE, /**< Retrieving any entity */
  DDS_KIND_TOPIC, /**< Topic entity */
  DDS_KIND_PARTICIPANT, /**< Domain Participant entity */
  DDS_KIND_READER, /**< Reader entity */
  DDS_KIND_WRITER, /**< Writer entity */
  DDS_KIND_SUBSCRIBER, /**< Subscriber entity */
  DDS_KIND_PUBLISHER, /**< Publisher entity */
  DDS_KIND_COND_READ, /**< ReadCondition entity */
  DDS_KIND_COND_QUERY, /**< QueryCondition entity */
  DDS_KIND_COND_GUARD, /**< GuardCondition entity */
  DDS_KIND_WAITSET, /**< WaitSet entity */
  DDS_KIND_DOMAIN, /**< Domain entity */
  DDS_KIND_CYCLONEDDS /**< CycloneDDS library entity */
} dds_entity_kind_t;
/**
 * @anchor DDS_KIND_MAX
 * @ingroup internal
 * @brief Max entity kind, used for loops.
 */
#define DDS_KIND_MAX DDS_KIND_CYCLONEDDS

/**
 * @ingroup internal
 * @warning Private API
 * @brief Instance handles are uint64_t behind the scenes
 */
typedef uint64_t dds_instance_handle_t;

/**
 * @ingroup domain
 * @brief Domain IDs are 32 bit unsigned integers.
 */
typedef uint32_t dds_domainid_t;

/**
 * @ingroup topic
 * @brief Scope for dds_find_topic()
 */
typedef enum dds_find_scope
{
  DDS_FIND_SCOPE_GLOBAL, /**< locate the topic anywhere CycloneDDS knows about */
  DDS_FIND_SCOPE_LOCAL_DOMAIN, /**< locate the topic locally within domain boundaries */
  DDS_FIND_SCOPE_PARTICIPANT /**< locate the topic within the current participant */
}
dds_find_scope_t;

/**
 * @ingroup builtintopic
 * @brief Type identifier kind for getting endpoint type identifier
 */
typedef enum dds_typeid_kind
{
  DDS_TYPEID_MINIMAL, /**< XTypes Minimal Type ID */
  DDS_TYPEID_COMPLETE /**< XTypes Complete Type ID */
}
dds_typeid_kind_t;

/**
 * @brief Enable or disable write batching.
 * Overrides default configuration setting for write batching (Internal/WriteBatch).
 *
 * @param[in] enable Enables or disables write batching for all writers.
 */
DDS_EXPORT void dds_write_set_batch (bool enable);

#if defined (__cplusplus)
}
#endif
#endif
