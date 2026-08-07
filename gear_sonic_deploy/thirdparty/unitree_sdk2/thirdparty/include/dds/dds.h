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
#ifndef DDS_H
#define DDS_H

/**
 * @file
 * @brief Eclipse Cyclone DDS C header
 * Main header of the Cyclone DDS C library, containing everything you need
 * for your DDS application.
 */

/**
 * @defgroup dds (DDS Functionality)
 */
/**
 * @defgroup deprecated (Deprecated functionality)
 */

#if defined (__cplusplus)
#define restrict
#endif

#include "dds/export.h"
#include "dds/features.h"

#include "dds/ddsc/dds_basic_types.h"

/* Sub components */

#include "dds/ddsrt/time.h"
#include "dds/ddsrt/retcode.h"
#include "dds/ddsrt/log.h"
#include "dds/ddsc/dds_public_impl.h"
#include "dds/ddsc/dds_public_alloc.h"
#include "dds/ddsc/dds_public_qos.h"
#include "dds/ddsc/dds_public_error.h"
#include "dds/ddsc/dds_public_status.h"
#include "dds/ddsc/dds_public_listener.h"

#if defined (__cplusplus)
extern "C" {
#endif


/**
 * @brief DDS Type Identifier (XTypes)
 * @ingroup dds
 * DOC_TODO
 */
typedef struct ddsi_typeid dds_typeid_t;

/**
 * @brief DDS Type Information (XTypes)
 * @ingroup dds
 * DOC_TODO
 */
typedef struct ddsi_typeinfo dds_typeinfo_t;

/**
 * @brief DDS Type Object (XTypes)
 * @ingroup dds
 * DOC_TODO
 */
typedef struct ddsi_typeobj dds_typeobj_t;

/**
 * @brief Reader History Cache
 * @ingroup dds
 * DOC_TODO
 */
struct dds_rhc;

/**
 * @brief DDSI parameter list
 * @ingroup dds
 * DOC_TODO
 */
struct ddsi_plist;

/**
 * @anchor ddsi_sertype
 * @brief DDSI sertype
 * @ingroup dds
 * DOC_TODO
 */
struct ddsi_sertype;

/**
 * @anchor ddsi_serdata
 * @brief DDSI Serdata
 * @ingroup dds
 * DOC_TODO
 */
struct ddsi_serdata;

/**
 * @ingroup deprecated
 * @warning The DDSI sertopic functionality was moved to ddsi_sertype.
 * It has been retained as a symbol to ensure binary compatibility.
 */
struct ddsi_sertopic;

/**
 * @brief DDSI Config
 * @ingroup dds
 * DOC_TODO
 */
struct ddsi_config;

/**
 * @brief Indicates that the library uses ddsi_sertype instead of ddsi_sertopic
 * @ingroup dds
 * If sertype is used, the function dds_create_topic_sertype requires a topic name parameter,
 * as this field is not included in ddsi_sertype.
 */
#define DDS_HAS_DDSI_SERTYPE 1

/**
 * @defgroup builtintopic (Builtin Topic Support)
 * @ingroup dds
 */
/**
 * @defgroup builtintopic_constants (Constants)
 * @ingroup builtintopic
 * @brief Convenience constants for referring to builtin topics
 * These constants can be used in place of an actual dds_topic_t, when creating
 * readers or writers for builtin-topics.
 */
/**
 * @def DDS_BUILTIN_TOPIC_DCPSPARTICIPANT
 * @ingroup builtintopic_constants
 * Pseudo dds_topic_t for the builtin topic DcpsParticipant. Samples from this topic are
 * @ref dds_builtintopic_participant structs.
 */
/**
 * @def DDS_BUILTIN_TOPIC_DCPSTOPIC
 * @ingroup builtintopic_constants
 * Pseudo dds_topic_t for the builtin topic DcpsTopic. Samples from this topic are
 * @ref dds_builtintopic_topic structs. Note that this only works if you have specified
 * ENABLE_TOPIC_DISCOVERY in your cmake build.
 */
/**
 * @def DDS_BUILTIN_TOPIC_DCPSPUBLICATION
 * @ingroup builtintopic_constants
 * Pseudo dds_topic_t for the builtin topic DcpsPublication. Samples from this topic are
 * @ref dds_builtintopic_endpoint structs.
 */
/**
 * @def DDS_BUILTIN_TOPIC_DCPSSUBSCRIPTION
 * @ingroup builtintopic_constants
 * Pseudo dds_topic_t for the builtin topic DcpsSubscription. Samples from this topic are
 * @ref dds_builtintopic_endpoint structs.
 */

#define DDS_BUILTIN_TOPIC_DCPSPARTICIPANT  ((dds_entity_t) (DDS_MIN_PSEUDO_HANDLE + 1))
#define DDS_BUILTIN_TOPIC_DCPSTOPIC        ((dds_entity_t) (DDS_MIN_PSEUDO_HANDLE + 2))
#define DDS_BUILTIN_TOPIC_DCPSPUBLICATION  ((dds_entity_t) (DDS_MIN_PSEUDO_HANDLE + 3))
#define DDS_BUILTIN_TOPIC_DCPSSUBSCRIPTION ((dds_entity_t) (DDS_MIN_PSEUDO_HANDLE + 4))


/**
 * @ingroup DOC_TODO
 * Special handle representing the entity which forces the dds_data_allocator to allocate on heap
 */
#define DDS_DATA_ALLOCATOR_ALLOC_ON_HEAP   ((dds_entity_t) (DDS_MIN_PSEUDO_HANDLE + 257))

/**
 * @defgroup entity_status (Entity Status)
 * @ingroup entity
 * All entities have a set of "status conditions"
 * (following the DCPS spec), read peeks, take reads & resets (analogously to read &
 * take operations on reader). The "mask" allows operating only on a subset of the statuses.
 * Enabled status analogously to DCPS spec.
 * @{
 */
/**
 * @brief These identifiers are used to generate the bitshifted identifiers.
 * By using bitflags instead of these IDs the process of building status masks is
 * simplified to using simple binary OR operations.
 * DOC_TODO fix the refs
 */
typedef enum dds_status_id {
  DDS_INCONSISTENT_TOPIC_STATUS_ID,         /**< See @ref DDS_INCONSISTENT_TOPIC_STATUS */
  DDS_OFFERED_DEADLINE_MISSED_STATUS_ID,    /**< See @ref DDS_OFFERED_DEADLINE_MISSED_STATUS */
  DDS_REQUESTED_DEADLINE_MISSED_STATUS_ID,  /**< See @ref DDS_REQUESTED_DEADLINE_MISSED_STATUS */
  DDS_OFFERED_INCOMPATIBLE_QOS_STATUS_ID,   /**< See @ref DDS_OFFERED_INCOMPATIBLE_QOS_STATUS */
  DDS_REQUESTED_INCOMPATIBLE_QOS_STATUS_ID, /**< See @ref DDS_REQUESTED_INCOMPATIBLE_QOS_STATUS */
  DDS_SAMPLE_LOST_STATUS_ID,                /**< See @ref DDS_SAMPLE_LOST_STATUS */
  DDS_SAMPLE_REJECTED_STATUS_ID,            /**< See @ref DDS_SAMPLE_REJECTED_STATUS */
  DDS_DATA_ON_READERS_STATUS_ID,            /**< See @ref DDS_DATA_ON_READERS_STATUS */
  DDS_DATA_AVAILABLE_STATUS_ID,             /**< See @ref DDS_DATA_AVAILABLE_STATUS */
  DDS_LIVELINESS_LOST_STATUS_ID,            /**< See @ref DDS_LIVELINESS_LOST_STATUS */
  DDS_LIVELINESS_CHANGED_STATUS_ID,         /**< See @ref DDS_LIVELINESS_CHANGED_STATUS */
  DDS_PUBLICATION_MATCHED_STATUS_ID,        /**< See @ref DDS_PUBLICATION_MATCHED_STATUS */
  DDS_SUBSCRIPTION_MATCHED_STATUS_ID        /**< See @ref DDS_SUBSCRIPTION_MATCHED_STATUS */
} dds_status_id_t;

/** Helper value to indicate the highest bit that can be set in a status mask. */
#define DDS_STATUS_ID_MAX (DDS_SUBSCRIPTION_MATCHED_STATUS_ID)

/**
 * @anchor DDS_INCONSISTENT_TOPIC_STATUS
 * Another topic exists with the same name but with different characteristics.
 */
#define DDS_INCONSISTENT_TOPIC_STATUS          (1u << DDS_INCONSISTENT_TOPIC_STATUS_ID)
/**
 * @anchor DDS_OFFERED_DEADLINE_MISSED_STATUS
 * The deadline that the writer has committed through its deadline QoS policy was not respected for a specific instance. */
#define DDS_OFFERED_DEADLINE_MISSED_STATUS     (1u << DDS_OFFERED_DEADLINE_MISSED_STATUS_ID)
/**
 * @anchor DDS_REQUESTED_DEADLINE_MISSED_STATUS
 * The deadline that the reader was expecting through its deadline QoS policy was not respected for a specific instance. */
#define DDS_REQUESTED_DEADLINE_MISSED_STATUS   (1u << DDS_REQUESTED_DEADLINE_MISSED_STATUS_ID)
/**
 * @anchor DDS_OFFERED_INCOMPATIBLE_QOS_STATUS
 * A QoS policy setting was incompatible with what was requested. */
#define DDS_OFFERED_INCOMPATIBLE_QOS_STATUS    (1u << DDS_OFFERED_INCOMPATIBLE_QOS_STATUS_ID)
/**
 * @anchor DDS_REQUESTED_INCOMPATIBLE_QOS_STATUS
 * A QoS policy setting was incompatible with what is offered. */
#define DDS_REQUESTED_INCOMPATIBLE_QOS_STATUS  (1u << DDS_REQUESTED_INCOMPATIBLE_QOS_STATUS_ID)
/**
 * @anchor DDS_SAMPLE_LOST_STATUS
 * A sample has been lost (never received). */
#define DDS_SAMPLE_LOST_STATUS                 (1u << DDS_SAMPLE_LOST_STATUS_ID)
/**
 * @anchor DDS_SAMPLE_REJECTED_STATUS
 * A (received) sample has been rejected. */
#define DDS_SAMPLE_REJECTED_STATUS             (1u << DDS_SAMPLE_REJECTED_STATUS_ID)
/**
 * @anchor DDS_DATA_ON_READERS_STATUS
 * New information is available in some of the data readers of a subscriber. */
#define DDS_DATA_ON_READERS_STATUS             (1u << DDS_DATA_ON_READERS_STATUS_ID)
/**
 * @anchor DDS_DATA_AVAILABLE_STATUS
 * New information is available in a data reader. */
#define DDS_DATA_AVAILABLE_STATUS              (1u << DDS_DATA_AVAILABLE_STATUS_ID)
/**
 * @anchor DDS_LIVELINESS_LOST_STATUS
 * The liveliness that the DDS_DataWriter has committed through its liveliness QoS policy was not respected; thus readers will consider the writer as no longer "alive". */
#define DDS_LIVELINESS_LOST_STATUS             (1u << DDS_LIVELINESS_LOST_STATUS_ID)
/**
 * @anchor DDS_LIVELINESS_CHANGED_STATUS
 * The liveliness of one or more writers, that were writing instances read through the readers has changed. Some writers have become "alive" or "not alive". */
#define DDS_LIVELINESS_CHANGED_STATUS          (1u << DDS_LIVELINESS_CHANGED_STATUS_ID)
/**
 * @anchor DDS_PUBLICATION_MATCHED_STATUS
 * The writer has found a reader that matches the topic and has a compatible QoS. */
#define DDS_PUBLICATION_MATCHED_STATUS         (1u << DDS_PUBLICATION_MATCHED_STATUS_ID)
/**
 * @anchor DDS_SUBSCRIPTION_MATCHED_STATUS
 * The reader has found a writer that matches the topic and has a compatible QoS. */
#define DDS_SUBSCRIPTION_MATCHED_STATUS        (1u << DDS_SUBSCRIPTION_MATCHED_STATUS_ID)
/** @}*/ // end group entity_status

/**
 * @defgroup subscription (Subscription)
 * @ingroup dds
 * DOC_TODO This contains the definitions regarding subscribing to data.
 */

/**
 * @defgroup subdata (Data access)
 * @ingroup subscription
 * Every sample you read from DDS comes with some metadata, which you can inspect and filter on.
 * @{
 */

/** Read state for a data value */
typedef enum dds_sample_state
{
  DDS_SST_READ = DDS_READ_SAMPLE_STATE, /**<DataReader has already accessed the sample by read */
  DDS_SST_NOT_READ = DDS_NOT_READ_SAMPLE_STATE /**<DataReader has not accessed the sample before */
}
dds_sample_state_t;

/** View state of an instance relative to the samples */
typedef enum dds_view_state
{
  /** DataReader is accessing the sample for the first time when the instance is alive */
  DDS_VST_NEW = DDS_NEW_VIEW_STATE,
  /** DataReader accessed the sample before */
  DDS_VST_OLD = DDS_NOT_NEW_VIEW_STATE
}
dds_view_state_t;

/** Defines the state of the instance */
typedef enum dds_instance_state
{
  /** Samples received for the instance from the live data writers */
  DDS_IST_ALIVE = DDS_ALIVE_INSTANCE_STATE,
  /** Instance was explicitly disposed by the data writer */
  DDS_IST_NOT_ALIVE_DISPOSED = DDS_NOT_ALIVE_DISPOSED_INSTANCE_STATE,
  /** Instance has been declared as not alive by data reader as there are no live data writers writing that instance */
  DDS_IST_NOT_ALIVE_NO_WRITERS = DDS_NOT_ALIVE_NO_WRITERS_INSTANCE_STATE
}
dds_instance_state_t;

/** Contains information about the associated data value */
typedef struct dds_sample_info
{
  /** Sample state */
  dds_sample_state_t sample_state;
  /** View state */
  dds_view_state_t view_state;
  /** Instance state */
  dds_instance_state_t instance_state;
  /** Indicates whether there is a data associated with a sample
   *  - true, indicates the data is valid
   *  - false, indicates the data is invalid, no data to read */
  bool valid_data;
  /** timestamp of a data instance when it is written */
  dds_time_t source_timestamp;
  /** handle to the data instance */
  dds_instance_handle_t instance_handle;
  /** handle to the publisher */
  dds_instance_handle_t publication_handle;
  /** count of instance state change from NOT_ALIVE_DISPOSED to ALIVE */
  uint32_t disposed_generation_count;
  /** count of instance state change from NOT_ALIVE_NO_WRITERS to ALIVE */
  uint32_t no_writers_generation_count;
  /** indicates the number of samples of the same instance that follow the current one in the collection */
  uint32_t sample_rank;
  /** difference in generations between the sample and most recent sample of the same instance that appears in the returned collection */
  uint32_t generation_rank;
  /** difference in generations between the sample and most recent sample of the same instance when read/take was called */
  uint32_t absolute_generation_rank;
}
dds_sample_info_t;

/** @}*/ // end group subdata

/**
 * @brief Structure of a GUID in any builtin topic sample.
 * @ingroup builtintopic
 */
typedef struct dds_builtintopic_guid
{
  uint8_t v[16]; /**< 16-byte unique identifier */
}
dds_builtintopic_guid_t;

/**
 * @brief Structure of a GUID in any builtin topic sample.
 * @ingroup builtintopic
 * @ref dds_builtintopic_guid_t is a bit of a weird name for what everyone just calls a GUID,
 * so let us try and switch to using the more logical one.
 */
typedef struct dds_builtintopic_guid dds_guid_t;

/**
 * @brief Sample structure of the Builtin topic DcpsParticipant.
 * @ingroup builtintopic
 */
typedef struct dds_builtintopic_participant
{
  dds_guid_t key; /**< The GUID that uniquely identifies the participant on the network */
  dds_qos_t *qos; /**< The QoS of the participant */
}
dds_builtintopic_participant_t;

/**
 * @brief Structure of a key in the Builtin topic DcpsTopic.
 * @ingroup builtintopic
 */
typedef struct dds_builtintopic_topic_key {
  unsigned char d[16]; /**< 16-byte unique identifier */
} dds_builtintopic_topic_key_t;

/**
 * @brief Sample structure of the Builtin topic DcpsTopic.
 * @ingroup builtintopic
 */
typedef struct dds_builtintopic_topic
{
  dds_builtintopic_topic_key_t key; /**< The GUID that uniquely identifies the topic on the network */
  char *topic_name; /**< The name of the topic, potentially unicode. */
  char *type_name; /**< The name of the type, potentially unicode. */
  dds_qos_t *qos; /**< The QoS of the topic */
}
dds_builtintopic_topic_t;

/**
 * @brief Sample structure of the Builtin topic DcpsPublication and DcpsSubscription.
 * @ingroup builtintopic
 */
typedef struct dds_builtintopic_endpoint
{
  dds_guid_t key; /**< The GUID that uniquely identifies the endpoint on the network */
  dds_guid_t participant_key; /**< The GUID of the participant this endpoint belongs to. */
  dds_instance_handle_t participant_instance_handle; /**< The instance handle the participant assigned to this enpoint. */
  char *topic_name; /**< The name of the topic, potentially unicode. */
  char *type_name; /**< The name of the type, potentially unicode. */
  dds_qos_t *qos; /**< The QoS of the endpoint */
}
dds_builtintopic_endpoint_t;

/**
 * @defgroup entity (Entities)
 * @ingroup dds
 * @brief Every DDS object in the library is an Entity.
 * All entities are represented by a process-private handle, with one
 * call to enable an entity when it was created disabled.
 * An entity is created enabled by default.
 * Note: disabled creation is currently not supported.
*/

/**
 * @brief Enable entity.
 * @ingroup entity
 *
 * @note Delayed entity enabling is not supported yet (CHAM-96).
 *
 * This operation enables the dds_entity_t. Created dds_entity_t objects can start in
 * either an enabled or disabled state. This is controlled by the value of the
 * entityfactory policy on the corresponding parent entity for the given
 * entity. Enabled entities are immediately activated at creation time meaning
 * all their immutable QoS settings can no longer be changed. Disabled Entities are not
 * yet activated, so it is still possible to change their immutable QoS settings. However,
 * once activated the immutable QoS settings can no longer be changed.
 * Creating disabled entities can make sense when the creator of the DDS_Entity
 * does not yet know which QoS settings to apply, thus allowing another piece of code
 * to set the QoS later on.
 *
 * The default setting of DDS_EntityFactoryQosPolicy is such that, by default,
 * entities are created in an enabled state so that it is not necessary to explicitly call
 * dds_enable on newly-created entities.
 *
 * The dds_enable operation produces the same results no matter how
 * many times it is performed. Calling dds_enable on an already
 * enabled DDS_Entity returns DDS_RETCODE_OK and has no effect.
 *
 * If an Entity has not yet been enabled, the only operations that can be invoked
 * on it are: the ones to set, get or copy the QosPolicy settings, the ones that set
 * (or get) the Listener, the ones that get the Status and the dds_get_status_changes
 * operation (although the status of a disabled entity never changes). Other operations
 * will return the error DDS_RETCODE_NOT_ENABLED.
 *
 * Entities created with a parent that is disabled, are created disabled regardless of
 * the setting of the entityfactory policy.
 *
 * If the entityfactory policy has autoenable_created_entities
 * set to TRUE, the dds_enable operation on the parent will
 * automatically enable all child entities created with the parent.
 *
 * The Listeners associated with an Entity are not called until the
 * Entity is enabled. Conditions associated with an Entity that
 * is not enabled are "inactive", that is, have a trigger_value which is FALSE.
 *
 * @param[in]  entity  The entity to enable.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The listeners of to the entity have been successfully been copied
 *             into the specified listener parameter.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The parent of the given Entity is not enabled.
 */
DDS_EXPORT dds_return_t
dds_enable(dds_entity_t entity);

/*
  All entities are represented by a process-private handle, with one
  call to delete an entity and all entities it logically contains.
  That is, it is equivalent to combination of
  delete_contained_entities and delete_xxx in the DCPS API.
*/

/**
 * @brief Delete given entity.
 * @ingroup entity
 *
 * This operation will delete the given entity. It will also automatically
 * delete all its children, childrens' children, etc entities.
 *
 * @param[in]  entity  Entity to delete.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The entity and its children (recursive are deleted).
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
/* TODO: Link to generic dds entity relations documentation. */
DDS_EXPORT dds_return_t
dds_delete(dds_entity_t entity);

/**
 * @brief Get entity publisher.
 * @ingroup entity
 *
 * This operation returns the publisher to which the given entity belongs.
 * For instance, it will return the Publisher that was used when
 * creating a DataWriter (when that DataWriter was provided here).
 *
 * @param[in]  writer  Entity from which to get its publisher.
 *
 * @returns A valid entity or an error code.
 *
 * @retval >0
 *             A valid publisher handle.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_entity_t
dds_get_publisher(dds_entity_t writer);

/**
 * @brief Get entity subscriber.
 * @ingroup entity
 *
 * This operation returns the subscriber to which the given entity belongs.
 * For instance, it will return the Subscriber that was used when
 * creating a DataReader (when that DataReader was provided here).
 *
 * @param[in]  entity  Entity from which to get its subscriber.
 *
 * @returns A valid subscriber handle or an error code.
 *
 * @retval >0
 *             A valid subscriber handle.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * DOC_TODO: Link to generic dds entity relations documentation.
 */
DDS_EXPORT dds_entity_t
dds_get_subscriber(dds_entity_t entity);

/**
 * @brief Get entity datareader.
 * @ingroup entity
 *
 * This operation returns the datareader to which the given entity belongs.
 * For instance, it will return the DataReader that was used when
 * creating a ReadCondition (when that ReadCondition was provided here).
 *
 * @param[in]  entity  Entity from which to get its datareader.
 *
 * @returns A valid reader handle or an error code.
 *
 * @retval >0
 *             A valid reader handle.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * DOC_TODO: Link to generic dds entity relations documentation.
 */
DDS_EXPORT dds_entity_t
dds_get_datareader(dds_entity_t entity);

/**
 * @defgroup condition (Conditions)
 * @ingroup dds
 * @brief Conditions allow you to express conditional interest in samples,
 * to be used in read/take operations or attach to Waitsets.
 */

/**
 * @brief Get the mask of a condition.
 * @ingroup condition
 *
 * This operation returns the mask that was used to create the given
 * condition.
 *
 * @param[in]  condition  Read or Query condition that has a mask.
 * @param[out] mask       Where to store the mask of the condition.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Success (given mask is set).
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The mask arg is NULL.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_mask(dds_entity_t condition, uint32_t *mask);

/**
 * @brief Returns the instance handle that represents the entity.
 * @ingroup entity
 *
 * @param[in]   entity  Entity of which to get the instance handle.
 * @param[out]  ihdl    Pointer to dds_instance_handle_t.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Success.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * DOC_TODO: Check list of return codes is complete.
 * */
DDS_EXPORT dds_return_t
dds_get_instance_handle(dds_entity_t entity, dds_instance_handle_t *ihdl);

/**
 * @brief Returns the GUID that represents the entity in the network,
 * and therefore only supports participants, readers and writers.
 * @ingroup entity
 *
 * @param[in]   entity  Entity of which to get the instance handle.
 * @param[out]  guid    Where to store the GUID.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Success.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 *
 * DOC_TODO: Check list of return codes is complete.
 */
DDS_EXPORT dds_return_t
dds_get_guid (dds_entity_t entity, dds_guid_t *guid);

/**
 * @brief Read the status set for the entity
 * @ingroup entity_status
 *
 * This operation reads the status(es) set for the entity based on
 * the enabled status and mask set. It does not clear the read status(es).
 *
 * @param[in]  entity  Entity on which the status has to be read.
 * @param[out] status  Returns the status set on the entity, based on the enabled status.
 * @param[in]  mask    Filter the status condition to be read, 0 means all statuses
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Success.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter, status is a null pointer or
 *             mask has bits set outside the status range.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object or mask has status
 *             bits set that are undefined for the type of entity.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_read_status(dds_entity_t entity, uint32_t *status, uint32_t mask);

/**
 * @brief Read the status set for the entity
 * @ingroup entity_status
 *
 * This operation reads the status(es) set for the entity based on the enabled
 * status and mask set. It clears the status set after reading.
 *
 * @param[in]  entity  Entity on which the status has to be read.
 * @param[out] status  Returns the status set on the entity, based on the enabled status.
 * @param[in]  mask    Filter the status condition to be read, 0 means all statuses
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Success.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter, status is a null pointer or
 *             mask has bits set outside the status range.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object or mask has status
 *             bits set that are undefined for the type of entity.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_take_status(dds_entity_t entity, uint32_t *status, uint32_t mask);

/**
 * @brief Get changed status(es)
 * @ingroup entity_status
 *
 * This operation returns the status changes since they were last read.
 *
 * @param[in]  entity  Entity on which the statuses are read.
 * @param[out] status  Returns the current set of triggered statuses.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Success.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_status_changes(dds_entity_t entity, uint32_t *status);

/**
 * @anchor dds_get_status_mask
 * @brief Get enabled status on entity
 * @ingroup entity_status
 *
 * This operation returns the status enabled on the entity
 *
 * @param[in]  entity  Entity to get the status.
 * @param[out] mask    Mask of enabled statuses set on the entity.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Success.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_status_mask(dds_entity_t entity, uint32_t *mask);

/**
 * @deprecated Get enabled status on entity. Use \ref dds_get_status_mask instead.
 * @ingroup deprecated
 *
 * @param[in] entity  Entity to get the status.
 * @param[out] mask   Mask of enabled statuses set on the entity.
 * @returns A dds_return_t indicating success of failure.
 */
DDS_DEPRECATED_EXPORT dds_return_t
dds_get_enabled_status(dds_entity_t entity, uint32_t *mask);

/**
 * @anchor dds_set_status_mask
 * @brief Set status enabled on entity
 * @ingroup entity_status
 *
 * This operation enables the status(es) based on the mask set
 *
 * @param[in]  entity  Entity to enable the status.
 * @param[in]  mask    Status value that indicates the status to be enabled.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Success.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_set_status_mask(dds_entity_t entity, uint32_t mask);

/**
 * @deprecated Set enabled status on entity. Use \ref dds_set_status_mask instead.
 * @ingroup deprecated
 *
 * @param[in] entity  Entity to enable the status.
 * @param[out] mask   Status value that indicates the status to be enabled.
 * @returns A dds_return_t indicating success of failure.
 */
DDS_DEPRECATED_EXPORT dds_return_t
dds_set_enabled_status(dds_entity_t entity, uint32_t mask);

/**
 * @defgroup entity_qos (Entity QoS)
 * @ingroup entity
 * @brief Almost all entities have get/set qos operations defined on them,
 * again following the DCPS spec. But unlike the DCPS spec, the
 * "present" field in qos_t allows one to initialize just the one QoS
 * one wants to set & pass it to set_qos.
 */

/**
 * @brief Get entity QoS policies.
 * @ingroup entity_qos
 *
 * This operation allows access to the existing set of QoS policies
 * for the entity.
 *
 * @param[in]  entity  Entity on which to get qos.
 * @param[out] qos     Pointer to the qos structure that returns the set policies.
 *
 * @returns A dds_return_t indicating success or failure. The QoS object will have
 * at least all QoS relevant for the entity present and the corresponding dds_qget_...
 * will return true.
 *
 * @retval DDS_RETCODE_OK
 *             The existing set of QoS policy values applied to the entity
 *             has successfully been copied into the specified qos parameter.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The qos parameter is NULL.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 *
 * DOC_TODO: Link to generic QoS information documentation.
 */
DDS_EXPORT dds_return_t
dds_get_qos(dds_entity_t entity, dds_qos_t *qos);

/**
 * @brief Set entity QoS policies.
 * @ingroup entity_qos
 *
 * This operation replaces the existing set of Qos Policy settings for an
 * entity. The parameter qos must contain the struct with the QosPolicy
 * settings which is checked for self-consistency.
 *
 * The set of QosPolicy settings specified by the qos parameter are applied on
 * top of the existing QoS, replacing the values of any policies previously set
 * (provided, the operation returned DDS_RETCODE_OK).
 *
 * Not all policies are changeable when the entity is enabled.
 *
 * @note Currently only Latency Budget and Ownership Strength are changeable QoS
 *       that can be set.
 *
 * @param[in]  entity  Entity from which to get qos.
 * @param[in]  qos     Pointer to the qos structure that provides the policies.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The new QoS policies are set.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The qos parameter is NULL.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_IMMUTABLE_POLICY
 *             The entity is enabled and one or more of the policies of the QoS
 *             are immutable.
 * @retval DDS_RETCODE_INCONSISTENT_POLICY
 *             A few policies within the QoS are not consistent with each other.
 *
 * DOC_TODO: Link to generic QoS information documentation.
 */
DDS_EXPORT dds_return_t
dds_set_qos(dds_entity_t entity, const dds_qos_t * qos);

/**
 * @defgroup entity_listener (Entity Listener)
 * @ingroup entity
 * @brief Get or set listener associated with an entity,
 * type of listener provided much match type of entity.
 */

/**
 * @brief Get entity listeners.
 * @ingroup entity_listener
 *
 * This operation allows access to the existing listeners attached to
 * the entity.
 *
 * @param[in]  entity   Entity on which to get the listeners.
 * @param[out] listener Pointer to the listener structure that returns the
 *                      set of listener callbacks.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The listeners of to the entity have been successfully been
 *             copied into the specified listener parameter.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The listener parameter is NULL.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 *
 * DOC_TODO: Link to (generic) Listener and status information.
 */
DDS_EXPORT dds_return_t
dds_get_listener(dds_entity_t entity, dds_listener_t * listener);

/**
 * @brief Set entity listeners.
 * @ingroup entity_listener
 *
 * This operation attaches a dds_listener_t to the dds_entity_t. Only one
 * Listener can be attached to each Entity. If a Listener was already
 * attached, this operation will replace it with the new one. In other
 * words, all related callbacks are replaced (possibly with NULL).
 *
 * When listener parameter is NULL, all listener callbacks that were possibly
 * set on the Entity will be removed.
 *
 * @note Not all listener callbacks are related to all entities.
 *
 * ## Communication Status
 * For each communication status, the StatusChangedFlag flag is initially set to
 * FALSE. It becomes TRUE whenever that plain communication status changes. For
 * each plain communication status activated in the mask, the associated
 * Listener callback is invoked and the communication status is reset
 * to FALSE, as the listener implicitly accesses the status which is passed as a
 * parameter to that operation.
 * The status is reset prior to calling the listener, so if the application calls
 * the get_<status_name> from inside the listener it will see the
 * status already reset.
 *
 * ## Status Propagation
 * In case a related callback within the Listener is not set, the Listener of
 * the Parent entity is called recursively, until a Listener with the appropriate
 * callback set has been found and called. This allows the application to set
 * (for instance) a default behaviour in the Listener of the containing Publisher
 * and a DataWriter specific behaviour when needed. In case the callback is not
 * set in the Publishers' Listener either, the communication status will be
 * propagated to the Listener of the DomainParticipant of the containing
 * DomainParticipant. In case the callback is not set in the DomainParticipants'
 * Listener either, the Communication Status flag will be set, resulting in a
 * possible WaitSet trigger.
 *
 * @param[in]  entity    Entity on which to get the listeners.
 * @param[in]  listener  Pointer to the listener structure that contains the
 *                       set of listener callbacks (maybe NULL).
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The listeners of to the entity have been successfully been
 *             copied into the specified listener parameter.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * DOC_TODO: Link to (generic) Listener and status information.
 */
DDS_EXPORT dds_return_t
dds_set_listener(dds_entity_t entity, const dds_listener_t * listener);

/*
  Creation functions for various entities. Creating a subscriber or
  publisher is optional: if one creates a reader as a descendant of a
  participant, it is as if a subscriber is created specially for
  that reader.

  QoS default values are those of the DDS specification, but the
  inheritance rules are different:

    * publishers and subscribers inherit from the participant QoS
    * readers and writers always inherit from the topic QoS
    * the QoS's present in the "qos" parameter override the inherited values
*/

/**
 * @defgroup domain (Domain)
 * @ingroup DDS
 */

/**
 * @defgroup domain_participant (DomainParticipant)
 * @ingroup domain
 */

/**
 * @brief Creates a new instance of a DDS participant in a domain
 * @ingroup domain_participant
 *
 * If domain is set (not DDS_DOMAIN_DEFAULT) then it must match if the domain has also
 * been configured or an error status will be returned.
 * Currently only a single domain can be configured by providing configuration file.
 * If no configuration file exists, the default domain is configured as 0.
 *
 *
 * @param[in]  domain The domain in which to create the participant (can be DDS_DOMAIN_DEFAULT). DDS_DOMAIN_DEFAULT is for using the domain in the configuration.
 * @param[in]  qos The QoS to set on the new participant (can be NULL).
 * @param[in]  listener Any listener functions associated with the new participant (can be NULL).

 * @returns A valid participant handle or an error code.
 *
 * @retval >0
 *             A valid participant handle.
 * @retval DDS_RETCODE_NOT_ALLOWED_BY_SECURITY
 *             An invalid DDS Security configuration was specified (whether
 *             that be missing or incorrect entries, expired certificates,
 *             or anything else related to the security settings and
 *             implementation).
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             Some security properties specified in the QoS, but the Cyclone
 *             build does not include support for DDS Security.
 * @retval DDS_RETCODE_OUT_OF_RESOURCES
 *             Some resource limit (maximum participants, memory, handles,
 *             &c.) prevented creation of the participant.
 * @retval DDS_RETCODE_ERROR
 *             The "CYCLONEDDS_URI" environment variable lists non-existent
 *             or invalid configuration files, or contains invalid embedded
 *             configuration items; or an unspecified internal error has
 *             occurred.
 */
DDS_EXPORT dds_entity_t
dds_create_participant(
  const dds_domainid_t domain,
  const dds_qos_t *qos,
  const dds_listener_t *listener);

/**
 * @brief Creates a domain with a given configuration
 * @ingroup domain
 *
 * To explicitly create a domain based on a configuration passed as a string.
 *
 * It will not be created if a domain with the given domain id already exists.
 * This could have been created implicitly by a dds_create_participant().
 *
 * Please be aware that the given domain_id always takes precedence over the
 * configuration.
 *
 * | domain_id | domain id in config | result                        |
 * |:----------|:--------------------|:------------------------------|
 * | n         | any (or absent)     | n, config is used             |
 * | n         | m == n              | n, config is used             |
 * | n         | m != n              | n, config is ignored: default |
 *
 * Config models:
 *  -# @code{xml}
 *     <CycloneDDS>
 *        <Domain id="X">...</Domain>
 *        <!-- <Domain .../> -->
 *      </CycloneDDS>
 *      @endcode
 *      where ... is all that can today be set in children of CycloneDDS
 *      with the exception of the id
 *  -# @code{xml}
 *     <CycloneDDS>
 *        <Domain><Id>X</Id></Domain>
 *        <!-- more things here ... -->
 *     </CycloneDDS>
 *     @endcode
 *     Legacy form, domain id must be the first element in the file with
 *     a value (if nothing has been set previously, it a warning is good
 *     enough)
 *
 * Using NULL or "" as config will create a domain with default settings.
 *
 *
 * @param[in]  domain The domain to be created. DEFAULT_DOMAIN is not allowed.
 * @param[in]  config A configuration string containing file names and/or XML fragments representing the configuration.
 *
 * @returns A valid entity handle or an error code.
 *
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Illegal value for domain id or the configfile parameter is NULL.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The domain already existed and cannot be created again.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 */
DDS_EXPORT dds_entity_t
dds_create_domain(const dds_domainid_t domain, const char *config);

/**
 * @brief Creates a domain with a given configuration, specified as an
 * initializer (unstable interface)
 * @ingroup domain
 *
 * To explicitly create a domain based on a configuration passed as a raw
 * initializer rather than as an XML string. This allows bypassing the XML
 * parsing, but tightly couples the initializing to implementation.  See
 * dds/ddsi/ddsi_config.h:ddsi_config_init_default for a way to initialize
 * the default configuration.
 *
 * It will not be created if a domain with the given domain id already exists.
 * This could have been created implicitly by a dds_create_participant().
 *
 * Please be aware that the given domain_id always takes precedence over the
 * configuration.
 *
 * @param[in]  domain The domain to be created. DEFAULT_DOMAIN is not allowed.
 * @param[in]  config A configuration initializer.  The lifetime of any pointers
 *             in config must be at least that of the lifetime of the domain.
 *
 * @returns A valid entity handle or an error code.
 *
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Illegal value for domain id or the config parameter is NULL.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The domain already existed and cannot be created again.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 */
DDS_EXPORT dds_entity_t
dds_create_domain_with_rawconfig(const dds_domainid_t domain, const struct ddsi_config *config);

/**
 * @brief Get entity parent.
 * @ingroup entity
 *
 * This operation returns the parent to which the given entity belongs.
 * For instance, it will return the Participant that was used when
 * creating a Publisher (when that Publisher was provided here).
 *
 * When a reader or a writer are created with a participant, then a
 * subscriber or publisher are created implicitly.
 * This function will return the implicit parent and not the used
 * participant.
 *
 * @param[in]  entity  Entity from which to get its parent.
 *
 * @returns A valid entity handle or an error code.
 *
 * @retval >0
 *             A valid entity handle.
 * @retval DDS_ENTITY_NIL
 *             Called with a participant.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * DOC_TODO: Link to generic dds entity relations documentation.
 */
DDS_EXPORT dds_entity_t
dds_get_parent(dds_entity_t entity);

/**
 * @brief Get entity participant.
 * @ingroup entity
 *
 * This operation returns the participant to which the given entity belongs.
 * For instance, it will return the Participant that was used when
 * creating a Publisher that was used to create a DataWriter (when that
 * DataWriter was provided here).
 *
 * DOC_TODO: Link to generic dds entity relations documentation.
 *
 * @param[in]  entity  Entity from which to get its participant.
 *
 * @returns A valid entity or an error code.
 *
 * @retval >0
 *             A valid participant handle.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_entity_t
dds_get_participant(dds_entity_t entity);

/**
 * @brief Get entity children.
 * @ingroup entity
 *
 * This operation returns the children that the entity contains.
 * For instance, it will return all the Topics, Publishers and Subscribers
 * of the Participant that was used to create those entities (when that
 * Participant is provided here).
 *
 * This functions takes a pre-allocated list to put the children in and
 * will return the number of found children. It is possible that the given
 * size of the list is not the same as the number of found children. If
 * less children are found, then the last few entries in the list are
 * untouched. When more children are found, then only 'size' number of
 * entries are inserted into the list, but still complete count of the
 * found children is returned. Which children are returned in the latter
 * case is undefined.
 *
 * When supplying NULL as list and 0 as size, you can use this to acquire
 * the number of children without having to pre-allocate a list.
 *
 * When a reader or a writer are created with a participant, then a
 * subscriber or publisher are created implicitly.
 * When used on the participant, this function will return the implicit
 * subscriber and/or publisher and not the related reader/writer.
 *
 * @param[in]  entity   Entity from which to get its children.
 * @param[out] children Pre-allocated array to contain the found children.
 * @param[in]  size     Size of the pre-allocated children's list.
 *
 * @returns Number of children or an error code.
 *
 * @retval >=0
 *             Number of found children (can be larger than 'size').
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The children parameter is NULL, while a size is provided.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
/* TODO: Link to generic dds entity relations documentation. */
DDS_EXPORT dds_return_t
dds_get_children(dds_entity_t entity, dds_entity_t *children, size_t size);

/**
 * @brief Get the domain id to which this entity is attached.
 * @ingroup entity
 *
 * When creating a participant entity, it is attached to a certain domain.
 * All the children (like Publishers) and childrens' children (like
 * DataReaders), etc are also attached to that domain.
 *
 * This function will return the original domain ID when called on
 * any of the entities within that hierarchy.  For entities not associated
 * with a domain, the id is set to DDS_DOMAIN_DEFAULT.
 *
 * @param[in]  entity   Entity from which to get its children.
 * @param[out] id       Pointer to put the domain ID in.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Domain ID was returned.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The id parameter is NULL.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_get_domainid(dds_entity_t entity, dds_domainid_t *id);

/**
 * @brief Get participants of a domain.
 * @ingroup domain
 *
 * This operation acquires the participants created on a domain and returns
 * the number of found participants.
 *
 * This function takes a domain id with the size of pre-allocated participant's
 * list in and will return the number of found participants. It is possible that
 * the given size of the list is not the same as the number of found participants.
 * If less participants are found, then the last few entries in an array stay
 * untouched. If more participants are found and the array is too small, then the
 * participants returned are undefined.
 *
 * @param[in]  domain_id    The domain id.
 * @param[out] participants The participant for domain.
 * @param[in]  size         Size of the pre-allocated participant's list.
 *
 * @returns Number of participants found or and error code.
 *
 * @retval >0
 *             Number of participants found.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The participant parameter is NULL, while a size is provided.
 */
DDS_EXPORT dds_return_t
dds_lookup_participant(
  dds_domainid_t domain_id,
  dds_entity_t *participants,
  size_t size);

/**
 * @defgroup topic (Topic)
 * @ingroup dds
 */

/**
 * @brief Creates a new topic with default type handling.
 * @ingroup topic
 *
 * The type name for the topic is taken from the generated descriptor. Topic
 * matching is done on a combination of topic name and type name. Each successful
 * call to dds_create_topic creates a new topic entity sharing the same QoS
 * settings with all other topics of the same name.
 *
 * @param[in]  participant  Participant on which to create the topic.
 * @param[in]  descriptor   An IDL generated topic descriptor.
 * @param[in]  name         Name of the topic.
 * @param[in]  qos          QoS to set on the new topic (can be NULL).
 * @param[in]  listener     Any listener functions associated with the new topic (can be NULL).
 *
 * @returns A valid, unique topic handle or an error code.
 *
 * @retval >=0
 *             A valid unique topic handle.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Either participant, descriptor, name or qos is invalid.
 * @retval DDS_RETCODE_INCONSISTENT_POLICY
 *             QoS mismatch between qos and an existing topic's QoS.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             Mismatch between type name in descriptor and pre-existing
 *             topic's type name.
 */
DDS_EXPORT dds_entity_t
dds_create_topic(
  dds_entity_t participant,
  const dds_topic_descriptor_t *descriptor,
  const char *name,
  const dds_qos_t *qos,
  const dds_listener_t *listener);


/**
 * @brief Indicates that the library defines the dds_create_topic_sertype function
 * @ingroup topic
 * Introduced to help with the change from sertopic to sertype. If you are using
 * a modern CycloneDDS version you will not need this.
 */
#define DDS_HAS_CREATE_TOPIC_SERTYPE 1

/**
 * @brief Creates a new topic with provided type handling.
 * @ingroup topic
 *
 * The name for the type is taken from the provided "sertype" object. Type
 * matching is done on a combination of topic name and type name. Each successful
 * call to dds_create_topic creates a new topic entity sharing the same QoS
 * settings with all other topics of the same name.
 *
 * In case this function returns a valid handle, the ownership of the provided
 * sertype is handed over to Cyclone. On return, the caller gets in the sertype parameter a
 * pointer to the sertype that is actually used by the topic. This can be the provided sertype
 * (if this sertype was not yet known in the domain), or a sertype thas was
 * already known in the domain.
 *
 * @param[in]     participant  Participant on which to create the topic.
 * @param[in]     name         Topic name
 * @param[in,out] sertype      Internal description of the type . On return, the sertype parameter is set to the actual sertype that is used by the topic.
 * @param[in]     qos          QoS to set on the new topic (can be NULL).
 * @param[in]     listener     Any listener functions associated with the new topic (can be NULL).
 * @param[in]     sedp_plist   Topic description to be published as part of discovery (if NULL, not published).
 *
 * @returns A valid, unique topic handle or an error code. Iff a valid handle, the domain takes ownership of provided serdata.
 *
 * @retval >=0
 *             A valid unique topic handle.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Either participant, descriptor, name or qos is invalid.
 * @retval DDS_RETCODE_INCONSISTENT_POLICY
 *             QoS mismatch between qos and an existing topic's QoS.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             Mismatch between type name in sertype and pre-existing
 *             topic's type name.
 */
DDS_EXPORT dds_entity_t
dds_create_topic_sertype (
  dds_entity_t participant,
  const char *name,
  struct ddsi_sertype **sertype,
  const dds_qos_t *qos,
  const dds_listener_t *listener,
  const struct ddsi_plist *sedp_plist);

/**
 * @brief Indicates that the library defines the dds_create_topic_generic function
 * @ingroup topic
 * Introduced to help with the change from sertopic to sertype. You should probably
 * move to using sertype instead of sertopic and ignore this.
 */
#define DDS_HAS_CREATE_TOPIC_GENERIC 1

/**
 * @anchor dds_create_topic_generic
 * @brief Creates a new topic with provided type handling.
 * @ingroup topic
 *
 * DOC_TODO: this description does not make any sense, this is for the legacy sertopic?
 *           deprecate?
 *
 * The name for the type is taken from the provided "sertype" object. Type
 * matching is done on a combination of topic name and type name. Each successful
 * call to dds_create_topic creates a new topic entity sharing the same QoS
 * settings with all other topics of the same name.
 *
 * In case this function returns a valid handle, the ownership of the provided
 * sertype is handed over to Cyclone. On return, the caller gets in the sertype parameter a
 * pointer to the sertype that is actually used by the topic. This can be the provided sertype
 * (if this sertype was not yet known in the domain), or a sertype thas was
 * already known in the domain.
 *
 * @param[in]     participant  Participant on which to create the topic.
 * @param[in,out] sertopic     Legacy internal description of the type. On return, the sertype parameter is set to the actual sertype that is used by the topic.
 * @param[in]     qos          QoS to set on the new topic (can be NULL).
 * @param[in]     listener     Any listener functions associated with the new topic (can be NULL).
 * @param[in]     sedp_plist   Topic description to be published as part of discovery (if NULL, not published).
 *
 * @returns A valid, unique topic handle or an error code. Iff a valid handle, the domain takes ownership of provided serdata.
 *
 * @retval >=0
 *             A valid unique topic handle.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Either participant, descriptor, name or qos is invalid.
 * @retval DDS_RETCODE_INCONSISTENT_POLICY
 *             QoS mismatch between qos and an existing topic's QoS.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             Mismatch between type name in sertype and pre-existing
 *             topic's type name.
 */
DDS_EXPORT dds_entity_t
dds_create_topic_generic (
  dds_entity_t participant,
  struct ddsi_sertopic **sertopic,
  const dds_qos_t *qos,
  const dds_listener_t *listener,
  const struct ddsi_plist *sedp_plist);

/**
 * @deprecated Creates a new topic with provided type handling.
 * @ingroup deprecated
 * Use @ref dds_create_topic_sertype instead.
 *
 * @param[in]     participant  Participant on which to create the topic.
 * @param[in,out] sertopic     Legacy internal description of the type. On return, the sertype parameter is set to the actual sertype that is used by the topic.
 * @param[in]     qos          QoS to set on the new topic (can be NULL).
 * @param[in]     listener     Any listener functions associated with the new topic (can be NULL).
 * @param[in]     sedp_plist   Topic description to be published as part of discovery (if NULL, not published).
 *
 * @returns       A valid, unique topic handle or an error code.
 */
DDS_DEPRECATED_EXPORT dds_entity_t
dds_create_topic_arbitrary (
  dds_entity_t participant,
  struct ddsi_sertopic *sertopic,
  const dds_qos_t *qos,
  const dds_listener_t *listener,
  const struct ddsi_plist *sedp_plist);

/**
 * @brief Finds a locally created or discovered remote topic by topic name and type information
 * @ingroup topic
 *
 * Finds a locally created topic or a discovered remote topic based on the topic
 * name and type. In case the topic is not found, this function will wait for
 * the topic to become available until the provided time out.
 *
 * When using the scope DDS_FIND_SCOPE_LOCAL_DOMAIN, there will be no requests sent
 * over the network for resolving the type in case it is unresolved. This also applies
 * to dependent types: in case a dependency of the provided type is unresolved, no
 * requests will be sent for resolving the type when using LOCAL_DOMAIN scope.
 *
 * In case the scope is DDS_FIND_SCOPE_GLOBAL, for unresolved types (or dependencies)
 * a type lookup request will be sent.
 *
 * In case no type information is provided and multiple (discovered) topics exist
 * with the provided name, an arbitrary topic with that name will be returned.
 * In this scenario, it would be better to read DCPSTopic data and use that to
 * get the required topic meta-data.
 *
 * The returned topic should be released with dds_delete.
 *
 * @param[in]  scope        The scope used to find the topic. In case topic discovery is not enabled in the build, SCOPE_GLOBAL cannot be used.
 * @param[in]  participant  The handle of the participant the found topic will be created in
 * @param[in]  name         The name of the topic to find.
 * @param[in]  type_info    The type information of the topic to find. Optional, and should not be provided in case topic discovery is not enabled in the build.
 * @param[in]  timeout      The timeout for waiting for the topic to become available
 *
 * @returns A valid topic handle or an error code.
 *
 * @retval >0
 *             A valid topic handle.
 * @retval 0
 *             No topic of this name exists
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Participant or type information was invalid.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The provided type could not be found.
 */
DDS_EXPORT dds_entity_t
dds_find_topic (dds_find_scope_t scope, dds_entity_t participant, const char *name, const dds_typeinfo_t *type_info, dds_duration_t timeout);

/**
 * @deprecated Finds a locally created or discovered remote topic by topic name
 * @ingroup deprecated
 * Use @ref dds_find_topic instead.
 *
 * Finds a locally created topic or a discovered remote topic based on the topic
 * name. In case the topic is not found, this function will wait for the topic
 * to become available until the provided time out.
 *
 * In case multiple (discovered) topics exist with the provided name, this function
 * will return randomly one of these topic. The caller can decide to read DCPSTopic
 * data and select one of the topic definitions to create the topic.
 *
 * The returned topic should be released with dds_delete.
 *
 * @param[in]  scope        The scope used to find the topic
 * @param[in]  participant  The handle of the participant the found topic will be created in
 * @param[in]  name         The name of the topic to find.
 * @param[in]  timeout      The timeout for waiting for the topic to become available
 *
 * @returns A valid topic handle or an error code.
 *
 * @retval >0
 *             A valid topic handle.
 * @retval 0
 *             No topic of this name existed yet
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Participant handle or scope invalid
 */
DDS_DEPRECATED_EXPORT dds_entity_t
dds_find_topic_scoped (dds_find_scope_t scope, dds_entity_t participant, const char *name, dds_duration_t timeout);


/**
 * @ingroup topic
 * @brief Creates topic descriptor for the provided type_info
 *
 * @param[in]  scope        The scope used to find the type: DDS_FIND_SCOPE_LOCAL_DOMAIN or DDS_FIND_SCOPE_GLOBAL. In case DDS_FIND_SCOPE_GLOBAL is used, a type lookup request will be sent to other nodes.
 * @param[in]  participant  The handle of the participant.
 * @param[in]  type_info    The type (dds_typeinfo_t) of the topic to find.
 * @param[in]  timeout      The timeout for waiting for the type to become available
 * @param[out] descriptor - Pointer to a dds_topic_descriptor_t pointer that will be allocated and populated. To free allocated memory for this descriptor, use dds_delete_topic_descriptor.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The topic descriptor has been succesfully created.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Type_info or descriptor parameter not provided, invalid entity (not a participant) or scope invalid.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The participant or the type_id was not found.
 * @retval DDS_RETCODE_TIMEOUT
 *             Type was not resolved within the provided timeout
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Cyclone DDS built without type discovery
 *             (cf. DDS_HAS_TYPE_DISCOVERY)
 */
DDS_EXPORT dds_return_t
dds_create_topic_descriptor (dds_find_scope_t scope, dds_entity_t participant, const dds_typeinfo_t *type_info, dds_duration_t timeout, dds_topic_descriptor_t **descriptor);

/**
 * @ingroup topic
 * @brief Delete memory allocated to the provided topic descriptor
 *
 * @param[in] descriptor - Pointer to a dds_topic_descriptor_t
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The topic descriptor has been succesfully deleted.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             No descriptor provided
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Cyclone DDS built without type discovery
 *             (cf. DDS_HAS_TYPE_DISCOVERY)
 */
DDS_EXPORT dds_return_t
dds_delete_topic_descriptor (dds_topic_descriptor_t *descriptor);

/**
 * @brief Returns the name of a given topic.
 * @ingroup topic
 *
 * @param[in]  topic  The topic.
 * @param[out] name   Buffer to write the topic name to.
 * @param[in]  size   Number of bytes available in the buffer.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @return Actual length of topic name (name is truncated if return value >= size) or error
 */
DDS_EXPORT dds_return_t
dds_get_name(dds_entity_t topic, char *name, size_t size);

/**
 * @brief Returns the type name of a given topic.
 * @ingroup topic
 *
 * @param[in]  topic  The topic.
 * @param[out] name   Buffer to write the topic type name to.
 * @param[in]  size   Number of bytes available in the buffer.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @return Actual length of type name (name is truncated if return value >= size) or error
 */
DDS_EXPORT dds_return_t
dds_get_type_name(dds_entity_t topic, char *name, size_t size);

/**
 * @defgroup topic_filter (Topic filters)
 * @ingroup topic
 * Topic filter functions.
 * @warning part of the Unstable API: no guarantee that any
 *          of this will be maintained for backwards compatibility.
 *
 * Sampleinfo is all zero when filtering in a write call (i.e., writer created
 * using a filtered topic, which one perhaps shouldn't be doing), otherwise it
 * has as much filled in correctly as is possible given the context and the rest
 * fixed:
 *    - sample_state         DDS_SST_NOT_READ;
 *    - publication_handle   set to writer's instance handle
 *    - source_timestamp     set to source timestamp of sample
 *    - ranks                0
 *    - valid_data           true
 *    - instance_handle      set to instance handle of existing instance if the
 *                           sample matches an existing instance, otherwise to what
 *                           the instance handle will be if it passes the filter
 *    - view_state           set to instance view state if sample being filtered
 *                           matches an existing instance, NEW if not
 *    - instance_state       set to instance state if sample being filtered
 *                           matches an existing instance, NEW if not
 *    - generation counts    set to instance's generation counts if the sample
 *                           matches an existing instance instance, 0 if not
 */

/**
 * @anchor dds_topic_filter_sample_fn
 * @brief Topic filter function that only needs to look at the sample.
 * @ingroup topic_filter
 * @warning Unstable API
 */
typedef bool (*dds_topic_filter_sample_fn) (const void * sample);

/**
 * @anchor dds_topic_filter_sample_arg_fn
 * @brief Topic filter function that only needs to look at the sample and a custom argument.
 * @ingroup topic_filter
 * @warning Unstable API
 */
typedef bool (*dds_topic_filter_sample_arg_fn) (const void * sample, void * arg);

/**
 * @anchor dds_topic_filter_sampleinfo_arg_fn
 * @brief Topic filter function that only needs to look at the sampleinfo and a custom argument.
 * @ingroup topic_filter
 * @warning Unstable API
 */
typedef bool (*dds_topic_filter_sampleinfo_arg_fn) (const dds_sample_info_t * sampleinfo, void * arg);

/**
 * @anchor dds_topic_filter_sample_sampleinfo_arg_fn
 * @brief Topic filter function that needs to look at the sample, the sampleinfo and a custom argument.
 * @ingroup topic_filter
 * @warning Unstable API
 */
typedef bool (*dds_topic_filter_sample_sampleinfo_arg_fn) (const void * sample, const dds_sample_info_t * sampleinfo, void * arg);

/**
 * @anchor dds_topic_filter_fn
 * @brief See \ref dds_topic_filter_sample_fn
 * @ingroup topic_filter
 * @warning Unstable API
 */
typedef dds_topic_filter_sample_fn dds_topic_filter_fn;

/**
 * @anchor dds_topic_filter_arg_fn
 * @brief See \ref dds_topic_filter_sample_arg_fn
 * @ingroup topic_filter
 * @warning Unstable API
 */
typedef dds_topic_filter_sample_arg_fn dds_topic_filter_arg_fn;

/**
 * @brief Topic filter mode;
 * @ingroup topic_filter
 * @warning Unstable API
 */
enum dds_topic_filter_mode {
  DDS_TOPIC_FILTER_NONE,                  /**< Can be used to reset topic filter */
  DDS_TOPIC_FILTER_SAMPLE,                /**< Use with \ref dds_topic_filter_sample_fn */
  DDS_TOPIC_FILTER_SAMPLE_ARG,            /**< Use with \ref dds_topic_filter_sample_arg_fn */
  DDS_TOPIC_FILTER_SAMPLEINFO_ARG,        /**< Use with \ref dds_topic_filter_sampleinfo_arg_fn */
  DDS_TOPIC_FILTER_SAMPLE_SAMPLEINFO_ARG, /**< Use with \ref dds_topic_filter_sample_sampleinfo_arg_fn */
};

/**
 * @brief Union of all filter function types;
 * @ingroup topic_filter
 * @warning Unstable API
*/
union dds_topic_filter_function_union {
  dds_topic_filter_sample_fn sample; /**< Use with mode dds_topic_filter_mode::DDS_TOPIC_FILTER_SAMPLE */
  dds_topic_filter_sample_arg_fn sample_arg; /**< Use with mode dds_topic_filter_mode::DDS_TOPIC_FILTER_SAMPLE_ARG */
  dds_topic_filter_sampleinfo_arg_fn sampleinfo_arg; /**< Use with mode dds_topic_filter_mode::DDS_TOPIC_FILTER_SAMPLEINFO_ARG */
  dds_topic_filter_sample_sampleinfo_arg_fn sample_sampleinfo_arg; /**< Use with mode dds_topic_filter_mode::DDS_TOPIC_FILTER_SAMPLE_SAMPLEINFO_ARG */
};

/**
 * @brief Full topic filter container;
 * @ingroup topic_filter
 * @warning Unstable API
 */
struct dds_topic_filter {
  enum dds_topic_filter_mode mode;         /**< Provide a mode */
  union dds_topic_filter_function_union f; /**< Provide a filter function */
  void *arg;                               /**< Provide an argument, can be NULL */
};

/**
 * @brief Sets a filter on a topic. To be replaced by proper filtering on readers,
 * @ingroup topic_filter
 * @warning Unstable API
 * @deprecated use @ref dds_set_topic_filter_and_arg or @ref dds_set_topic_filter_extended instead
 *
 * Not thread-safe with respect to data being read/written using readers/writers
 * using this topic.  Be sure to create a topic entity specific to the reader you
 * want to filter, then set the filter function, and only then create the reader.
 * And don't change it unless you know there are no concurrent writes.
 *
 * @param[in]  topic   The topic on which the content filter is set.
 * @param[in]  filter  The filter function used to filter topic samples.
 */
DDS_DEPRECATED_EXPORT void
dds_set_topic_filter(dds_entity_t topic, dds_topic_filter_fn filter);

/**
 * @brief Sets a filter on a topic. To be replaced by proper filtering on readers,
 * @ingroup topic_filter
 * @warning Unstable API
 * @deprecated use @ref dds_set_topic_filter_and_arg or @ref dds_set_topic_filter_extended instead
 *
 * Not thread-safe with respect to data being read/written using readers/writers
 * using this topic.  Be sure to create a topic entity specific to the reader you
 * want to filter, then set the filter function, and only then create the reader.
 * And don't change it unless you know there are no concurrent writes.
 *
 * @param[in]  topic   The topic on which the content filter is set.
 * @param[in]  filter  The filter function used to filter topic samples.
 */
DDS_DEPRECATED_EXPORT void
dds_topic_set_filter(dds_entity_t topic, dds_topic_filter_fn filter);

/**
 * @anchor dds_set_topic_filter_and_arg
 * @brief Sets a filter and filter argument on a topic.
 * @ingroup topic_filter
 * @warning Unstable API
 * To be replaced by proper filtering on readers.
 *
 * Not thread-safe with respect to data being read/written using readers/writers
 * using this topic.  Be sure to create a topic entity specific to the reader you
 * want to filter, then set the filter function, and only then create the reader.
 * And don't change it unless you know there are no concurrent writes.
 *
 * @param[in]  topic   The topic on which the content filter is set.
 * @param[in]  filter  The filter function used to filter topic samples.
 * @param[in]  arg     Argument for the filter function.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK  Filter set successfully
 * @retval DDS_RETCODE_BAD_PARAMETER  The topic handle is invalid
*/
DDS_EXPORT dds_return_t
dds_set_topic_filter_and_arg(
  dds_entity_t topic,
  dds_topic_filter_arg_fn filter,
  void *arg);

/**
 * @anchor dds_set_topic_filter_extended
 * @brief Sets a filter and filter argument on a topic.
 * @ingroup topic_filter
 * @warning Unstable API
 * To be replaced by proper filtering on readers.
 *
 * Not thread-safe with respect to data being read/written using readers/writers
 * using this topic.  Be sure to create a topic entity specific to the reader you
 * want to filter, then set the filter function, and only then create the reader.
 * And don't change it unless you know there are no concurrent writes.
 *
 * @param[in]  topic   The topic on which the content filter is set.
 * @param[in]  filter  The filter specification.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK  Filter set successfully
 * @retval DDS_RETCODE_BAD_PARAMETER  The topic handle is invalid
*/
DDS_EXPORT dds_return_t
dds_set_topic_filter_extended(
  dds_entity_t topic,
  const struct dds_topic_filter *filter);

/**
 * @brief Gets the filter for a topic.
 * @ingroup topic_filter
 * @deprecated Use dds_get_topic_filter_and_arg() or dds_get_topic_filter_extended() instead.
 * @warning Unstable API
 *
 * To be replaced by proper filtering on readers.
 *
 * @param[in]  topic  The topic from which to get the filter.
 *
 * @returns The topic filter, or 0 when of type other than "sample".
 */
DDS_DEPRECATED_EXPORT dds_topic_filter_fn
dds_get_topic_filter(dds_entity_t topic);

/**
 * @brief Gets the filter for a topic.
 * @ingroup topic_filter
 * @deprecated Use dds_get_topic_filter_and_arg() or dds_get_topic_filter_extended() instead.
 * @warning Unstable API
 *
 * To be replaced by proper filtering on readers.
 *
 * @param[in]  topic  The topic from which to get the filter.
 *
 * @returns The topic filter, or 0 when of type other than "sample".
 */
DDS_DEPRECATED_EXPORT dds_topic_filter_fn
dds_topic_get_filter(dds_entity_t topic);

/**
 * @brief Gets the filter for a topic.
 * @ingroup topic_filter
 * @warning Unstable API
 *
 * To be replaced by proper filtering on readers
 *
 * @param[in]  topic  The topic from which to get the filter.
 * @param[out] fn     The topic filter function (fn may be NULL).
 * @param[out] arg    Filter function argument (arg may be NULL).
 *
 * @retval DDS_RETCODE_OK  Filter set successfully
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET  Filter is not of "none" or "sample_arg"
 * @retval DDS_RETCODE_BAD_PARAMETER  The topic handle is invalid
 */
DDS_EXPORT dds_return_t
dds_get_topic_filter_and_arg (
  dds_entity_t topic,
  dds_topic_filter_arg_fn *fn,
  void **arg);

/**
 * @brief Gets the filter for a topic.
 * @ingroup topic_filter
 * @warning Unstable API
 *
 * To be replaced by proper filtering on readers
 *
 * @param[in]  topic  The topic from which to get the filter.
 * @param[out] filter The topic filter specification.
 *
 * @retval DDS_RETCODE_OK  Filter set successfully
 * @retval DDS_RETCODE_BAD_PARAMETER  The topic handle is invalid
 */
DDS_EXPORT dds_return_t
dds_get_topic_filter_extended (
  dds_entity_t topic,
  struct dds_topic_filter *filter);

/**
 * @defgroup subscriber (Subscriber)
 * @ingroup subscription
 * DOC_TODO The Subscriber is a DDS Entity
 */

/**
 * @brief Creates a new instance of a DDS subscriber
 * @ingroup subscriber
 *
 * @param[in]  participant The participant on which the subscriber is being created.
 * @param[in]  qos         The QoS to set on the new subscriber (can be NULL).
 * @param[in]  listener    Any listener functions associated with the new subscriber (can be NULL).
 *
 * @returns A valid subscriber handle or an error code.
 *
 * @retval >0
 *             A valid subscriber handle.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the parameters is invalid.
 */
DDS_EXPORT dds_entity_t
dds_create_subscriber(
  dds_entity_t participant,
  const dds_qos_t *qos,
  const dds_listener_t *listener);


/**
 * @defgroup publication (Publication)
 * @ingroup dds
 * DOC_TODO This contains the definitions regarding publication of data.
 */

/**
 * @defgroup publisher (Publisher)
 * @ingroup publication
 * DOC_TODO The Publisher is a DDS Entity
 */

/**
 * @brief Creates a new instance of a DDS publisher
 * @ingroup publisher
 *
 * @param[in]  participant The participant to create a publisher for.
 * @param[in]  qos         The QoS to set on the new publisher (can be NULL).
 * @param[in]  listener    Any listener functions associated with the new publisher (can be NULL).
 *
 * @returns A valid publisher handle or an error code.
 *
 * @retval >0
 *            A valid publisher handle.
 * @retval DDS_RETCODE_ERROR
 *            An internal error has occurred.
 */
/* TODO: Check list of error codes is complete. */
DDS_EXPORT dds_entity_t
dds_create_publisher(
  dds_entity_t participant,
  const dds_qos_t *qos,
  const dds_listener_t *listener);

/**
 * @brief Suspends the publications of the Publisher
 * @ingroup publisher
 *
 * This operation is a hint to the Service so it can optimize its performance by e.g., collecting
 * modifications to DDS writers and then batching them. The Service is not required to use the hint.
 *
 * Every invocation of this operation must be matched by a corresponding call to @see dds_resume
 * indicating that the set of modifications has completed.
 *
 * @param[in]  publisher The publisher for which all publications will be suspended.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Publications suspended successfully.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The pub parameter is not a valid publisher.
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Operation is not supported.
 */
DDS_EXPORT dds_return_t
dds_suspend(dds_entity_t publisher);

/**
 * @brief Resumes the publications of the Publisher
 * @ingroup publisher
 *
 * This operation is a hint to the Service to indicate that the application has
 * completed changes initiated by a previous dds_suspend(). The Service is not
 * required to use the hint.
 *
 * The call to resume_publications must match a previous call to @see suspend_publications.
 *
 * @param[in]  publisher The publisher for which all publications will be resumed.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Publications resumed successfully.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The pub parameter is not a valid publisher.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             No previous matching dds_suspend().
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Operation is not supported.
 */
DDS_EXPORT dds_return_t
dds_resume(dds_entity_t publisher);

/**
 * @brief Waits at most for the duration timeout for acks for data in the publisher or writer.
 * @ingroup publication
 *
 * This operation blocks the calling thread until either all data written by the publisher
 * or writer is acknowledged by all matched reliable reader entities, or else the duration
 * specified by the timeout parameter elapses, whichever happens first.
 *
 * @param[in]  publisher_or_writer Publisher or writer whose acknowledgments must be waited for
 * @param[in]  timeout             How long to wait for acknowledgments before time out
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             All acknowledgments successfully received with the timeout.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The publisher_or_writer is not a valid publisher or writer.
 * @retval DDS_RETCODE_TIMEOUT
 *             Timeout expired before all acknowledgments from reliable reader entities were received.
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Operation is not supported.
 */
DDS_EXPORT dds_return_t
dds_wait_for_acks(dds_entity_t publisher_or_writer, dds_duration_t timeout);


/**
 * @defgroup reader (Reader)
 * @ingroup subscription
 * DOC_TODO The reader is a DDS Entity
 */

/**
 * @brief Creates a new instance of a DDS reader.
 * @ingroup reader
 *
 * When a participant is used to create a reader, an implicit subscriber is created.
 * This implicit subscriber will be deleted automatically when the created reader
 * is deleted.
 *
 * @param[in]  participant_or_subscriber The participant or subscriber on which the reader is being created.
 * @param[in]  topic                     The topic to read.
 * @param[in]  qos                       The QoS to set on the new reader (can be NULL).
 * @param[in]  listener                  Any listener functions associated with the new reader (can be NULL).
 *
 * @returns A valid reader handle or an error code.
 *
 * @retval >0
 *            A valid reader handle.
 * @retval DDS_RETCODE_ERROR
 *            An internal error occurred.
 *
 * DOC_TODO: Complete list of error codes
 */
DDS_EXPORT dds_entity_t
dds_create_reader(
  dds_entity_t participant_or_subscriber,
  dds_entity_t topic,
  const dds_qos_t *qos,
  const dds_listener_t *listener);

/**
 * @brief Creates a new instance of a DDS reader with a custom history cache.
 * @ingroup reader
 *
 * When a participant is used to create a reader, an implicit subscriber is created.
 * This implicit subscriber will be deleted automatically when the created reader
 * is deleted.
 *
 * @param[in]  participant_or_subscriber The participant or subscriber on which the reader is being created.
 * @param[in]  topic                     The topic to read.
 * @param[in]  qos                       The QoS to set on the new reader (can be NULL).
 * @param[in]  listener                  Any listener functions associated with the new reader (can be NULL).
 * @param[in]  rhc                       Reader history cache to use, reader becomes the owner
 *
 * @returns A valid reader handle or an error code.
 *
 * @retval >0
 *            A valid reader handle.
 * @retval DDS_RETCODE_ERROR
 *            An internal error occurred.
 *
 * DOC_TODO: Complete list of error codes
 */
DDS_EXPORT dds_entity_t
dds_create_reader_rhc(
  dds_entity_t participant_or_subscriber,
  dds_entity_t topic,
  const dds_qos_t *qos,
  const dds_listener_t *listener,
  struct dds_rhc *rhc);

/**
 * @brief Wait until reader receives all historic data
 * @ingroup reader
 *
 * The operation blocks the calling thread until either all "historical" data is
 * received, or else the duration specified by the max_wait parameter elapses, whichever happens
 * first. A return value of 0 indicates that all the "historical" data was received; a return
 * value of TIMEOUT indicates that max_wait elapsed before all the data was received.
 *
 * @param[in]  reader    The reader on which to wait for historical data.
 * @param[in]  max_wait  How long to wait for historical data before time out.
 *
 * @returns a status, 0 on success, TIMEOUT on timeout or a negative value to indicate error.
 *
 * DOC_TODO: Complete list of error codes
 */
DDS_EXPORT dds_return_t
dds_reader_wait_for_historical_data(
  dds_entity_t reader,
  dds_duration_t max_wait);

/**
 * @defgroup writer (Writer)
 * @ingroup publication
 * DOC_TODO The writer is a DDS Entity
 */

/**
 * @brief Creates a new instance of a DDS writer.
 * @ingroup writer
 *
 * When a participant is used to create a writer, an implicit publisher is created.
 * This implicit publisher will be deleted automatically when the created writer
 * is deleted.
 *
 * @param[in]  participant_or_publisher The participant or publisher on which the writer is being created.
 * @param[in]  topic The topic to write.
 * @param[in]  qos The QoS to set on the new writer (can be NULL).
 * @param[in]  listener Any listener functions associated with the new writer (can be NULL).
 *
 * @returns A valid writer handle or an error code.
 *
 * @returns >0
 *              A valid writer handle.
 * @returns DDS_RETCODE_ERROR
 *              An internal error occurred.
 *
 * DOC_TODO: Complete list of error codes
 */
DDS_EXPORT dds_entity_t
dds_create_writer(
  dds_entity_t participant_or_publisher,
  dds_entity_t topic,
  const dds_qos_t *qos,
  const dds_listener_t *listener);


/**
 * @defgroup writing (Writing data)
 * @ingroup writer
 * Writing data (and variants of it) is straightforward. The first set
 * is equivalent to the second set with -1 passed for "timestamp",
 * meaning, substitute the result of a call to time(). The dispose
 * and unregister operations take an object of the topic's type, but
 * only touch the key fields; the remained may be undefined.
*/

/**
 * @brief Registers an instance
 * @ingroup writing
 *
 * This operation registers an instance with a key value to the data writer and
 * returns an instance handle that could be used for successive write & dispose
 * operations. When the handle is not allocated, the function will return an
 * error and the handle will be un-touched.
 *
 * @param[in]  writer  The writer to which instance has be associated.
 * @param[out] handle  The instance handle.
 * @param[in]  data    The instance with the key value.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *            The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *            One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *            The operation is invoked on an inappropriate object.
 */
DDS_EXPORT dds_return_t
dds_register_instance(
  dds_entity_t writer,
  dds_instance_handle_t *handle,
  const void *data);

/**
 * @brief Unregisters an instance by instance
 * @ingroup writing
 *
 * This operation reverses the action of register instance, removes all information regarding
 * the instance and unregisters an instance with a key value from the data writer.
 *
 * @param[in]  writer  The writer to which instance is associated.
 * @param[in]  data    The instance with the key value.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 */
DDS_EXPORT dds_return_t
dds_unregister_instance(dds_entity_t writer, const void *data);

/**
 * @brief Unregisters an instance by instance handle
 * @ingroup writing
 *
 * This operation unregisters the instance which is identified by the key fields of the given
 * typed instance handle.
 *
 * @param[in]  writer  The writer to which instance is associated.
 * @param[in]  handle  The instance handle.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 */
DDS_EXPORT dds_return_t
dds_unregister_instance_ih(dds_entity_t writer, dds_instance_handle_t handle);

/**
 * @brief Unregisters an instance by instance with timestamp
 * @ingroup writing
 *
 * This operation reverses the action of register instance, removes all information regarding
 * the instance and unregisters an instance with a key value from the data writer. It also
 * provides a value for the timestamp explicitly.
 *
 * @param[in]  writer    The writer to which instance is associated.
 * @param[in]  data      The instance with the key value.
 * @param[in]  timestamp The timestamp used at registration.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 */
DDS_EXPORT dds_return_t
dds_unregister_instance_ts(
  dds_entity_t writer,
  const void *data,
  dds_time_t timestamp);

/**
 * @brief Unregisters an instance by instance handle with timestamp
 * @ingroup writing
 *
 * This operation unregisters an instance with a key value from the handle. Instance can be identified
 * from instance handle. If an unregistered key ID is passed as an instance data, an error is logged and
 * not flagged as return value.
 *
 * @param[in]  writer    The writer to which instance is associated.
 * @param[in]  handle    The instance handle.
 * @param[in]  timestamp The timestamp used at registration.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object
 */
DDS_EXPORT dds_return_t
dds_unregister_instance_ih_ts(
  dds_entity_t writer,
  dds_instance_handle_t handle,
  dds_time_t timestamp);

/**
 * @brief This operation modifies and disposes a data instance.
 * @ingroup writing
 *
 * This operation requests the Data Distribution Service to modify the instance and
 * mark it for deletion. Copies of the instance and its corresponding samples, which are
 * stored in every connected reader and, dependent on the QoS policy settings (also in
 * the Transient and Persistent stores) will be modified and marked for deletion by
 * setting their dds_instance_state_t to DDS_IST_NOT_ALIVE_DISPOSED.
 *
 * @par Blocking
 * If the history QoS policy is set to DDS_HISTORY_KEEP_ALL, the
 * dds_writedispose operation on the writer may block if the modification
 * would cause data to be lost because one of the limits, specified in the
 * resource_limits QoS policy, to be exceeded. In case the synchronous
 * attribute value of the reliability Qos policy is set to true for
 * communicating writers and readers then the writer will wait until
 * all synchronous readers have acknowledged the data. Under these
 * circumstances, the max_blocking_time attribute of the reliability
 * QoS policy configures the maximum time the dds_writedispose operation
 * may block.
 * If max_blocking_time elapses before the writer is able to store the
 * modification without exceeding the limits and all expected acknowledgements
 * are received, the dds_writedispose operation will fail and returns
 * DDS_RETCODE_TIMEOUT.
 *
 * @param[in]  writer The writer to dispose the data instance from.
 * @param[in]  data   The data to be written and disposed.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The sample is written and the instance is marked for deletion.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             At least one of the arguments is invalid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_TIMEOUT
 *             Either the current action overflowed the available resources as
 *             specified by the combination of the reliability QoS policy,
 *             history QoS policy and resource_limits QoS policy, or the
 *             current action was waiting for data delivery acknowledgement
 *             by synchronous readers. This caused blocking of this operation,
 *             which could not be resolved before max_blocking_time of the
 *             reliability QoS policy elapsed.
 */
DDS_EXPORT dds_return_t
dds_writedispose(dds_entity_t writer, const void *data);

/**
 * @brief This operation modifies and disposes a data instance with a specific
 *        timestamp.
 * @ingroup writing
 *
 * This operation performs the same functions as dds_writedispose() except that
 * the application provides the value for the source_timestamp that is made
 * available to connected reader objects. This timestamp is important for the
 * interpretation of the destination_order QoS policy.
 *
 * @param[in]  writer    The writer to dispose the data instance from.
 * @param[in]  data      The data to be written and disposed.
 * @param[in]  timestamp The timestamp used as source timestamp.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The sample is written and the instance is marked for deletion.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             At least one of the arguments is invalid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_TIMEOUT
 *             Either the current action overflowed the available resources as
 *             specified by the combination of the reliability QoS policy,
 *             history QoS policy and resource_limits QoS policy, or the
 *             current action was waiting for data delivery acknowledgement
 *             by synchronous readers. This caused blocking of this operation,
 *             which could not be resolved before max_blocking_time of the
 *             reliability QoS policy elapsed.
 */
DDS_EXPORT dds_return_t
dds_writedispose_ts(
  dds_entity_t writer,
  const void *data,
  dds_time_t timestamp);

/**
 * @brief This operation disposes an instance, identified by the data sample.
 * @ingroup writing
 *
 * This operation requests the Data Distribution Service to modify the instance and
 * mark it for deletion. Copies of the instance and its corresponding samples, which are
 * stored in every connected reader and, dependent on the QoS policy settings (also in
 * the Transient and Persistent stores) will be modified and marked for deletion by
 * setting their dds_instance_state_t to DDS_IST_NOT_ALIVE_DISPOSED.
 *
 * @par Blocking
 * If the history QoS policy is set to DDS_HISTORY_KEEP_ALL, the
 * dds_writedispose operation on the writer may block if the modification
 * would cause data to be lost because one of the limits, specified in the
 * resource_limits QoS policy, to be exceeded. In case the synchronous
 * attribute value of the reliability Qos policy is set to true for
 * communicating writers and readers then the writer will wait until
 * all synchronous readers have acknowledged the data. Under these
 * circumstances, the max_blocking_time attribute of the reliability
 * QoS policy configures the maximum time the dds_writedispose operation
 * may block.
 * If max_blocking_time elapses before the writer is able to store the
 * modification without exceeding the limits and all expected acknowledgements
 * are received, the dds_writedispose operation will fail and returns
 * DDS_RETCODE_TIMEOUT.
 *
 * @param[in]  writer The writer to dispose the data instance from.
 * @param[in]  data   The data sample that identifies the instance
 *                    to be disposed.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The sample is written and the instance is marked for deletion.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             At least one of the arguments is invalid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_TIMEOUT
 *             Either the current action overflowed the available resources as
 *             specified by the combination of the reliability QoS policy,
 *             history QoS policy and resource_limits QoS policy, or the
 *             current action was waiting for data delivery acknowledgement
 *             by synchronous readers. This caused blocking of this operation,
 *             which could not be resolved before max_blocking_time of the
 *             reliability QoS policy elapsed.
 */
DDS_EXPORT dds_return_t
dds_dispose(dds_entity_t writer, const void *data);

/**
 * @brief This operation disposes an instance with a specific timestamp, identified by the data sample.
 * @ingroup writing
 *
 * This operation performs the same functions as dds_dispose() except that
 * the application provides the value for the source_timestamp that is made
 * available to connected reader objects. This timestamp is important for the
 * interpretation of the destination_order QoS policy.
 *
 * @param[in]  writer    The writer to dispose the data instance from.
 * @param[in]  data      The data sample that identifies the instance
 *                       to be disposed.
 * @param[in]  timestamp The timestamp used as source timestamp.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The sample is written and the instance is marked for deletion
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             At least one of the arguments is invalid
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted
 * @retval DDS_RETCODE_TIMEOUT
 *             Either the current action overflowed the available resources as
 *             specified by the combination of the reliability QoS policy,
 *             history QoS policy and resource_limits QoS policy, or the
 *             current action was waiting for data delivery acknowledgment
 *             by synchronous readers. This caused blocking of this operation,
 *             which could not be resolved before max_blocking_time of the
 *             reliability QoS policy elapsed.
 */
DDS_EXPORT dds_return_t
dds_dispose_ts(
  dds_entity_t writer,
  const void *data,
  dds_time_t timestamp);

/**
 * @brief This operation disposes an instance, identified by the instance handle.
 * @ingroup writing
 *
 * This operation requests the Data Distribution Service to modify the instance and
 * mark it for deletion. Copies of the instance and its corresponding samples, which are
 * stored in every connected reader and, dependent on the QoS policy settings (also in
 * the Transient and Persistent stores) will be modified and marked for deletion by
 * setting their dds_instance_state_t to DDS_IST_NOT_ALIVE_DISPOSED.
 *
 * @par Instance Handle
 * The given instance handle must correspond to the value that was returned by either
 * the dds_register_instance operation, dds_register_instance_ts or dds_lookup_instance.
 * If there is no correspondence, then the result of the operation is unspecified.
 *
 * @param[in]  writer The writer to dispose the data instance from.
 * @param[in]  handle The handle to identify an instance.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The sample is written and the instance is marked for deletion.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             At least one of the arguments is invalid
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this writer
 */
DDS_EXPORT dds_return_t
dds_dispose_ih(dds_entity_t writer, dds_instance_handle_t handle);

/**
 * @brief This operation disposes an instance with a specific timestamp, identified by the instance handle.
 * @ingroup writing
 *
 * This operation performs the same functions as dds_dispose_ih() except that
 * the application provides the value for the source_timestamp that is made
 * available to connected reader objects. This timestamp is important for the
 * interpretation of the destination_order QoS policy.
 *
 * @param[in]  writer    The writer to dispose the data instance from.
 * @param[in]  handle    The handle to identify an instance.
 * @param[in]  timestamp The timestamp used as source timestamp.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The sample is written and the instance is marked for deletion.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             At least one of the arguments is invalid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this writer.
 */
DDS_EXPORT dds_return_t
dds_dispose_ih_ts(
  dds_entity_t writer,
  dds_instance_handle_t handle,
  dds_time_t timestamp);

/**
 * @brief Write the value of a data instance
 * @ingroup writing
 *
 * With this API, the value of the source timestamp is automatically made
 * available to the data reader by the service.
 *
 * @param[in]  writer The writer entity.
 * @param[in]  data Value to be written.
 *
 * @returns dds_return_t indicating success or failure.
 */
DDS_EXPORT dds_return_t
dds_write(dds_entity_t writer, const void *data);

/**
 * @brief Flush a writers batched writes
 * @ingroup writing
 *
 * When using the WriteBatch mode you can manually batch small writes into larger
 * datapackets for network efficiency. The normal dds_write() calls will no longer
 * automatically decide when to send data, you will do that manually using this function.
 *
 * DOC_TODO check if my assumptions about how this function works are correct
 *
 * @param[in]  writer The writer entity.
 */
DDS_EXPORT void
dds_write_flush(dds_entity_t writer);

/**
 * @brief Write a serialized value of a data instance
 * @ingroup writing
 *
 * This call causes the writer to write the serialized value that is provided
 * in the serdata argument.  Timestamp and statusinfo fields are set to the
 * current time and 0 (indicating a regular write), respectively.
 *
 * @param[in]  writer The writer entity.
 * @param[in]  serdata Serialized value to be written.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The writer successfully wrote the serialized value.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_TIMEOUT
 *             The writer failed to write the serialized value reliably within the specified max_blocking_time.
 */
DDS_EXPORT dds_return_t
dds_writecdr(dds_entity_t writer, struct ddsi_serdata *serdata);

/**
 * @brief Write a serialized value of a data instance
 * @ingroup writing
 *
 * This call causes the writer to write the serialized value that is provided
 * in the serdata argument.  Timestamp and statusinfo are used as is.
 *
 * @param[in]  writer The writer entity.
 * @param[in]  serdata Serialized value to be written.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The writer successfully wrote the serialized value.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_TIMEOUT
 *             The writer failed to write the serialized value reliably within the specified max_blocking_time.
 */
DDS_EXPORT dds_return_t
dds_forwardcdr(dds_entity_t writer, struct ddsi_serdata *serdata);

/**
 * @brief Write the value of a data instance along with the source timestamp passed.
 * @ingroup writing
 *
 * @param[in]  writer The writer entity.
 * @param[in]  data Value to be written.
 * @param[in]  timestamp Source timestamp.
 *
 * @returns A dds_return_t indicating success or failure.
 */
DDS_EXPORT dds_return_t
dds_write_ts(
  dds_entity_t writer,
  const void *data,
  dds_time_t timestamp);

/**
 * @defgroup readcondition (ReadCondition)
 * @ingroup condition
 */
/**
 * @defgroup querycondition (QueryCondition)
 * @ingroup condition
 */
/**
 * @defgroup guardcondition (GuardCondition)
 * @ingroup condition
 */

/**
 * @brief Creates a readcondition associated to the given reader.
 * @ingroup readcondition
 *
 * The readcondition allows specifying which samples are of interest in
 * a data reader's history, by means of a mask. The mask is or'd with
 * the flags that are dds_sample_state_t, dds_view_state_t and
 * dds_instance_state_t.
 *
 * Based on the mask value set, the readcondition gets triggered when
 * data is available on the reader.
 *
 * Waitsets allow waiting for an event on some of any set of entities.
 * This means that the readcondition can be used to wake up a waitset when
 * data is in the reader history with states that matches the given mask.
 *
 * @note The parent reader and every of its associated conditions (whether
 *       they are readconditions or queryconditions) share the same resources.
 *       This means that one of these entities reads or takes data, the states
 *       of the data will change for other entities automatically. For instance,
 *       if one reads a sample, then the sample state will become 'read' for all
 *       associated reader/conditions. Or if one takes a sample, then it's not
 *       available to any other associated reader/condition.
 *
 * @param[in]  reader  Reader to associate the condition to.
 * @param[in]  mask    Interest (dds_sample_state_t|dds_view_state_t|dds_instance_state_t).
 *
 * @returns A valid condition handle or an error code.
 *
 * @retval >0
 *             A valid condition handle
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_entity_t
dds_create_readcondition(dds_entity_t reader, uint32_t mask);

/**
 * @brief Function signature for a querycondition filter
 * @ingroup querycondition
 */
typedef bool (*dds_querycondition_filter_fn) (const void * sample);

/**
 * @brief Creates a queryondition associated to the given reader.
 * @ingroup querycondition
 *
 * The queryondition allows specifying which samples are of interest in
 * a data reader's history, by means of a mask and a filter. The mask is
 * or'd with the flags that are dds_sample_state_t, dds_view_state_t and
 * dds_instance_state_t.
 *
 * Based on the mask value set and data that matches the filter, the
 * querycondition gets triggered when data is available on the reader.
 *
 * Waitsets allow waiting for an event on some of any set of entities.
 * This means that the querycondition can be used to wake up a waitset when
 * data is in the reader history with states that matches the given mask
 * and filter.
 *
 * @note The parent reader and every of its associated conditions (whether
 *       they are readconditions or queryconditions) share the same resources.
 *       This means that one of these entities reads or takes data, the states
 *       of the data will change for other entities automatically. For instance,
 *       if one reads a sample, then the sample state will become 'read' for all
 *       associated reader/conditions. Or if one takes a sample, then it's not
 *       available to any other associated reader/condition.
 *
 * @param[in]  reader  Reader to associate the condition to.
 * @param[in]  mask    Interest (dds_sample_state_t|dds_view_state_t|dds_instance_state_t).
 * @param[in]  filter  Callback that the application can use to filter specific samples.
 *
 * @returns A valid condition handle or an error code
 *
 * @retval >=0
 *             A valid condition handle.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_entity_t
dds_create_querycondition(
  dds_entity_t reader,
  uint32_t mask,
  dds_querycondition_filter_fn filter);

/**
 * @brief Creates a guardcondition.
 * @ingroup guardcondition
 *
 * Waitsets allow waiting for an event on some of any set of entities.
 * This means that the guardcondition can be used to wake up a waitset when
 * data is in the reader history with states that matches the given mask.
 *
 * @param[in]   participant  Participant on which to create the guardcondition.
 *
 * @returns A valid condition handle or an error code.
 *
 * @retval >0
 *             A valid condition handle
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_entity_t
dds_create_guardcondition(dds_entity_t participant);

/**
 * @brief Sets the trigger status of a guardcondition.
 * @ingroup guardcondition
 *
 * @param[in]   guardcond  Guard condition to set the trigger status of.
 * @param[in]   triggered  The triggered status to set.
 *
 * @retval DDS_RETCODE_OK
 *             Operation successful
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_set_guardcondition(dds_entity_t guardcond, bool triggered);

/**
 * @brief Reads the trigger status of a guardcondition.
 * @ingroup guardcondition
 *
 * @param[in]   guardcond  Guard condition to read the trigger status of.
 * @param[out]  triggered  The triggered status read from the guard condition.
 *
 * @retval DDS_RETCODE_OK
 *             Operation successful
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_read_guardcondition(dds_entity_t guardcond, bool *triggered);

/**
 * @brief Reads and resets the trigger status of a guardcondition.
 * @ingroup guardcondition
 *
 * @param[in]   guardcond  Guard condition to read and reset the trigger status of.
 * @param[out]  triggered  The triggered status read from the guard condition.
 *
 * @retval DDS_RETCODE_OK
 *             Operation successful
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_take_guardcondition(dds_entity_t guardcond, bool *triggered);

/**
 * @defgroup waitset (WaitSet)
 * @ingroup dds
 */

/**
 * @brief Waitset attachment argument.
 * @ingroup waitset
 *
 * Every entity that is attached to the waitset can be accompanied by such
 * an attachment argument. When the waitset wait is unblocked because of an
 * entity that triggered, then the returning array will be populated with
 * these attachment arguments that are related to the triggered entity.
 */
typedef intptr_t dds_attach_t;

/**
 * @brief Create a waitset and allocate the resources required
 * @ingroup waitset
 *
 * A WaitSet object allows an application to wait until one or more of the
 * conditions of the attached entities evaluates to TRUE or until the timeout
 * expires.
 *
 * @param[in]  participant  Domain participant which the WaitSet contains.
 *
 * @returns A valid waitset handle or an error code.
 *
 * @retval >=0
 *             A valid waitset handle.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_entity_t
dds_create_waitset(dds_entity_t participant);

/**
 * @brief Acquire previously attached entities.
 * @ingroup waitset
 *
 * This functions takes a pre-allocated list to put the entities in and
 * will return the number of found entities. It is possible that the given
 * size of the list is not the same as the number of found entities. If
 * less entities are found, then the last few entries in the list are
 * untouched. When more entities are found, then only 'size' number of
 * entries are inserted into the list, but still the complete count of the
 * found entities is returned. Which entities are returned in the latter
 * case is undefined.
 *
 * @param[in]  waitset  Waitset from which to get its attached entities.
 * @param[out] entities Pre-allocated array to contain the found entities.
 * @param[in]  size     Size of the pre-allocated entities' list.
 *
 * @returns A dds_return_t with the number of children or an error code.
 *
 * @retval >=0
 *             Number of children found (can be larger than 'size').
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entities parameter is NULL, while a size is provided.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The waitset has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_waitset_get_entities(
  dds_entity_t waitset,
  dds_entity_t *entities,
  size_t size);

/**
 * @brief This operation attaches an Entity to the WaitSet.
 * @ingroup waitset
 *
 * This operation attaches an Entity to the WaitSet. The dds_waitset_wait()
 * will block when none of the attached entities are triggered. 'Triggered'
 * (dds_triggered()) doesn't mean the same for every entity:
 *  - Reader/Writer/Publisher/Subscriber/Topic/Participant
 *      - These are triggered when their status changed.
 *  - WaitSet
 *      - Triggered when trigger value was set to true by the application.
 *        It stays triggered until application sets the trigger value to
 *        false (dds_waitset_set_trigger()). This can be used to wake up an
 *        waitset for different reasons (f.i. termination) than the 'normal'
 *        status change (like new data).
 *  - ReadCondition/QueryCondition
 *      - Triggered when data is available on the related Reader that matches
 *        the Condition.
 *
 * Multiple entities can be attached to a single waitset. A particular entity
 * can be attached to multiple waitsets. However, a particular entity can not
 * be attached to a particular waitset multiple times.
 *
 * @param[in]  waitset  The waitset to attach the given entity to.
 * @param[in]  entity   The entity to attach.
 * @param[in]  x        Blob that will be supplied when the waitset wait is
 *                      triggered by the given entity.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Entity attached.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The given waitset or entity are not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The waitset has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The entity was already attached.
 */
DDS_EXPORT dds_return_t
dds_waitset_attach(
  dds_entity_t waitset,
  dds_entity_t entity,
  dds_attach_t x);

/**
 * @brief This operation detaches an Entity from the WaitSet.
 * @ingroup waitset
 *
 * @param[in]  waitset  The waitset to detach the given entity from.
 * @param[in]  entity   The entity to detach.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Entity detached.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The given waitset or entity are not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The waitset has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The entity is not attached.
 */
DDS_EXPORT dds_return_t
dds_waitset_detach(
  dds_entity_t waitset,
  dds_entity_t entity);

/**
 * @brief Sets the trigger_value associated with a waitset.
 * @ingroup waitset
 *
 * When the waitset is attached to itself and the trigger value is
 * set to 'true', then the waitset will wake up just like with an
 * other status change of the attached entities.
 *
 * This can be used to forcefully wake up a waitset, for instance
 * when the application wants to shut down. So, when the trigger
 * value is true, the waitset will wake up or not wait at all.
 *
 * The trigger value will remain true until the application sets it
 * false again deliberately.
 *
 * @param[in]  waitset  The waitset to set the trigger value on.
 * @param[in]  trigger  The trigger value to set.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Trigger value set.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The given waitset is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The waitset has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_waitset_set_trigger(
  dds_entity_t waitset,
  bool trigger);

/**
 * @brief This operation allows an application thread to wait for the a status
 *        change or other trigger on (one of) the entities that are attached to
 *        the WaitSet.
 * @ingroup waitset
 *
 * The dds_waitset_wait() operation blocks until the some of the attached
 * entities have triggered or "reltimeout" has elapsed.
 * 'Triggered' (dds_triggered()) doesn't mean the same for every entity:
 *  - Reader/Writer/Publisher/Subscriber/Topic/Participant
 *      - These are triggered when their status changed.
 *  - WaitSet
 *      - Triggered when trigger value was set to true by the application.
 *        It stays triggered until application sets the trigger value to
 *        false (dds_waitset_set_trigger()). This can be used to wake up an
 *        waitset for different reasons (f.i. termination) than the 'normal'
 *        status change (like new data).
 *  - ReadCondition/QueryCondition
 *      - Triggered when data is available on the related Reader that matches
 *        the Condition.
 *
 * This functions takes a pre-allocated list to put the "xs" blobs in (that
 * were provided during the attach of the related entities) and will return
 * the number of triggered entities. It is possible that the given size
 * of the list is not the same as the number of triggered entities. If less
 * entities were triggered, then the last few entries in the list are
 * untouched. When more entities are triggered, then only 'size' number of
 * entries are inserted into the list, but still the complete count of the
 * triggered entities is returned. Which "xs" blobs are returned in the
 * latter case is undefined.
 *
 * In case of a time out, the return value is 0.
 *
 * Deleting the waitset while the application is blocked results in an
 * error code (i.e. < 0) returned by "wait".
 *
 * Multiple threads may block on a single waitset at the same time;
 * the calls are entirely independent.
 *
 * An empty waitset never triggers (i.e., dds_waitset_wait on an empty
 * waitset is essentially equivalent to a sleep).
 *
 * The "dds_waitset_wait_until" operation is the same as the
 * "dds_waitset_wait" except that it takes an absolute timeout.
 *
 * @param[in]  waitset    The waitset to set the trigger value on.
 * @param[out] xs         Pre-allocated list to store the 'blobs' that were
 *                        provided during the attach of the triggered entities.
 * @param[in]  nxs        The size of the pre-allocated blobs list.
 * @param[in]  reltimeout Relative timeout
 *
 * @returns A dds_return_t with the number of entities triggered or an error code
 *
 * @retval >0
 *             Number of entities triggered.
 * @retval  0
 *             Time out (no entities were triggered).
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The given waitset is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The waitset has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_waitset_wait(
  dds_entity_t waitset,
  dds_attach_t *xs,
  size_t nxs,
  dds_duration_t reltimeout);

/**
 * @brief This operation allows an application thread to wait for the a status
 *        change or other trigger on (one of) the entities that are attached to
 *        the WaitSet.
 * @ingroup waitset
 *
 * The dds_waitset_wait() operation blocks until the some of the attached
 * entities have triggered or "abstimeout" has been reached.
 * 'Triggered' (dds_triggered()) doesn't mean the same for every entity:
 *  - Reader/Writer/Publisher/Subscriber/Topic/Participant
 *      - These are triggered when their status changed.
 *  - WaitSet
 *      - Triggered when trigger value was set to true by the application.
 *        It stays triggered until application sets the trigger value to
 *        false (dds_waitset_set_trigger()). This can be used to wake up an
 *        waitset for different reasons (f.i. termination) than the 'normal'
 *        status change (like new data).
 *  - ReadCondition/QueryCondition
 *      - Triggered when data is available on the related Reader that matches
 *        the Condition.
 *
 * This functions takes a pre-allocated list to put the "xs" blobs in (that
 * were provided during the attach of the related entities) and will return
 * the number of triggered entities. It is possible that the given size
 * of the list is not the same as the number of triggered entities. If less
 * entities were triggered, then the last few entries in the list are
 * untouched. When more entities are triggered, then only 'size' number of
 * entries are inserted into the list, but still the complete count of the
 * triggered entities is returned. Which "xs" blobs are returned in the
 * latter case is undefined.
 *
 * In case of a time out, the return value is 0.
 *
 * Deleting the waitset while the application is blocked results in an
 * error code (i.e. < 0) returned by "wait".
 *
 * Multiple threads may block on a single waitset at the same time;
 * the calls are entirely independent.
 *
 * An empty waitset never triggers (i.e., dds_waitset_wait on an empty
 * waitset is essentially equivalent to a sleep).
 *
 * The "dds_waitset_wait" operation is the same as the
 * "dds_waitset_wait_until" except that it takes an relative timeout.
 *
 * The "dds_waitset_wait" operation is the same as the "dds_wait"
 * except that it takes an absolute timeout.
 *
 * @param[in]  waitset    The waitset to set the trigger value on.
 * @param[out] xs         Pre-allocated list to store the 'blobs' that were
 *                        provided during the attach of the triggered entities.
 * @param[in]  nxs        The size of the pre-allocated blobs list.
 * @param[in]  abstimeout Absolute timeout
 *
 * @returns A dds_return_t with the number of entities triggered or an error code.
 *
 * @retval >0
 *             Number of entities triggered.
 * @retval  0
 *             Time out (no entities were triggered).
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The given waitset is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The waitset has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_waitset_wait_until(
  dds_entity_t waitset,
  dds_attach_t *xs,
  size_t nxs,
  dds_time_t abstimeout);

/**
 * @defgroup reading (Reading Data)
 * @ingroup reader
 * There are a number of ways to aquire data, divided into variations of "read" and "take".
 * The return value of a read/take operation is the number of elements returned. "max_samples"
 * should have the same type, as one can't return more than MAX_INT
 * this way, anyway. X, Y, CX, CY return to the various filtering
 * options, see the DCPS spec.
 *
 * O ::= read | take
 *
 * X             => CX
 * (empty)          (empty)
 * _next_instance   instance_handle_t prev
 *
 * Y             => CY
 * (empty)          uint32_t mask
 * _cond            cond_t cond -- refers to a read condition (or query if implemented)
 */

/**
 * @brief Access and read the collection of data values (of same type) and sample info from the
 *        data reader, readcondition or querycondition.
 * @ingroup reading
 *
 * Return value provides information about number of samples read, which will
 * be <= maxs. Based on the count, the buffer will contain data to be read only
 * when valid_data bit in sample info structure is set.
 * The buffer required for data values, could be allocated explicitly or can
 * use the memory from data reader to prevent copy. In the latter case, buffer and
 * sample_info should be returned back, once it is no longer using the Data.
 * Data values once read will remain in the buffer with the sample_state set to READ
 * and view_state set to NOT_NEW.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  bufsz The size of buffer provided.
 * @param[in]  maxs Maximum number of samples to read.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_read(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  size_t bufsz,
  uint32_t maxs);

/**
 * @brief Access and read loaned samples of data reader, readcondition or querycondition.
 * @ingroup reading
 *
 * After dds_read_wl function is being called and the data has been handled, dds_return_loan() function must be called to possibly free memory.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL)
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value
 * @param[in]  maxs Maximum number of samples to read
 *
 * @returns A dds_return_t with the number of samples read or an error code
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_read_wl(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  uint32_t maxs);

/**
 * @brief Read the collection of data values and sample info from the data reader, readcondition
 *        or querycondition based on mask.
 * @ingroup reading
 *
 * When using a readcondition or querycondition, their masks are or'd with the given mask.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  bufsz The size of buffer provided.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_read_mask(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  size_t bufsz,
  uint32_t maxs,
  uint32_t mask);

/**
 * @brief Access and read loaned samples of data reader, readcondition
 *        or querycondition based on mask
 * @ingroup reading
 *
 * When using a readcondition or querycondition, their masks are or'd with the given mask.
 *
 * After dds_read_mask_wl function is being called and the data has been handled, dds_return_loan() function must be called to possibly free memory
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_read_mask_wl(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  uint32_t maxs,
  uint32_t mask);

/**
 * @brief Access and read the collection of data values (of same type) and sample info from the
 *        data reader, readcondition or querycondition, coped by the provided instance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_read, except that only data scoped to
 * the provided instance handle is read.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  bufsz The size of buffer provided.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  handle Instance handle related to the samples to read.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_read_instance(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  size_t bufsz,
  uint32_t maxs,
  dds_instance_handle_t handle);

/**
 * @brief Access and read loaned samples of data reader, readcondition or querycondition,
 *        scoped by the provided instance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_read_wl, except that only data
 * scoped to the provided instance handle is read.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  handle Instance handle related to the samples to read.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_read_instance_wl(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  uint32_t maxs,
  dds_instance_handle_t handle);

/**
 * @brief Read the collection of data values and sample info from the data reader, readcondition
 *        or querycondition based on mask and scoped by the provided instance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_read_mask, except that only data
 * scoped to the provided instance handle is read.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  bufsz The size of buffer provided.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  handle Instance handle related to the samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_read_instance_mask(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  size_t bufsz,
  uint32_t maxs,
  dds_instance_handle_t handle,
  uint32_t mask);

/**
 * @brief Access and read loaned samples of data reader, readcondition or
 *        querycondition based on mask, scoped by the provided instance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_read_mask_wl, except that
 * only data scoped to the provided instance handle is read.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  handle Instance handle related to the samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_read_instance_mask_wl(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  uint32_t maxs,
  dds_instance_handle_t handle,
  uint32_t mask);

/**
 * @brief Access the collection of data values (of same type) and sample info from the
 *        data reader, readcondition or querycondition.
 * @ingroup reading
 *
 * Data value once read is removed from the Data Reader cannot to
 * 'read' or 'taken' again.
 * Return value provides information about number of samples read, which will
 * be <= maxs. Based on the count, the buffer will contain data to be read only
 * when valid_data bit in sample info structure is set.
 * The buffer required for data values, could be allocated explicitly or can
 * use the memory from data reader to prevent copy. In the latter case, buffer and
 * sample_info should be returned back, once it is no longer using the Data.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  bufsz The size of buffer provided.
 * @param[in]  maxs Maximum number of samples to read.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_take(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  size_t bufsz,
  uint32_t maxs);

/**
 * @brief Access loaned samples of data reader, readcondition or querycondition.
 * @ingroup reading
 *
 * After dds_take_wl function is being called and the data has been handled, dds_return_loan() function must be called to possibly free memory
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  maxs Maximum number of samples to read.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_take_wl(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  uint32_t maxs);

/**
 * @brief Take the collection of data values (of same type) and sample info from the
 *        data reader, readcondition or querycondition based on mask
 * @ingroup reading
 *
 * When using a readcondition or querycondition, their masks are or'd with the given mask.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  bufsz The size of buffer provided.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_take_mask(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  size_t bufsz,
  uint32_t maxs,
  uint32_t mask);

/**
 * @brief  Access loaned samples of data reader, readcondition or querycondition based on mask.
 * @ingroup reading
 *
 * When using a readcondition or querycondition, their masks are or'd with the given mask.
 *
 * After dds_take_mask_wl function is being called and the data has been handled, dds_return_loan() function must be called to possibly free memory
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_take_mask_wl(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  uint32_t maxs,
  uint32_t mask);

/**
 * @anchor DDS_HAS_READCDR
 * @ingroup reading
 * @brief Set when function dds_has_readcdr is defined.
 */
#define DDS_HAS_READCDR 1

/**
 * @brief Access the collection of serialized data values (of same type) and
 *        sample info from the data reader, readcondition or querycondition.
 * @ingroup reading
 *
 * This call accesses the serialized data from the data reader, readcondition or
 * querycondition and makes it available to the application. The serialized data
 * is made available through @ref ddsi_serdata structures. Returned samples are
 * marked as READ.
 *
 * Return value provides information about the number of samples read, which will
 * be <= maxs. Based on the count, the buffer will contain serialized data to be
 * read only when valid_data bit in sample info structure is set.
 * The buffer required for data values, could be allocated explicitly or can
 * use the memory from data reader to prevent copy. In the latter case, buffer and
 * sample_info should be returned back, once it is no longer using the data.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to @ref ddsi_serdata structures that contain
 *                 the serialized data. The pointers can be NULL.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The precondition for this operation is not met.
 */
DDS_EXPORT dds_return_t
dds_readcdr(
  dds_entity_t reader_or_condition,
  struct ddsi_serdata **buf,
  uint32_t maxs,
  dds_sample_info_t *si,
  uint32_t mask);

/**
 * @brief Access the collection of serialized data values (of same type) and
 *        sample info from the data reader, readcondition or querycondition
 *        scoped by the provided instance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_read_instance_wl, except that
 * samples are now in their serialized form. The serialized data is made available through
 * @ref ddsi_serdata structures. Returned samples are marked as READ.
 *
 * Return value provides information about the number of samples read, which will
 * be <= maxs. Based on the count, the buffer will contain serialized data to be
 * read only when valid_data bit in sample info structure is set.
 * The buffer required for data values, could be allocated explicitly or can
 * use the memory from data reader to prevent copy. In the latter case, buffer and
 * sample_info should be returned back, once it is no longer using the data.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to @ref ddsi_serdata structures that contain
 *                 the serialized data. The pointers can be NULL.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  handle Instance handle related to the samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_readcdr_instance (
    dds_entity_t reader_or_condition,
    struct ddsi_serdata **buf,
    uint32_t maxs,
    dds_sample_info_t *si,
    dds_instance_handle_t handle,
    uint32_t mask);

/**
 * @brief Access the collection of serialized data values (of same type) and
 *        sample info from the data reader, readcondition or querycondition.
 * @ingroup reading
 *
 * This call accesses the serialized data from the data reader, readcondition or
 * querycondition and makes it available to the application. The serialized data
 * is made available through @ref ddsi_serdata structures. Once read the data is
 * removed from the reader and cannot be 'read' or 'taken' again.
 *
 * Return value provides information about the number of samples read, which will
 * be <= maxs. Based on the count, the buffer will contain serialized data to be
 * read only when valid_data bit in sample info structure is set.
 * The buffer required for data values, could be allocated explicitly or can
 * use the memory from data reader to prevent copy. In the latter case, buffer and
 * sample_info should be returned back, once it is no longer using the data.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to @ref ddsi_serdata structures that contain
 *                 the serialized data. The pointers can be NULL.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The precondition for this operation is not met.
 */
DDS_EXPORT dds_return_t
dds_takecdr(
  dds_entity_t reader_or_condition,
  struct ddsi_serdata **buf,
  uint32_t maxs,
  dds_sample_info_t *si,
  uint32_t mask);

/**
 * @brief Access the collection of serialized data values (of same type) and
 *        sample info from the data reader, readcondition or querycondition
 *        scoped by the provided instance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_take_instance_wl, except that
 * samples are now in their serialized form. The serialized data is made available through
 * @ref ddsi_serdata structures. Returned samples are marked as READ.
 *
 * Return value provides information about the number of samples read, which will
 * be <= maxs. Based on the count, the buffer will contain serialized data to be
 * read only when valid_data bit in sample info structure is set.
 * The buffer required for data values, could be allocated explicitly or can
 * use the memory from data reader to prevent copy. In the latter case, buffer and
 * sample_info should be returned back, once it is no longer using the data.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to @ref ddsi_serdata structures that contain
 *                 the serialized data. The pointers can be NULL.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  handle Instance handle related to the samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_takecdr_instance (
    dds_entity_t reader_or_condition,
    struct ddsi_serdata **buf,
    uint32_t maxs,
    dds_sample_info_t *si,
    dds_instance_handle_t handle,
    uint32_t mask);


/**
 * @brief Access the collection of data values (of same type) and sample info from the
 *        data reader, readcondition or querycondition but scoped by the given
 *        instance handle.
 * @ingroup reading
 *
 * This operation mplements the same functionality as dds_take, except that only data
 * scoped to the provided instance handle is taken.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  bufsz The size of buffer provided.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  handle Instance handle related to the samples to read.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_take_instance(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  size_t bufsz,
  uint32_t maxs,
  dds_instance_handle_t handle);

/**
 * @brief Access loaned samples of data reader, readcondition or querycondition,
 *        scoped by the given instance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_take_wl, except that
 * only data scoped to the provided instance handle is read.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  handle Instance handle related to the samples to read.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_take_instance_wl(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  uint32_t maxs,
  dds_instance_handle_t handle);

/**
 * @brief Take the collection of data values (of same type) and sample info from the
 *        data reader, readcondition or querycondition based on mask and scoped
 *        by the given instance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_take_mask, except that only
 * data scoped to the provided instance handle is read.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  bufsz The size of buffer provided.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  handle Instance handle related to the samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples read or an error code.
 *
 * @retval >=0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_take_instance_mask(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  size_t bufsz,
  uint32_t maxs,
  dds_instance_handle_t handle,
  uint32_t mask);

/**
 * @brief  Access loaned samples of data reader, readcondition or querycondition based
 *         on mask and scoped by the given intance handle.
 * @ingroup reading
 *
 * This operation implements the same functionality as dds_take_mask_wl, except that
 * only data scoped to the provided instance handle is read.
 *
 * @param[in]  reader_or_condition Reader, readcondition or querycondition entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si Pointer to an array of @ref dds_sample_info_t returned for each data value.
 * @param[in]  maxs Maximum number of samples to read.
 * @param[in]  handle Instance handle related to the samples to read.
 * @param[in]  mask Filter the data based on dds_sample_state_t|dds_view_state_t|dds_instance_state_t.
 *
 * @returns A dds_return_t with the number of samples or an error code.
 *
 * @retval >= 0
 *             Number of samples read.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the given arguments is not valid.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             The instance handle has not been registered with this reader.
 */
DDS_EXPORT dds_return_t
dds_take_instance_mask_wl(
  dds_entity_t reader_or_condition,
  void **buf,
  dds_sample_info_t *si,
  uint32_t maxs,
  dds_instance_handle_t handle,
  uint32_t mask);

/**
 * @brief Read, copy and remove the status set for the entity
 * @ingroup reading
 *
 * This operation copies the next, non-previously accessed
 * data value and corresponding sample info and removes from
 * the data reader. As an entity, only reader is accepted.
 *
 * The read/take next functions return a single sample. The returned sample
 * has a sample state of NOT_READ, a view state of ANY_VIEW_STATE and an
 * instance state of ANY_INSTANCE_STATE.
 *
 * @param[in]  reader The reader entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si The pointer to @ref dds_sample_info_t returned for a data value.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_take_next(
  dds_entity_t reader,
  void **buf,
  dds_sample_info_t *si);

/**
 * @brief Read, copy and remove the status set for the entity
 * @ingroup reading
 *
 * This operation copies the next, non-previously accessed
 * data value and corresponding sample info and removes from
 * the data reader. As an entity, only reader is accepted.
 *
 * The read/take next functions return a single sample. The returned sample
 * has a sample state of NOT_READ, a view state of ANY_VIEW_STATE and an
 * instance state of ANY_INSTANCE_STATE.
 *
 * After dds_take_next_wl function is being called and the data has been handled,
 * dds_return_loan() function must be called to possibly free memory.
 *
 * @param[in]  reader The reader entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si The pointer to @ref dds_sample_info_t returned for a data value.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_take_next_wl(
  dds_entity_t reader,
  void **buf,
  dds_sample_info_t *si);

/**
 * @brief Read and copy the status set for the entity
 * @ingroup reading
 *
 * This operation copies the next, non-previously accessed
 * data value and corresponding sample info. As an entity,
 * only reader is accepted.
 *
 * The read/take next functions return a single sample. The returned sample
 * has a sample state of NOT_READ, a view state of ANY_VIEW_STATE and an
 * instance state of ANY_INSTANCE_STATE.
 *
 * @param[in]  reader The reader entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si The pointer to @ref dds_sample_info_t returned for a data value.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_read_next(
  dds_entity_t reader,
  void **buf,
  dds_sample_info_t *si);

/**
 * @brief Read and copy the status set for the loaned sample
 * @ingroup reading
 *
 * This operation copies the next, non-previously accessed
 * data value and corresponding loaned sample info. As an entity,
 * only reader is accepted.
 *
 * The read/take next functions return a single sample. The returned sample
 * has a sample state of NOT_READ, a view state of ANY_VIEW_STATE and an
 * instance state of ANY_INSTANCE_STATE.
 *
 * After dds_read_next_wl function is being called and the data has been handled,
 * dds_return_loan() function must be called to possibly free memory.
 *
 * @param[in]  reader The reader entity.
 * @param[out] buf An array of pointers to samples into which data is read (pointers can be NULL).
 * @param[out] si The pointer to @ref dds_sample_info_t returned for a data value.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_read_next_wl(
  dds_entity_t reader,
  void **buf,
  dds_sample_info_t *si);


/**
 * @defgroup loan (Loans API)
 * @ingroup dds
 */

/**
 * @brief Return loaned samples to a reader or writer
 * @ingroup loan
 *
 * Used to release sample buffers returned by a read/take operation (a reader-loan)
 * or, in case shared memory is enabled, of the loan_sample operation (a writer-loan).
 *
 * When the application provides an empty buffer to a reader-loan, memory is allocated and
 * managed by DDS. By calling dds_return_loan(), the reader-loan is released so that the buffer
 * can be reused during a successive read/take operation. When a condition is provided, the
 * reader to which the condition belongs is looked up.
 *
 * Writer-loans are normally released implicitly when writing a loaned sample, but you can
 * cancel a writer-loan prematurely by invoking the return_loan() operation. For writer loans, buf is
 * overwritten with null pointers for all successfully returned entries. Any failure causes it to abort,
 * possibly midway through buf.
 *
 * @param[in] entity The entity that the loan belongs to.
 * @param[in,out] buf An array of (pointers to) samples, some or all of which will be set to null pointers.
 * @param[in] bufsz The number of (pointers to) samples stored in buf.
 *
 * @returns A dds_return_t indicating success or failure
 * @retval DDS_RETCODE_OK
 *             - the operation was successful; for a writer loan, all entries in buf are set to null
 *             - this specifically includes cases where bufsz <= 0 while entity is valid
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             - the entity parameter is not a valid parameter
 *             - buf is null, or bufsz > 0 and buf[0] = null
 *             - (for writer loans) buf[0 <= i < bufsz] is null; operation is aborted, all buf[j < i] = null on return
 * @retval DDS_RETCODE_PRECONDITION_NOT_MET
 *             - (for reader loans) buf was already returned (not guaranteed to be detected)
 *             - (for writer loans) buf[0 <= i < bufsz] does not correspond to an outstanding loan, all buf[j < i] = null on return
 * @retval DDS_RETCODE_UNSUPPORTED
 *             - (for writer loans) invoked on a writer not supporting loans.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             - the operation is invoked on an inappropriate object.
 */
DDS_EXPORT dds_return_t
dds_return_loan(
  dds_entity_t entity,
  void **buf,
  int32_t bufsz);

/**
 * @defgroup instance_handle (Instance Handles)
 * @ingroup dds
 * Instance handle <=> key value mapping.
 * Functions exactly as read w.r.t. treatment of data
 * parameter. On output, only key values set.
 * @code{c}
 * T x = { ... };
 * T y;
 * dds_instance_handle_t ih;
 * ih = dds_lookup_instance (e, &x);
 * dds_instance_get_key (e, ih, &y);
 * @endcode
*/

/**
 * @brief This operation takes a sample and returns an instance handle to be used for subsequent operations.
 * @ingroup instance_handle
 *
 * @param[in]  entity Reader or Writer entity.
 * @param[in]  data   Sample with a key fields set.
 *
 * @returns instance handle or DDS_HANDLE_NIL if instance could not be found from key.
 */
DDS_EXPORT dds_instance_handle_t
dds_lookup_instance(dds_entity_t entity, const void *data);

/**
 * @deprecated Get enabled status on entity. Use \ref dds_lookup_instance instead.
 * @ingroup instance_handle
 *
 * @param[in]  entity Reader or Writer entity.
 * @param[in]  data   Sample with a key fields set.
 *
 * @returns instance handle or DDS_HANDLE_NIL if instance could not be found from key.
 */
DDS_DEPRECATED_EXPORT dds_instance_handle_t
dds_instance_lookup(dds_entity_t entity, const void *data);

/**
 * @brief This operation takes an instance handle and return a key-value corresponding to it.
 * @ingroup instance_handle
 *
 * @param[in]  entity Reader, writer, readcondition or querycondition entity.
 * @param[in]  inst   Instance handle.
 * @param[out] data   pointer to an instance, to which the key ID corresponding to the instance handle will be
 *    returned, the sample in the instance should be ignored.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One of the parameters was invalid or the topic does not exist.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 *
 * DOC_TODO: Check return codes for completeness
 */
DDS_EXPORT dds_return_t
dds_instance_get_key(
  dds_entity_t entity,
  dds_instance_handle_t inst,
  void *data);

/**
 * @brief Begin coherent publishing or begin accessing a coherent set in a subscriber
 * @ingroup publication
 *
 * Invoking on a Writer or Reader behaves as if dds_begin_coherent was invoked on its parent
 * Publisher or Subscriber respectively.
 *
 * @param[in]  entity The entity that is prepared for coherent access.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_ERROR
 *             An internal error has occurred.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The provided entity is invalid or not supported.
 */
DDS_EXPORT dds_return_t
dds_begin_coherent(dds_entity_t entity);

/**
 * @brief End coherent publishing or end accessing a coherent set in a subscriber
 * @ingroup publication
 *
 * Invoking on a Writer or Reader behaves as if dds_end_coherent was invoked on its parent
 * Publisher or Subscriber respectively.
 *
 * @param[in] entity The entity on which coherent access is finished.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The provided entity is invalid or not supported.
 */
DDS_EXPORT dds_return_t
dds_end_coherent(dds_entity_t entity);

/**
 * @brief Trigger DATA_AVAILABLE event on contained readers
 * @ingroup subscriber
 *
 * The DATA_AVAILABLE event is broadcast to all readers owned by this subscriber that currently
 * have new data available. Any on_data_available listener callbacks attached to respective
 * readers are invoked.
 *
 * @param[in] subscriber A valid subscriber handle.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The provided subscriber is invalid.
 */
DDS_EXPORT dds_return_t
dds_notify_readers(dds_entity_t subscriber);

/**
 * @brief Checks whether the entity has one of its enabled statuses triggered.
 * @ingroup entity
 *
 * @param[in]  entity  Entity for which to check for triggered status.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_return_t
dds_triggered(dds_entity_t entity);

/**
 * @brief Get the topic
 * @ingroup entity
 *
 * This operation returns a topic (handle) when the function call is done
 * with reader, writer, read condition or query condition. For instance, it
 * will return the topic when it is used for creating the reader or writer.
 * For the conditions, it returns the topic that is used for creating the reader
 * which was used to create the condition.
 *
 * @param[in] entity The entity.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_ALREADY_DELETED
 *             The entity has already been deleted.
 */
DDS_EXPORT dds_entity_t
dds_get_topic(dds_entity_t entity);

/**
 * @brief Get instance handles of the data readers matching a writer
 * @ingroup builtintopic
 *
 * This operation fills the provided array with the instance handles
 * of the data readers that match the writer.  On successful output,
 * the number of entries of "rds" set is the minimum of the return
 * value and the value of "nrds".
 *
 * @param[in] writer   The writer.
 * @param[in] rds      The array to be filled.
 * @param[in] nrds     The size of the rds array, at most the first
 *                     nrds entries will be filled.  rds = NULL and nrds = 0
 *                     is a valid way of determining the number of matched
 *                     readers, but inefficient compared to relying on the
 *                     matched publication status.
 *
 * @returns A dds_return_t indicating the number of matched readers
 *             or failure.  The return value may be larger than nrds
 *             if there are more matching readers than the array can
 *             hold.
 *
 * @retval >=0
 *             The number of matching readers.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not valid or rds = NULL and
 *             nrds > 0.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 */
DDS_EXPORT dds_return_t
dds_get_matched_subscriptions (
  dds_entity_t writer,
  dds_instance_handle_t *rds,
  size_t nrds);

/**
 * @brief Get a description of a reader matched with the provided writer
 * @ingroup builtintopic
 *
 * This operation looks up the reader instance handle in the set of
 * readers matched with the specified writer, returning a freshly
 * allocated sample of the DCPSSubscription built-in topic if found,
 * and NULL if not.  The caller is responsible for freeing the
 * memory allocated, e.g. using dds_builtintopic_free_endpoint.
 *
 * This operation is similar to performing a read of the given
 * instance handle on a reader of the DCPSSubscription built-in
 * topic, but this operation additionally filters on whether the
 * reader is matched by the provided writer.
 *
 * @param[in] writer   The writer.
 * @param[in] ih       The instance handle of a reader.
 *
 * @returns A newly allocated sample containing the information on the
 *             reader, or a NULL pointer for any kind of failure.
 *
 * @retval != NULL
 *             The requested data
 * @retval NULL
 *             The writer is not valid or ih is not an instance handle
 *             of a matched reader.
 */
DDS_EXPORT dds_builtintopic_endpoint_t *
dds_get_matched_subscription_data (
  dds_entity_t writer,
  dds_instance_handle_t ih);

/**
 * @brief Get instance handles of the data writers matching a reader
 * @ingroup builtintopic
 *
 * This operation fills the provided array with the instance handles
 * of the data writers that match the reader.  On successful output,
 * the number of entries of "wrs" set is the minimum of the return
 * value and the value of "nwrs".
 *
 * @param[in] reader   The reader.
 * @param[in] wrs      The array to be filled.
 * @param[in] nwrs     The size of the wrs array, at most the first
 *             nwrs entries will be filled.  wrs = NULL and wrds = 0
 *             is a valid way of determining the number of matched
 *             readers, but inefficient compared to relying on the
 *             matched publication status.
 *
 * @returns A dds_return_t indicating the number of matched writers
 *             or failure.  The return value may be larger than nwrs
 *             if there are more matching writers than the array can
 *             hold.
 *
 * @retval >=0
 *             The number of matching writers.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not valid or wrs = NULL and
 *             nwrs > 0.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 */
DDS_EXPORT dds_return_t
dds_get_matched_publications (
  dds_entity_t reader,
  dds_instance_handle_t *wrs,
  size_t nwrs);

/**
 * @brief Get a description of a writer matched with the provided reader
 * @ingroup builtintopic
 *
 * This operation looks up the writer instance handle in the set of
 * writers matched with the specified reader, returning a freshly
 * allocated sample of the DCPSPublication built-in topic if found,
 * and NULL if not.  The caller is responsible for freeing the
 * memory allocated, e.g. using dds_builtintopic_free_endpoint.
 *
 * This operation is similar to performing a read of the given
 * instance handle on a reader of the DCPSPublication built-in
 * topic, but this operation additionally filters on whether the
 * writer is matched by the provided reader.
 *
 * @param[in] reader   The reader.
 * @param[in] ih       The instance handle of a writer.
 *
 * @returns A newly allocated sample containing the information on the
 *             writer, or a NULL pointer for any kind of failure.
 *
 * @retval != NULL
 *             The requested data
 * @retval NULL
 *             The reader is not valid or ih is not an instance handle
 *             of a matched writer.
 */
DDS_EXPORT dds_builtintopic_endpoint_t *
dds_get_matched_publication_data (
  dds_entity_t reader,
  dds_instance_handle_t ih);

#ifdef DDS_HAS_TYPE_DISCOVERY
/**
 * @brief Gets the type information from endpoint information that was
 *        retrieved by dds_get_matched_subscription_data or
 *        dds_get_matched_publication_data
 * @ingroup builtintopic
 *
 * @param[in] builtintopic_endpoint  The builtintopic endpoint struct
 * @param[out] type_info             Type information that will be allocated by this function in case of success.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             One or more parameters are invalid
 */
DDS_EXPORT dds_return_t
dds_builtintopic_get_endpoint_type_info (
  dds_builtintopic_endpoint_t * builtintopic_endpoint,
  const dds_typeinfo_t ** type_info);
#endif

/**
 * @brief Free the endpoint information that was retrieved by
 *        dds_get_matched_subscription_data or dds_get_matched_publication_data
 * @ingroup builtintopic
 *
 * This operation deallocates the memory of the fields in a
 * dds_builtintopic_endpoint_t struct and deallocates the
 * struct itself.
 *
 * @param[in] builtintopic_endpoint  The builtintopic endpoint struct
 */
DDS_EXPORT void
dds_builtintopic_free_endpoint (
  dds_builtintopic_endpoint_t * builtintopic_endpoint);

/**
 * @brief Free the provided topic information
 * @ingroup builtintopic
 *
 * This operation deallocates the memory of the fields in a
 * dds_builtintopic_topic_t struct and deallocates the
 * struct itself.
 *
 * @param[in] builtintopic_topic  The builtintopic topic struct
 */
DDS_EXPORT void
dds_builtintopic_free_topic (
  dds_builtintopic_topic_t * builtintopic_topic);

/**
 * @brief Free the provided participant information
 * @ingroup builtintopic
 *
 * This operation deallocates the memory of the fields in a
 * dds_builtintopic_participant_t struct and deallocates the
 * struct itself.
 *
 * @param[in] builtintopic_participant  The builtintopic participant struct
 */
DDS_EXPORT void
dds_builtintopic_free_participant (
  dds_builtintopic_participant_t * builtintopic_participant);

/**
 * @brief This operation manually asserts the liveliness of a writer
 *        or domain participant.
 * @ingroup entity
 *
 * This operation manually asserts the liveliness of a writer
 * or domain participant. This is used in combination with the Liveliness
 * QoS policy to indicate that the entity remains active. This operation need
 * only be used if the liveliness kind in the QoS is either
 * DDS_LIVELINESS_MANUAL_BY_PARTICIPANT or DDS_LIVELINESS_MANUAL_BY_TOPIC.
 *
 * @param[in] entity  A domain participant or writer
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 */
DDS_EXPORT dds_return_t
dds_assert_liveliness (
  dds_entity_t entity);


/**
 * @defgroup internal (Internal)
 * @ingroup dds
 */

/**
 * @defgroup testing (Testing tools)
 * @ingroup internal
 */

/**
 *
 * @brief This operation allows making the domain's network stack temporarily deaf and/or mute.
 * @ingroup testing
 * @warning Unstable API, for testing
 *
 * This is a support function for testing and, other special uses and is subject to change.
 *
 * @param[in] entity  A domain entity or an entity bound to a domain, such
 *                    as a participant, reader or writer.
 * @param[in] deaf    Whether to network stack should pretend to be deaf and
 *                    ignore any incoming packets.
 * @param[in] mute    Whether to network stack should pretend to be mute and
 *                    discard any outgoing packets where it normally would.
 *                    pass them to the operating system kernel for transmission.
 * @param[in] reset_after  Any value less than INFINITY will cause it to
 *                    set deaf = mute = false after reset_after ns have passed.
 *                    This is done by an event scheduled for the appropriate
 *                    time and otherwise forgotten. These events are not
 *                    affected by subsequent calls to this function.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter.
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
*/
DDS_EXPORT dds_return_t
dds_domain_set_deafmute (
  dds_entity_t entity,
  bool deaf,
  bool mute,
  dds_duration_t reset_after);


/**
 * @defgroup xtypes (XTypes)
 * @ingroup dds
 *
 * CycloneDDS supports XTypes, but most of that functionality outside the new IDL constructs
 * happens behind the scenes. However, some API functionality is added that allows inspecting
 * types at runtime. Using it in C is not very ergonomic, but dynamic languages like Python can
 * make good use of it.
 */

/**
 * @brief This function resolves the type for the provided type identifier,
 * which can e.g. be retrieved from endpoint or topic discovery data.
 * @ingroup xtypes
 *
 * @param[in]   entity              A domain entity or an entity bound to a domain, such
 *                                  as a participant, reader or writer.
 * @param[in]   type_id             Type identifier
 * @param[in]   timeout             Timeout for waiting for requested type information to be available
 * @param[out]  type_obj            The type information, untouched if type is not resolved
 *
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The entity parameter is not a valid parameter, type_id or type name
 *             is not provided, or the sertype out parameter is NULL
 * @retval DDS_RETCODE_NOT_FOUND
 *             A type with the provided type_id and type_name was not found
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Cyclone DDS built without type discovery
 *             (cf. DDS_HAS_TYPE_DISCOVERY)
*/
DDS_EXPORT dds_return_t
dds_get_typeobj (
  dds_entity_t entity,
  const dds_typeid_t *type_id,
  dds_duration_t timeout,
  dds_typeobj_t **type_obj);

/**
 * @brief Free the type object that was retrieved using dds_get_typeobj
 * @ingroup xtypes
 *
 * @param[in]  type_obj     The type object
 *
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The type_obj parameter is NULL
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Cyclone DDS built without type discovery
 *             (cf. DDS_HAS_TYPE_DISCOVERY)
*/
DDS_EXPORT dds_return_t
dds_free_typeobj (
  dds_typeobj_t *type_obj);

/**
 * @brief This function gets the type information from the
 * provided topic, reader or writer
 * @ingroup xtypes
 *
 * @param[in]   entity          A topic/reader/writer entity
 * @param[out]  type_info       The type information, untouched if returncode indicates failure
 *
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The type_info parameter is null
 * @retval DDS_RETCODE_NOT_FOUND
 *             The entity does not have type information set
 * @retval DDS_RETCODE_ILLEGAL_OPERATION
 *             The operation is invoked on an inappropriate object.
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Cyclone DDS built without type discovery
 *             (cf. DDS_HAS_TYPE_DISCOVERY)
*/
DDS_EXPORT dds_return_t
dds_get_typeinfo (
  dds_entity_t entity,
  dds_typeinfo_t **type_info);

/**
 * @brief Free the type information that was retrieved using dds_get_typeinfo
 * @ingroup xtypes
 *
 * @param[in]  type_info     The type information
 *
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             The operation was successful.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             The type_info parameter is NULL
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Cyclone DDS built without type discovery
 *             (cf. DDS_HAS_TYPE_DISCOVERY)
*/
DDS_EXPORT dds_return_t
dds_free_typeinfo (
  dds_typeinfo_t *type_info);

#if defined (__cplusplus)
}
#endif
#endif /* DDS_H */
