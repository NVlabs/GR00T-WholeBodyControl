#ifndef OMG_DDS_CORE_POLICY_POLICYKIND_HPP_
#define OMG_DDS_CORE_POLICY_POLICYKIND_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Corp.
 * Copyright 2010, Real-Time Innovations, Inc.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dds/core/detail/conformance.hpp>
#include <dds/core/SafeEnumeration.hpp>
#include <vector>
#include <cstdint>

namespace dds
{
namespace core
{
namespace policy
{

#if defined (__SUNPRO_CC) && defined(SHARED)
#   undef SHARED
#endif
struct OwnershipKind_def
{
    enum Type
    {
        SHARED   /**< The same instance can be written by
                  *    multiple DataWriter objects. All updates will be made available to the
                  *    DataReader objects. In other words it does not have a specific owner.
                  */
        #ifdef  OMG_DDS_OWNERSHIP_SUPPORT
        ,
        EXCLUSIVE /**< The instance will only be accepted from one
                       DataWriter which is the only one whose modifications will be visible to the
                       DataReader objects.
                  */
        #endif  // OMG_DDS_OWNERSHIP_SUPPORT
    };
};

typedef dds::core::safe_enum<OwnershipKind_def> OwnershipKind;

struct DataRepresentationId_def
{
    enum Type: std::int16_t
    {
      XCDR1 = 0,  /**< Old style CDR serialization, does NOT support extensible types.*/
      XML = 1,    /**< XML representation, not supported by writers at this moment.*/
      XCDR2 = 2   /**< New style CDR serialization, DOES support extensible types.*/
    };
};

typedef dds::core::safe_enum<DataRepresentationId_def> DataRepresentationId;
typedef std::vector<DataRepresentationId> DataRepresentationIdSeq;

struct DurabilityKind_def
{
    enum Type
    {
        VOLATILE,       /**< The samples are not available to late-joining
                         *   DataReaders. In other words, only DataReaders, which were present at the
                         *   time of the writing and have subscribed to this Topic, will receive the sample.
                         *   When a DataReader subscribes afterwards (late-joining), it will only be able to
                         *   read the next written sample. This setting is typically used for data, which is
                         *   updated quickly.
                         */
        TRANSIENT_LOCAL /**< Currently behaves identically to the
                         *   TRANSIENT_DURABILITY_QOS, except for its RxO properties. The desired
                         *   behaviour of TRANSIENT_LOCAL_DURABILITY_QOS can be achieved from the
                         *   TRANSIENT_DURABILITY_QOS with the default (TRUE) setting of the
                         *   autodispose_unregistered_instances flag on the DataWriter and the
                         *   service_cleanup_delay set to 0 on the durability service. This is because for
                         *   TRANSIENT_LOCAL, the data should only remain available for late-joining
                         *   readers during the lifetime of its source writer, so it is not required to survive after
                         *   its source writer has been deleted. Since the deletion of a writer implicitly
                         *   unregisters all its instances, an autodispose_unregistered_instances
                         *   value of TRUE will also dispose the affected data from the durability store, and
                         *   thus prevent it from remaining available to late joining readers.
                         */
        #ifdef  OMG_DDS_PERSISTENCE_SUPPORT
        ,
        TRANSIENT,      /**< Some samples are available to late-joining
                         *   DataReaders (stored in memory). This means that the late-joining
                         *   DataReaders are able to read these previously written samples. The
                         *   DataReader does not necessarily have to exist at the time of writing. Not all
                         *   samples are stored (depending on QosPolicy History and QosPolicy
                         *   resource_limits). The storage does not depend on the DataWriter and will
                         *   outlive the DataWriter. This may be used to implement reallocation of
                         *   applications because the data is saved in the Data Distribution Service (not in the
                         *   DataWriter). This setting is typically used for state related information of an
                         *   application. In this case also the DurabilityServiceQosPolicy settings are
                         *   relevant for the behaviour of the Data Distribution Service.
                         */
        PERSISTENT      /**< The data is stored in permanent storage (e.g.
                         *   hard disk). This means that the samples are also available after a system restart.
                         *   The samples not only outlives the DataWriters, but even the Data Distribution
                         *   Service and the system. This setting is typically used for attributes and settings for
                         *   an application or the system. In this case also the
                         *   DurabilityServiceQosPolicy settings are relevant for the behaviour of the
                         *   Data Distribution Service.
                         */
        #endif  // #ifdef  OMG_DDS_PERSISTENCE_SUPPORT
    };
};
typedef dds::core::safe_enum<DurabilityKind_def> DurabilityKind;

struct PresentationAccessScopeKind_def
{
    enum Type
    {
        INSTANCE, /**< Presentation Access Scope is per instance. */
        TOPIC     /**< Presentation Access Scope is per topic. */

        #ifdef  OMG_DDS_OBJECT_MODEL_SUPPORT
        ,
        GROUP     /**< Presentation Access Scope is per group. */
        #endif  // OMG_DDS_OBJECT_MODEL_SUPPORT
    };
};
typedef dds::core::safe_enum<PresentationAccessScopeKind_def> PresentationAccessScopeKind;


struct ReliabilityKind_def
{
    enum Type
    {
        BEST_EFFORT, /**< The Data Distribution Service will only
                      *   attempt to deliver the data; no arrival-checks are being performed and any lost
                      *   data is not re-transmitted (non-reliable). Presumably new values for the samples
                      *   are generated often enough by the application so that it is not necessary to resent
                      *   or acknowledge any samples.
                      */
        RELIABLE     /**< The Data Distribution Service will attempt to
                      *   deliver all samples in the DataWriters history; arrival-checks are performed
                      *   and data may get re-transmitted in case of lost data. In the steady-state (no
                      *   modifications communicated via the DataWriter) the Data Distribution Service
                      *   guarantees that all samples in the DataWriter history will eventually be
                      *   delivered to the all DataReader objects. Outside the steady-state the
                      *   HistoryQosPolicy and ResourceLimitsQosPolicy determine how
                      *   samples become part of the history and whether samples can be discarded from it.
                      *   In this case also the max_blocking_time must be set.
                      */
    };
};
typedef dds::core::safe_enum<ReliabilityKind_def> ReliabilityKind;


struct DestinationOrderKind_def
{
    enum Type
    {
        BY_RECEPTION_TIMESTAMP, /**< The order is based on the timestamp, at the moment the sample was
                                 *   received by the DataReader.
                                 */
        BY_SOURCE_TIMESTAMP     /**< The order is based on the timestamp, which was set by the
                                 *   DataWriter. This means that the system needs some time synchronization.
                                 */
    };
};

typedef dds::core::safe_enum<DestinationOrderKind_def> DestinationOrderKind;

struct HistoryKind_def
{
    enum Type
    {
        KEEP_LAST, /**< The Data Distribution Service will only attempt to
                    *   keep the latest values of the instance and discard the older ones. The attribute
                    *   “depth” determines how many samples in history will be stored. In other words,
                    *   only the most recent samples in history are stored. On the publishing side, the
                    *   Data Distribution Service will only keep the most recent “depth” samples of each
                    *   instance of data (identified by its key) managed by the DataWriter. On the
                    *   subscribing side, the DataReader will only keep the most recent “depth”
                    *   samples received for each instance (identified by its key) until the application
                    *   “takes” them via the DataReader::take operation.
                    *   KEEP_LAST_HISTORY_QOS - is the default kind. The default value of depth is
                    *   1, indicating that only the most recent value should be delivered. If a depth other
                    *   than 1 is specified, it should be compatible with the settings of the
                    *   ResourcelimitsQosPolicy max_samples_per_instance. For these two
                    *   QosPolicy settings to be compatible, they must verify that depth <=
                    *   max_samples_per_instance, otherwise a
                    *   RETCODE_INCONSISTENT_POLICY is generated on relevant operations.
                    */
        KEEP_ALL   /**< All samples are stored, provided, the resources are
                    *   available. On the publishing side, the Data Distribution Service will attempt to
                    *   keep all samples (representing each value written) of each instance of data
                    *   (identified by its key) managed by the DataWriter until they can be delivered to
                    *   all subscribers. On the subscribing side, the Data Distribution Service will
                    *   attempt to keep all samples of each instance of data (identified by its key)
                    *   managed by the DataReader. These samples are kept until the application
                    *   “takes” them from the Data Distribution Service via the DataReader::take
                    *   operation. The setting of depth has no effect. Its implied value is
                    *   LENGTH_UNLIMITED. The resources that the Data Distribution Service can use to
                    *   keep this history are limited by the settings of the ResourceLimitsQosPolicy.
                    *   If the limit is reached, the behaviour of the Data Distribution Service will depend
                    *   on the ReliabilityQosPolicy. If the ReliabilityQosPolicy is
                    *   BEST_EFFORT_RELIABILITY_QOS, the old values are discarded. If
                    *   ReliabilityQosPolicy is RELIABLE_RELIABILITY_QOS, the Data
                    *   Distribution Service will block the DataWriter until it can deliver the necessary
                    *   old values to all subscribers.
                    */
    };
};

typedef dds::core::safe_enum<HistoryKind_def> HistoryKind;

struct LivelinessKind_def
{
    enum Type
    {
        AUTOMATIC,             /**< The Data Distribution Service will take care of
                                *   reporting the Liveliness automatically with a rate determined by the
                                *   lease_duration.
                                */
        MANUAL_BY_PARTICIPANT, /**< The application must take care
                                *   of reporting the liveliness before the lease_duration expires. If an Entity
                                *   reports its liveliness, all Entities within the same DomainParticipant that
                                *   have their liveliness kind set to MANUAL_BY_PARTICIPANT_LIVELINESS_QOS,
                                *   can be considered alive by the Data Distribution Service. Liveliness can reported
                                *   explicitly by calling the operation assert_liveliness on the
                                *   DomainParticipant or implicitly by writing some data.
                                */
        MANUAL_BY_TOPIC        /**< The application must take care of
                                *   reporting the liveliness before the lease_duration expires. This can explicitly
                                *   be done by calling the operation assert_liveliness on the DataWriter or
                                *   implicitly by writing some data.
                                */
    };
};
typedef dds::core::safe_enum<LivelinessKind_def> LivelinessKind;

struct TypeConsistencyKind_def
{
    enum Type
    {
        DISALLOW_TYPE_COERCION,
        ALLOW_TYPE_COERCION
    };
};

typedef dds::core::safe_enum<TypeConsistencyKind_def> TypeConsistencyKind;

}
}
}
#endif /* OMG_DDS_CORE_POLICY_POLICYKIND_HPP_ */
