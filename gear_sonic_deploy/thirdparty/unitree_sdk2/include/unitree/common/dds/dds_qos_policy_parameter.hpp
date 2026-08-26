#ifndef __UT_DDS_QOS_POLICY_PARAMETER_HPP__
#define __UT_DDS_QOS_POLICY_PARAMETER_HPP__

#include <unitree/common/json/json.hpp>

namespace unitree
{
namespace common
{
/*
 *----------------------------
 * Supported Qos policy:
 * 1. Deadline
 *      IDL:
 *      struct DeadlineQosPolicy {
 *           Duration_t period;
 *      };
 *
 * 2. DestinationOrder
 *      IDL:
 *      enum DestinationOrderQosPolicyKind {
 *          BY_RECEPTION_TIMESTAMP_DESTINATIONORDER_QOS,
 *          BY_SOURCE_TIMESTAMP_DESTINATIONORDER_QOS
 *      };
 *      struct DestinationOrderQosPolicy {
 *          DestinationOrderQosPolicyKind kind;
 *      };
 *
 * 3. Durability
 *      IDL:
 *      enum DurabilityQosPolicyKind {
 *          VOLATILE_DURABILITY_QOS, // Least Durability
 *          TRANSIENT_LOCAL_DURABILITY_QOS,
 *          TRANSIENT_DURABILITY_QOS,
 *          PERSISTENT_DURABILITY_QOS // Greatest Durability
 *      };
 *      struct DurabilityQosPolicy {
 *          DurabilityQosPolicyKind kind;
 *      };
 *
 * 4. DurabilityService
 *      IDL:
 *      struct DurabilityServiceQosPolicy {
 *          Duration_t service_cleanup_delay;
 *          HistoryQosPolicyKind history_kind;
 *          long history_depth;
 *          long max_samples;
 *          long max_instances;
 *          long max_samples_per_instance;
 *      };
 *
 * 5. EntityFactory
 *      IDL:
 *      struct EntityFactoryQosPolicy {
 *          boolean autoenable_created_entities;
 *      };
 *
 * 6. GroupData
 *      IDL:
 *      struct GroupDataQosPolicy {
 *          sequence<octet> value;
 *      };
 *
 * 7. History
 *      IDL:
 *      enum HistoryQosPolicyKind {
 *          KEEP_LAST_HISTORY_QOS,
 *          KEEP_ALL_HISTORY_QOS
 *      };
 *      struct HistoryQosPolicy {
 *          HistoryQosPolicyKind kind;
 *          long depth;
 *      };
 * 8. LatencyBudget
 *      IDL:
 *      struct LatencyBudgetQosPolicy {
 *          Duration_t duration;
 *      };
 *
 * 9. Lifespan
 *      IDL:
 *      struct LifespanQosPolicy {
 *          Duration_t duration;
 *      }
 *
 * 10.Liveliness
 *      IDL:
 *      enum LivelinessQosPolicyKind {
 *          AUTOMATIC_LIVELINESS_QOS,
 *          MANUAL_BY_PARTICIPANT_LIVELINESS_QOS,
 *          MANUAL_BY_TOPIC_LIVELINESS_QOS
 *      };
 *      struct LivelinessQosPolicy {
 *          LivelinessQosPolicyKind kind;
 *          Duration_t lease_duration;
 *      };
 *
 * 11.Ownership
 *      IDL:
 *      enum OwnershipQosPolicyKind {
 *          SHARED_OWNERSHIP_QOS,
 *          EXCLUSIVE_OWNERSHIP_QOS
 *      };
 *      struct OwnershipQosPolicy {
 *          OwnershipQosPolicyKind kind;
 *      };
 *
 * 12.OwnershipStrength
 *      IDL:
 *      struct OwnershipStrengthQosPolicy {
 *          long value;
 *      };
 *
 * 13.Partition
 *      IDL:
 *      struct PartitionQosPolicy {
 *          StringSeq name;
 *      };
 *
 * 14.Presentation
 *      IDL:
 *      enum PresentationQosPolicyAccessScopeKind {
 *          INSTANCE_PRESENTATION_QOS,
 *          TOPIC_PRESENTATION_QOS,
 *          GROUP_PRESENTATION_QOS
 *      };
 *      struct PresentationQosPolicy {
 *          PresentationQosPolicyAccessScopeKind access_scope;
 *          boolean coherent_access;
 *          boolean ordered_access;
 *      };
 *
 * 15.ReaderDataLifecycle
 *      IDL:
 *      struct ReaderDataLifecycleQosPolicy {
 *          Duration_t autopurge_nowriter_samples_delay;
 *          Duration_t autopurge_disposed_samples_delay;
 *      };
 *
 * 16.Reliability
 *      IDL:
 *      enum ReliabilityQosPolicyKind {
 *          BEST_EFFORT_RELIABILITY_QOS,
 *          RELIABLE_RELIABILITY_QOS
 *      };
 *      struct ReliabilityQosPolicy {
 *          ReliabilityQosPolicyKind kind;
 *          Duration_t max_blocking_time;
 *      };
 * 17.ResourceLimits
 *      IDL:
 *      struct ResourceLimitsQosPolicy {
 *          long max_samples;
 *          long max_instances;
 *          long max_samples_per_instance;
 *      };
 *
 * 18.TimeBasedFilter
 *      IDL:
 *      struct TimeBasedFilterQosPolicy {
 *          Duration_t minimum_separation;
 *      };
 *
 * 19.TopicData
 *      IDL:
 *      struct TopicDataQosPolicy {
 *          sequence<octet> value;
 *      };
 *
 * 20.TransportPriority
 *      IDL:
 *      struct TransportPriorityQosPolicy {
 *          long value;
 *      };
 *
 * 21.UserData
 *      IDL:
 *      struct UserDataQosPolicy {
 *          sequence<octet> value;
 *      };
 *
 * 22.WriterDataLifecycle
 *      IDL:
 *      struct WriterDataLifecycleQosPolicy {
 *          boolean autodispose_unregistered_instances;
 *      };
 *
 *-------------------------------------------------------------------------------------------------
 * QoS policis corresponding to entity: 
 * DomainParticipant: [2]  { EntityFactory, UserData }
 * Topic:             [13] { Deadline, DestinationOrder, Durability, DurabilityService, History,
 *                           LatencyBudget, Lifespan, Liveliness, Ownership, Reliability,
 *                           ResourceLimits, TopicData, TransportPriority }
 * Publisher:         [4]  { EntityFactory, GroupData, Partition, Presentation }
 * Subscriber:        [4]  { EntityFactory, GroupData, Partition, Presentation }
 * DataWriter:        [14] { Deadline, DestinationOrder, Durability, History, LatencyBudget, 
 *                           Lifespan, Liveliness, Ownership, OwnershipStrength, Reliability,
 *                           ResourceLimits, TransportPriority, UserData, WriterDataLifecycle }
 * DataReader:        [12] { Deadline, DestinationOrder, Durability, History, LatencyBudget
 *                           Liveliness, Ownership, Reliability, ReaderDataLifecycle, ResourceLimits,
 *                           TimeBasedFilter, UserData }
 *-------------------------------------------------------------------------------------------------
 */

/*
 * Qos policy name
 */
#define UT_DDS_QOS_POLICY_NAME_DEADLINE                 "Deadline"
#define UT_DDS_QOS_POLICY_NAME_DESTINATION_ORDER        "DestinationOrder"
#define UT_DDS_QOS_POLICY_NAME_DURABILITY               "Durability"
#define UT_DDS_QOS_POLICY_NAME_DURABILITY_SERVICE       "DurabilityService"
#define UT_DDS_QOS_POLICY_NAME_ENTITY_FACTORY           "EntityFactory"
#define UT_DDS_QOS_POLICY_NAME_GROUP_DATA               "GroupData"
#define UT_DDS_QOS_POLICY_NAME_HISTORY                  "History"
#define UT_DDS_QOS_POLICY_NAME_LATENCY_BUDGET           "LatencyBudget"
#define UT_DDS_QOS_POLICY_NAME_LIFESPAN                 "Lifespan"
#define UT_DDS_QOS_POLICY_NAME_LIVELINESS               "Liveliness"
#define UT_DDS_QOS_POLICY_NAME_OWNERSHIP                "Ownership"
#define UT_DDS_QOS_POLICY_NAME_OWNERSHIP_STRENGTH       "OwnershipStrength"
#define UT_DDS_QOS_POLICY_NAME_PARTITION                "Partition"
#define UT_DDS_QOS_POLICY_NAME_PRESENTATION             "Presentation"
#define UT_DDS_QOS_POLICY_NAME_READER_DATA_LIFECYCLE    "ReaderDataLifecycle"
#define UT_DDS_QOS_POLICY_NAME_RELIABILITY              "Reliability"
#define UT_DDS_QOS_POLICY_NAME_RESOURCE_LIMITS          "ResourceLimits"
#define UT_DDS_QOS_POLICY_NAME_TIME_BASED_FILTER        "TimeBasedFilter"
#define UT_DDS_QOS_POLICY_NAME_TOPIC_DATA               "TopicData"
#define UT_DDS_QOS_POLICY_NAME_TRANSPORT_PRIORITY       "TransportPriority"
#define UT_DDS_QOS_POLICY_NAME_WRITER_DATA_LIFECYCLE    "WriterDataLifecycle"
#define UT_DDS_QOS_POLICY_NAME_USER_DATA                "UserData"

/*
 * Qos Policy Member Name
 */
#define UT_DDS_QOS_POLICY_MEMBER_NAME_KIND              "kind"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_PEROID            "peroid"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_CLEANUP_DELAY     "service_cleanup_delay"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_HISTORY_KIND      "history_kind"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_HISTORY_DEPTH     "history_depth"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_MAX_SAMPLES       "max_samples"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_MAX_INSTANCES     "max_instances"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_MAX_SAMPLES_PER_INSTANCE  "max_samples_per_instance"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_AUTO_ENABLE       "autoenable_created_entities"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_VALUE             "value"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_DEPTH             "depth"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_DURATION          "duration"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_LEASE_DURATION    "lease_duration"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_NAME              "name"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_ACCESS_SCOPE      "access_scope"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_COHERENT_ACCESS   "coherent_access"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_ORDERED_ACCESS    "ordered_access"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_MAX_BLOCKING_TIME "max_blocking_time"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_AUTODISPOSE_UNREGISETED_INSTANCES "autodispose_unregistered_instances"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_AUTOPURGE_NOWRITER_SAMPLES_DELAY  "autopurge_nowriter_samples_delay"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_AUTOPURGE_DISPOSED_SAMPLES_DELAY  "autopurge_disposed_samples_delay"
#define UT_DDS_QOS_POLICY_MEMBER_NAME_MIN_SEP           "minimum_separation"

class DdsQosPolicyParameter
{
public:
    DdsQosPolicyParameter();
    virtual ~DdsQosPolicyParameter();

    virtual void Init(const JsonMap& data) = 0;

    void Update();
    bool Default() const;

protected:
    bool mDefault;
};

class DdsQosDeadlinePolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosDeadlinePolicyParameter();
    ~DdsQosDeadlinePolicyParameter();

    void Init(const JsonMap& data);
    int64_t GetPeriod() const;

private:
    int64_t mPeriod;
};

class DdsQosDestinationOrderPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosDestinationOrderPolicyParameter();
    ~DdsQosDestinationOrderPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetKind() const;

private:
    int32_t mKind;
};

class DdsQosDurabilityPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosDurabilityPolicyParameter();
    ~DdsQosDurabilityPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetKind() const;

private:
    int32_t mKind;
};

class DdsQosDurabilityServicePolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosDurabilityServicePolicyParameter();
    ~DdsQosDurabilityServicePolicyParameter();

    void Init(const JsonMap& data);

    int64_t GetCleanupDelay() const;
    int32_t GetHistoryKind() const;
    int32_t GetHistoryDepth() const;
    int32_t GetMaxSamples() const;
    int32_t GetMaxInstances() const;
    int32_t GetMaxSamplesPerInstance() const;

private:
    int64_t mCleanupDelay;
    int32_t mHistoryKind;
    int32_t mHistoryDepth;
    int32_t mMaxSamples;
    int32_t mMaxInstances;
    int32_t mMaxSamplesPerInstance;
};

class DdsQosEntityFactoryPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosEntityFactoryPolicyParameter();
    ~DdsQosEntityFactoryPolicyParameter();

    void Init(const JsonMap& data);

    int64_t GetAutoEnable() const;

private:
    bool mAutoEnable;
};

class DdsQosGroupDataPolicyParameter: public DdsQosPolicyParameter
{
public:
    DdsQosGroupDataPolicyParameter();
    ~DdsQosGroupDataPolicyParameter();

    void Init(const JsonMap& data);
    const std::vector<uint8_t>& GetValue() const;

private:
    std::vector<uint8_t> mValue;
};

class DdsQosHistoryPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosHistoryPolicyParameter();
    ~DdsQosHistoryPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetKind() const;
    int32_t GetDepth() const;

private:
    int32_t mKind;
    int32_t mDepth;
};

class DdsQosLatencyBudgetPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosLatencyBudgetPolicyParameter();
    ~DdsQosLatencyBudgetPolicyParameter();

    void Init(const JsonMap& data);

    int64_t GetDuration() const;

private:
    int64_t mDuration;
};

class DdsQosLifespanPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosLifespanPolicyParameter();
    ~DdsQosLifespanPolicyParameter();

    void Init(const JsonMap& data);

    int64_t GetDuration() const;

private:
    int64_t mDuration;
};

class DdsQosLivelinessPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosLivelinessPolicyParameter();
    ~DdsQosLivelinessPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetKind() const;
    int64_t GetLeaseDuration() const;

private:
    int32_t mKind;
    int64_t mLeaseDuration;
};

class DdsQosOwnershipPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosOwnershipPolicyParameter();
    ~DdsQosOwnershipPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetKind() const;

private:
    int32_t mKind;
};

class DdsQosOwnershipStrengthPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosOwnershipStrengthPolicyParameter();
    ~DdsQosOwnershipStrengthPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetValue() const;

private:
    int32_t mValue;
};

class DdsQosPartitionPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosPartitionPolicyParameter();
    ~DdsQosPartitionPolicyParameter();

    void Init(const JsonMap& data);

    const std::string& GetName() const;

private:
    std::string mName;
};

class DdsQosPresentationPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosPresentationPolicyParameter();
    ~DdsQosPresentationPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetAccessScopeKind() const;
    bool GetCoherentAccess() const;
    bool GetOrderedAccess() const;

private:
    int32_t mAccessScopeKind;
    bool mCoherentAccess;
    bool mOrderedAccess;
};

class DdsQosReaderDataLifecyclePolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosReaderDataLifecyclePolicyParameter();
    ~DdsQosReaderDataLifecyclePolicyParameter();

    void Init(const JsonMap& data);

    int64_t GetAutopurgeNowriterSamplesDelay() const;
    int64_t GetAutopurgeDisposedSamplesDelay() const;

private:
    int64_t mAutopurgeNowriterSamplesDelay;
    int64_t mAutopurgeDisposedSamplesDelay;
};

class DdsQosReliabilityPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosReliabilityPolicyParameter();
    ~DdsQosReliabilityPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetKind() const;
    int64_t GetMaxBlockingTime() const;

private:
    int32_t mKind;
    int64_t mMaxBlockingTime;
};

class DdsQosResourceLimitsPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosResourceLimitsPolicyParameter();
    ~DdsQosResourceLimitsPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetMaxSamples() const;
    int32_t GetMaxInstances() const;
    int32_t GetMaxSamplesPerInstance() const;

private:
    int32_t mMaxSamples;
    int32_t mMaxInstances;
    int32_t mMaxSamplesPerInstance;
};

class DdsQosTimeBasedFilterPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosTimeBasedFilterPolicyParameter();
    ~DdsQosTimeBasedFilterPolicyParameter();

    void Init(const JsonMap& data);

    int64_t GetMinSep() const;

private:
    int64_t mMinSep;
};

class DdsQosTopicDataPolicyParameter: public DdsQosPolicyParameter
{
public:
    DdsQosTopicDataPolicyParameter();
    ~DdsQosTopicDataPolicyParameter();

    void Init(const JsonMap& data);

    const std::vector<uint8_t>& GetValue() const;

private:
    std::vector<uint8_t> mValue;
};

class DdsQosTransportPriorityPolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosTransportPriorityPolicyParameter();
    ~DdsQosTransportPriorityPolicyParameter();

    void Init(const JsonMap& data);

    int32_t GetValue() const;

private:
    int32_t mValue;
};

class DdsQosWriterDataLifecyclePolicyParameter : public DdsQosPolicyParameter
{
public:
    DdsQosWriterDataLifecyclePolicyParameter();
    ~DdsQosWriterDataLifecyclePolicyParameter();

    void Init(const JsonMap& data);

    bool GetAutodisposeUnregisteredInstances() const;

private:
    bool mAutodisposeUnregisteredInstances;
};

class DdsQosUserDataPolicyParameter: public DdsQosPolicyParameter
{
public:
    DdsQosUserDataPolicyParameter();
    ~DdsQosUserDataPolicyParameter();

    void Init(const JsonMap& data);

    const std::vector<uint8_t>& GetValue() const;

private:
    std::vector<uint8_t> mValue;
};

}
}

#endif//__UT_DDS_QOS_POLICY_PARAMETER_HPP__
