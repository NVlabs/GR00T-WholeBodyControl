#ifndef __UT_DDS_QOS_POLICY_HPP__
#define __UT_DDS_QOS_POLICY_HPP__

#include <dds/dds.hpp>
#include <unitree/common/dds/dds_native.hpp>

namespace unitree
{
namespace common
{
class DdsQosPolicyName
{
public:
    explicit DdsQosPolicyName(const std::string& name) :
        mName(name)
    {}

    virtual ~DdsQosPolicyName()
    {}

    const std::string& GetName() const
    {
        return mName;
    }

protected:
    std::string mName;
};

class DdsDuration : public DdsNative<::dds::core::Duration>
{
public:
    explicit DdsDuration(int64_t nanoSecond);
    ~DdsDuration();
};

class DdsQosDeadlinePolicy : public DdsNative<::dds::core::policy::Deadline>, public DdsQosPolicyName
{
public:
    explicit DdsQosDeadlinePolicy(int64_t period);
    ~DdsQosDeadlinePolicy();
};

class DdsQosDestinationOrderPolicy : public DdsNative<::dds::core::policy::DestinationOrder>, public DdsQosPolicyName
{
public:
    explicit DdsQosDestinationOrderPolicy(int32_t kind);
    ~DdsQosDestinationOrderPolicy();
};

class DdsQosDurabilityPolicy : public DdsNative<::dds::core::policy::Durability>, public DdsQosPolicyName
{
public:
    explicit DdsQosDurabilityPolicy(int32_t kind);
    ~DdsQosDurabilityPolicy();
};

class DdsQosDurabilityServicePolicy : public DdsNative<::dds::core::policy::DurabilityService>, public DdsQosPolicyName
{
public:
    explicit DdsQosDurabilityServicePolicy(int64_t cleanupDelay, int32_t historyKind, int32_t historyDepth,
        int32_t maxSamples, int32_t maxInstances, int32_t maxSamplesPerInstance);
    ~DdsQosDurabilityServicePolicy();
};

class DdsQosEntityFactoryPolicy : public DdsNative<::dds::core::policy::EntityFactory>, public DdsQosPolicyName
{
public:
    explicit DdsQosEntityFactoryPolicy(bool autoEnable);
    ~DdsQosEntityFactoryPolicy();
};

class DdsQosGroupDataPolicy : public DdsNative<::dds::core::policy::GroupData>, public DdsQosPolicyName
{
public:
    explicit DdsQosGroupDataPolicy(const std::vector<uint8_t>& value);
    ~DdsQosGroupDataPolicy();
};

class DdsQosHistoryPolicy : public DdsNative<::dds::core::policy::History>, public DdsQosPolicyName
{
public:
    explicit DdsQosHistoryPolicy(int32_t kind, int32_t depth);
    ~DdsQosHistoryPolicy();
};

class DdsQosLatencyBudgetPolicy : public DdsNative<::dds::core::policy::LatencyBudget>, public DdsQosPolicyName
{
public:
    explicit DdsQosLatencyBudgetPolicy(int64_t duration);
    ~DdsQosLatencyBudgetPolicy();
};

class DdsQosLifespanPolicy : public DdsNative<::dds::core::policy::Lifespan>, public DdsQosPolicyName
{
public:
    explicit DdsQosLifespanPolicy(int64_t duration);
    ~DdsQosLifespanPolicy();
};

class DdsQosLivelinessPolicy : public DdsNative<::dds::core::policy::Liveliness>, public DdsQosPolicyName
{
public:
    explicit DdsQosLivelinessPolicy(int32_t kind, int64_t leaseDuration);
    ~DdsQosLivelinessPolicy();
};

class DdsQosOwnershipPolicy : public DdsNative<::dds::core::policy::Ownership>, public DdsQosPolicyName
{
public:
    explicit DdsQosOwnershipPolicy(int32_t kind);
    ~DdsQosOwnershipPolicy();
};

class DdsQosOwnershipStrengthPolicy : public DdsNative<::dds::core::policy::OwnershipStrength>, public DdsQosPolicyName
{
public:
    explicit DdsQosOwnershipStrengthPolicy(int32_t strength);
    ~DdsQosOwnershipStrengthPolicy();
};

class DdsQosPartitionPolicy : public DdsNative<::dds::core::policy::Partition>, public DdsQosPolicyName
{
public:
    explicit DdsQosPartitionPolicy(const std::string& name);
    ~DdsQosPartitionPolicy();
};

class DdsQosPresentationPolicy : public DdsNative<::dds::core::policy::Presentation>, public DdsQosPolicyName
{
public:
    explicit DdsQosPresentationPolicy(int32_t accessScopeKind, bool coherentAccess, bool orderedAccess);
    ~DdsQosPresentationPolicy();
};

class DdsQosReaderDataLifecyclePolicy : public DdsNative<::dds::core::policy::ReaderDataLifecycle>, public DdsQosPolicyName
{
public:
    explicit DdsQosReaderDataLifecyclePolicy(int64_t autopurgeNowriterSamplesDelay, int64_t autopurgeDisposedSamplesDelay);
    ~DdsQosReaderDataLifecyclePolicy();
};

class DdsQosReliabilityPolicy : public DdsNative<::dds::core::policy::Reliability>, public DdsQosPolicyName
{
public:
    explicit DdsQosReliabilityPolicy(int32_t kind, int64_t maxBlockingTime);
    ~DdsQosReliabilityPolicy();
};

class DdsQosResourceLimitsPolicy : public DdsNative<::dds::core::policy::ResourceLimits>, public DdsQosPolicyName
{
public:
    explicit DdsQosResourceLimitsPolicy(int32_t maxSamples, int32_t maxInstances, int32_t maxSamplesPerInstance);
    ~DdsQosResourceLimitsPolicy();
};

class DdsQosTimeBasedFilterPolicy : public DdsNative<::dds::core::policy::TimeBasedFilter>, public DdsQosPolicyName
{
public:
    explicit DdsQosTimeBasedFilterPolicy(int64_t minSep);
    ~DdsQosTimeBasedFilterPolicy();
};

class DdsQosTopicDataPolicy : public DdsNative<::dds::core::policy::TopicData>, public DdsQosPolicyName
{
public:
    explicit DdsQosTopicDataPolicy(const std::vector<uint8_t>& value);
    ~DdsQosTopicDataPolicy();
};

class DdsQosTransportPriorityPolicy : public DdsNative<::dds::core::policy::TransportPriority>, public DdsQosPolicyName
{
public:
    explicit DdsQosTransportPriorityPolicy(int32_t value);
    ~DdsQosTransportPriorityPolicy();
};

class DdsQosWriterDataLifecyclePolicy : public DdsNative<::dds::core::policy::WriterDataLifecycle>, public DdsQosPolicyName
{
public:
    explicit DdsQosWriterDataLifecyclePolicy(bool autodisposeUnregisteredInstances);
    ~DdsQosWriterDataLifecyclePolicy();
};

class DdsQosUserDataPolicy : public DdsNative<::dds::core::policy::UserData>, public DdsQosPolicyName
{
public:
    explicit DdsQosUserDataPolicy(const std::vector<uint8_t>& value);
    ~DdsQosUserDataPolicy();
};

}
}

#endif//__UT_DDS_QOS_POLICY_HPP__
