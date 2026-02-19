#ifndef __UT_DDS_QOS_PARAMETER_HPP__
#define __UT_DDS_QOS_PARAMETER_HPP__

#include <unitree/common/dds/dds_qos_policy_parameter.hpp>

namespace unitree
{
namespace common
{
class DdsQosParameter
{
public:
    DdsQosParameter();
    ~DdsQosParameter();

    void Init(const JsonMap& data);

#define __GET_POLICY(POLICY)                                    \
    const DdsQos##POLICY##PolicyParameter& Get##POLICY() const  \
    {                                                           \
        return m##POLICY;                                       \
    }

    __GET_POLICY(Deadline)
    __GET_POLICY(DestinationOrder)
    __GET_POLICY(Durability)
    __GET_POLICY(DurabilityService)
    __GET_POLICY(EntityFactory)
    __GET_POLICY(GroupData)
    __GET_POLICY(History)
    __GET_POLICY(LatencyBudget)
    __GET_POLICY(Lifespan)
    __GET_POLICY(Liveliness)
    __GET_POLICY(Ownership)
    __GET_POLICY(OwnershipStrength)
    __GET_POLICY(Partition)
    __GET_POLICY(Presentation)
    __GET_POLICY(ReaderDataLifecycle)
    __GET_POLICY(Reliability)
    __GET_POLICY(ResourceLimits)
    __GET_POLICY(TimeBasedFilter)
    __GET_POLICY(TopicData)
    __GET_POLICY(TransportPriority)
    __GET_POLICY(WriterDataLifecycle)
    __GET_POLICY(UserData)

#undef __GET_POLICY

    bool Default() const;

private:
    bool mDefault;

    DdsQosDeadlinePolicyParameter mDeadline;
    DdsQosDestinationOrderPolicyParameter mDestinationOrder;
    DdsQosDurabilityPolicyParameter mDurability;
    DdsQosDurabilityServicePolicyParameter mDurabilityService;
    DdsQosEntityFactoryPolicyParameter mEntityFactory;
    DdsQosGroupDataPolicyParameter mGroupData;
    DdsQosHistoryPolicyParameter mHistory;
    DdsQosLatencyBudgetPolicyParameter mLatencyBudget;
    DdsQosLifespanPolicyParameter mLifespan;
    DdsQosLivelinessPolicyParameter mLiveliness;
    DdsQosOwnershipPolicyParameter mOwnership;
    DdsQosOwnershipStrengthPolicyParameter mOwnershipStrength;
    DdsQosPartitionPolicyParameter mPartition;
    DdsQosPresentationPolicyParameter mPresentation;
    DdsQosReaderDataLifecyclePolicyParameter mReaderDataLifecycle;
    DdsQosReliabilityPolicyParameter mReliability;
    DdsQosResourceLimitsPolicyParameter mResourceLimits;
    DdsQosTimeBasedFilterPolicyParameter mTimeBasedFilter;
    DdsQosTopicDataPolicyParameter mTopicData;
    DdsQosTransportPriorityPolicyParameter mTransportPriority;
    DdsQosWriterDataLifecyclePolicyParameter mWriterDataLifecycle;
    DdsQosUserDataPolicyParameter mUserData;
};

}
}

#endif//__UT_DDS_QOS_PARAMETER_HPP__
