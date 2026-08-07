#ifndef __UT_DDS_QOS_HPP__
#define __UT_DDS_QOS_HPP__

#include <unitree/common/dds/dds_qos_policy.hpp>

using namespace org::eclipse::cyclonedds;

namespace unitree
{
namespace common
{
#define UT_DECL_DDS_QOS(QosType, QosNative)         \
class QosType: public QosNative                     \
{                                                   \
public:                                             \
    using NativeQosType = QosNative::NativeType;    \
    explicit QosType()                              \
    {                                               \
        InitPolicyDefault();                        \
    }                                               \
    template<typename POLICY>                       \
    void SetPolicy(const POLICY& policy)            \
    {                                               \
        mNative.policy(policy.GetNative());         \
        mPolicyNameSet.insert(policy.GetName());    \
    }                                               \
    bool HasPolicy(const std::string& name) const   \
    {                                               \
        return mPolicyNameSet.find(name) != mPolicyNameSet.end();   \
    }                                               \
    void CopyToNativeQos(NativeQosType& qos) const; \
private:                                            \
    void InitPolicyDefault();                       \
private:                                            \
    std::set<std::string> mPolicyNameSet;           \
};

/*
 * DdsParticipantQos
 */
UT_DECL_DDS_QOS(DdsParticipantQos, DdsNative<::dds::domain::qos::DomainParticipantQos>)

/*
 * DdsTopicQos
 */
UT_DECL_DDS_QOS(DdsTopicQos, DdsNative<::dds::topic::qos::TopicQos>)

/*
 * DdsPublisherQos
 */
UT_DECL_DDS_QOS(DdsPublisherQos, DdsNative<::dds::pub::qos::PublisherQos>)

/*
 * DdsSubscriberQos
 */
UT_DECL_DDS_QOS(DdsSubscriberQos, DdsNative<::dds::sub::qos::SubscriberQos>)

/*
 * DdsWriterQos
 */
UT_DECL_DDS_QOS(DdsWriterQos, DdsNative<::dds::pub::qos::DataWriterQos>)

/*
 * DdsReaderQos
 */
UT_DECL_DDS_QOS(DdsReaderQos, DdsNative<::dds::sub::qos::DataReaderQos>)

}
}

#endif//__UT_DDS_QOS_HPP__
