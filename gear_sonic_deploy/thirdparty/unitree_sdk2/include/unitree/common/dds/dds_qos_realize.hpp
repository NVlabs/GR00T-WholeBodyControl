#ifndef __UT_DDS_QOS_REALIZE_HPP__
#define __UT_DDS_QOS_REALIZE_HPP__

#include <unitree/common/dds/dds_qos_parameter.hpp>
#include <unitree/common/dds/dds_qos.hpp>

namespace unitree
{
namespace common
{
void Realize(const DdsQosParameter& parameter, DdsParticipantQos& qos);

void Realize(const DdsQosParameter& parameter, DdsTopicQos& qos);

void Realize(const DdsQosParameter& parameter, DdsPublisherQos& qos);

void Realize(const DdsQosParameter& parameter, DdsSubscriberQos& qos);

void Realize(const DdsQosParameter& parameter, DdsWriterQos& qos);

void Realize(const DdsQosParameter& parameter, DdsReaderQos& qos);

}
}

#endif//__UT_DDS_QOS_REALIZE_HPP__
