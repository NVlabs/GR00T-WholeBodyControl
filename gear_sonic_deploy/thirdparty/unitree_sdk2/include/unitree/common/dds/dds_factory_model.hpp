#ifndef __UT_DDS_FACTORY_MODEL_HPP__
#define __UT_DDS_FACTORY_MODEL_HPP__

#include <unitree/common/dds/dds_parameter.hpp>
#include <unitree/common/dds/dds_topic_channel.hpp>

namespace unitree
{
namespace common
{
class DdsFactoryModel
{
public:
    explicit DdsFactoryModel();
    ~DdsFactoryModel();

    void Init(uint32_t domainId, const std::string& ddsConfig = "");
    void Init(const std::string& ddsParameterFileName = "");
    void Init(const JsonMap& param);

    template<typename MSG>
    DdsTopicChannelPtr<MSG> CreateTopicChannel(const std::string& topic)
    {
        DdsTopicChannelPtr<MSG> channel = DdsTopicChannelPtr<MSG>(new DdsTopicChannel<MSG>());
        channel->SetTopic(mParticipant, topic, mTopicQos);
        return channel;
    }

    template<typename MSG>
    void SetWriter(DdsTopicChannelPtr<MSG>& channelPtr)
    {
        channelPtr->SetWriter(mPublisher, mWriterQos);
    }

    template<typename MSG>
    void SetReader(DdsTopicChannelPtr<MSG>& channelPtr, const std::function<void(const void*)>& handler, int32_t queuelen = 0)
    {
        DdsReaderCallback cb(handler);
        channelPtr->SetReader(mSubscriber, mReaderQos, cb, queuelen);
    }

private:
    DdsParticipantPtr mParticipant;
    DdsPublisherPtr mPublisher;
    DdsSubscriberPtr mSubscriber;

    DdsParticipantQos mParticipantQos;
    DdsTopicQos mTopicQos;
    DdsPublisherQos mPublisherQos;
    DdsSubscriberQos mSubscriberQos;
    DdsWriterQos mWriterQos;
    DdsReaderQos mReaderQos;

    Logger *mLogger;
};

using DdsFactoryModelPtr = std::shared_ptr<DdsFactoryModel>;

}
}

#endif//__UT_DDS_FACTORY_MODEL_HPP__
