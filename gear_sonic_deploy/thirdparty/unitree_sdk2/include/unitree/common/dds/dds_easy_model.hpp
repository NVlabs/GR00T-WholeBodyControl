#ifndef __DDS_EASY_MODEL_HPP__
#define __DDS_EASY_MODEL_HPP__

#include <unitree/common/dds/dds_topic_channel.hpp>
#include <unitree/common/dds/dds_parameter.hpp>
#include <unitree/common/dds/dds_qos_realize.hpp>
#include <unitree/common/string_tool.hpp>

#define UT_DDS_PARAMETER_CONFIG_FILENAME "dds_parameter.json"

namespace unitree
{
namespace common
{
class DdsEasyModel
{
public:
    explicit DdsEasyModel();
    ~DdsEasyModel();

    void Init(uint32_t domainId);
    void Init(const std::string& ddsParameterFileName = "");
    void Init(const JsonMap& param);

    template<typename MSG>
    void SetTopic(const std::string& topic)
    {
        DdsTopicChannelPtr<MSG> channel = GetChannel<MSG>(topic);
        if (!channel)
        {
            channel = DdsTopicChannelPtr<MSG>(new DdsTopicChannel<MSG>());
            mChannelMap[topic] = std::static_pointer_cast<DdsTopicChannelAbstract>(channel);

            DdsTopicQos topicQos;
            GetTopicQos(topic, topicQos);
            channel->SetTopic(mParticipant, topic, topicQos);
        }

        DdsWriterPtr<MSG> writer = channel->GetWriter();
        if (!writer)
        {
            DdsWriterQos writerQos;
            GetWriterQos(topic, writerQos);
            channel->SetWriter(GetPublisher(topic), writerQos);
        }
        else
        {
            UT_THROW(CommonException, std::string("topic reader is already exist. topic:") + topic);
        }
    }

    template<typename MSG>
    void SetTopic(const std::string& topic, const DdsMessageHandler& handler, int32_t queuelen = 0)
    {
        DdsReaderCallback cb(handler);
        SetTopic<MSG>(topic, cb, queuelen);
    }

    template<typename MSG>
    void SetTopic(const std::string& topic, const DdsReaderCallback& rcb, int32_t queuelen = 0)
    {
        DdsTopicChannelPtr<MSG> channel = GetChannel<MSG>(topic);
        if (!channel)
        {
            channel = DdsTopicChannelPtr<MSG>(new DdsTopicChannel<MSG>());
            mChannelMap[topic] = std::static_pointer_cast<DdsTopicChannelAbstract>(channel);

            DdsTopicQos topicQos;
            GetTopicQos(topic, topicQos);
            channel->SetTopic(mParticipant, topic, topicQos);
        }

        DdsReaderPtr<MSG> reader = channel->GetReader();
        if (!reader)
        {
            DdsReaderQos readerQos;
            GetReaderQos(topic, readerQos);
            channel->SetReader(GetSubscriber(topic), readerQos, rcb, queuelen);
        }
        else
        {
            UT_THROW(CommonException, std::string("topic reader is already exist. topic:") + topic);
        }
    }

    template<typename MSG>
    bool WriteMessage(const std::string topic, const MSG& message, int64_t waitMicrosec = 0)
    {
        DdsTopicChannelPtr<MSG> channel = GetChannel<MSG>(topic);
        if (channel == NULL)
        {
            return false;
        }

        return channel->Write(message, waitMicrosec);
    }

    bool WriteMessage(const std::string topic, const void* message, int64_t waitMicrosec = 0);

    int64_t GetLastDataAvailableTime(const std::string topic);

private:
    void GetTopicQos(const std::string& topic, DdsTopicQos& qos);
    void GetWriterQos(const std::string& topic, DdsWriterQos& qos);
    void GetReaderQos(const std::string& topic, DdsReaderQos& qos);

    DdsTopicChannelAbstractPtr GetChannel(const std::string& topic);

    template<typename MSG>
    DdsTopicChannelPtr<MSG> GetChannel(const std::string& topic)
    {
        DdsTopicChannelPtr<MSG> channel;

        DdsTopicChannelAbstractPtr channelAbstract = GetChannel(topic);
        if (channelAbstract)
        {
            channel = std::static_pointer_cast<DdsTopicChannel<MSG>>(channelAbstract);
        }

        return channel;
    }

    DdsSubscriberPtr GetSubscriber(const std::string& topic);
    DdsSubscriberPtr GetSubscriberDefault();

    DdsPublisherPtr GetPublisher(const std::string& topic);
    DdsPublisherPtr GetPublisherDefault();

private:
    DdsParameter mDdsParameter;

    DdsParticipantPtr mParticipant;
    std::vector<DdsPublisherPtr> mPublisherList;
    std::vector<DdsSubscriberPtr> mSubscriberList;
    DdsPublisherPtr mPublisherDefault;
    DdsSubscriberPtr mSubscriberDefault;

    std::map<std::string,DdsTopicChannelAbstractPtr> mChannelMap;

    Logger *mLogger;
};

using DdsEasyModelPtr = std::shared_ptr<DdsEasyModel>;

}
}

#endif//__DDS_EASY_MODEL_HPP__
