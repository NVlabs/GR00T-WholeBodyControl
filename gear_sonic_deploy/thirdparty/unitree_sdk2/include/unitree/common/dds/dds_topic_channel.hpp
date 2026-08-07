#ifndef __UT_DDS_TOPIC_CHANNEL_HPP__
#define __UT_DDS_TOPIC_CHANNEL_HPP__

#include <unitree/common/dds/dds_entity.hpp>

namespace unitree
{
namespace common
{
/*
 * @brief: DdsTopicChannelAbstract
 */
class DdsTopicChannelAbstract
{
public:
    virtual bool Write(const void* message, int64_t waitMicrosec) = 0;
    virtual int64_t GetLastDataAvailableTime() const = 0;
};

using DdsTopicChannelAbstractPtr = std::shared_ptr<DdsTopicChannelAbstract>;

#define UT_DDS_WAIT_MATCHED_TIME_MICRO_SEC 100000

/*
 * @brief: DdsTopicChannel
 */
template<typename MSG>
class DdsTopicChannel : public DdsTopicChannelAbstract
{
public:
    explicit DdsTopicChannel()
    {}

    ~DdsTopicChannel()
    {}

    void SetTopic(const DdsParticipantPtr& participant, const std::string& name, const DdsTopicQos& qos)
    {
        mTopic = DdsTopicPtr<MSG>(new DdsTopic<MSG>(participant, name, qos));
    }

    void SetWriter(const DdsPublisherPtr& publisher, const DdsWriterQos& qos)
    {
        mWriter = DdsWriterPtr<MSG>(new DdsWriter<MSG>(publisher, mTopic, qos));
        MicroSleep(UT_DDS_WAIT_MATCHED_TIME_MICRO_SEC);
    }

    void SetReader(const DdsSubscriberPtr& subscriber, const DdsReaderQos& qos, const DdsReaderCallback& cb, int32_t queuelen)
    {
        mReader = DdsReaderPtr<MSG>(new DdsReader<MSG>(subscriber, mTopic, qos));
        mReader->SetListener(cb, queuelen);
    }

    DdsWriterPtr<MSG> GetWriter() const
    {
        return mWriter;
    }

    DdsReaderPtr<MSG> GetReader() const
    {
        return mReader;
    }

    bool Write(const void* message, int64_t waitMicrosec)
    {
        return Write(*(const MSG*)message, waitMicrosec);
    }

    bool Write(const MSG& message, int64_t waitMicrosec)
    {
        return mWriter->Write(message, waitMicrosec);
    }

    int64_t GetLastDataAvailableTime() const
    {
        if (mReader)
        {
            return mReader->GetLastDataAvailableTime();
        }

        return 0;
    }

private:
    DdsTopicPtr<MSG> mTopic;
    DdsWriterPtr<MSG> mWriter;
    DdsReaderPtr<MSG> mReader;
};

template<typename MSG>
using DdsTopicChannelPtr = std::shared_ptr<DdsTopicChannel<MSG>>;

}
}

#endif//__UT_DDS_TOPIC_CHANNEL_HPP__
