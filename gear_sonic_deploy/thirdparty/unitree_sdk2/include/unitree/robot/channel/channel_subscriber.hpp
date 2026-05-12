#ifndef __UT_ROBOT_SDK_CHANNEL_SUBSCRIBER_HPP__
#define __UT_ROBOT_SDK_CHANNEL_SUBSCRIBER_HPP__

#include <unitree/robot/channel/channel_factory.hpp>

namespace unitree
{
namespace robot
{
template<typename MSG>
class ChannelSubscriber
{
public:
    explicit ChannelSubscriber(const std::string& channelName) :
        mChannelName(channelName), mQueueLen(0)
    {}

    explicit ChannelSubscriber(const std::string& channelName, const std::function<void(const void*)>& handler, int64_t queuelen = 0) :
        mChannelName(channelName), mQueueLen(queuelen), mHandler(handler)
    {}

    void InitChannel(const std::function<void(const void*)>& handler, int64_t queuelen = 0)
    {
        mHandler = handler;
        mQueueLen = queuelen;

        InitChannel();
    }

    void InitChannel()
    {
        if (mHandler)
        {
            mChannelPtr = ChannelFactory::Instance()->CreateRecvChannel<MSG>(mChannelName, mHandler, mQueueLen);
        }
        else
        {
            UT_THROW(common::CommonException, "subscribe handler is invalid");
        }
    }

    void CloseChannel()
    {
        mChannelPtr.reset();
    }

    int64_t GetLastDataAvailableTime() const
    {
        if (mChannelPtr)
        {
            return mChannelPtr->GetLastDataAvailableTime();
        }

        return -1;
    }

    const std::string& GetChannelName() const
    {
        return mChannelName;
    }

private:
    std::string mChannelName;
    int64_t mQueueLen;
    std::function<void(const void*)> mHandler;
    ChannelPtr<MSG> mChannelPtr;
};

template<typename MSG>
using ChannelSubscriberPtr = std::shared_ptr<ChannelSubscriber<MSG>>;

}
}

#endif//__UT_ROBOT_SDK_CHANNEL_SUBSCRIBER_HPP__
