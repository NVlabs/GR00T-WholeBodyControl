#ifndef __UT_ROBOT_SDK_CHANNEL_PUBLISHER_HPP__
#define __UT_ROBOT_SDK_CHANNEL_PUBLISHER_HPP__

#include <unitree/robot/channel/channel_factory.hpp>

namespace unitree
{
namespace robot
{
template<typename MSG>
class ChannelPublisher
{
public:
    explicit ChannelPublisher(const std::string& channelName) :
        mChannelName(channelName)
    {}

    void InitChannel()
    {
        mChannelPtr = ChannelFactory::Instance()->CreateSendChannel<MSG>(mChannelName);
    }

    bool Write(const MSG& msg, int64_t waitMicrosec = 0)
    {
        if (mChannelPtr)
        {
            return mChannelPtr->Write(msg, waitMicrosec);
        }

        return false;
    }

    void CloseChannel()
    {
        mChannelPtr.reset();
    }

    const std::string& GetChannelName() const
    {
        return mChannelName;
    }

private:
    std::string mChannelName;
    ChannelPtr<MSG> mChannelPtr;
};

template<typename MSG>
using ChannelPublisherPtr = std::shared_ptr<ChannelPublisher<MSG>>;

}
}

#endif//__UT_ROBOT_SDK_CHANNEL_PUBLISHER_HPP__
