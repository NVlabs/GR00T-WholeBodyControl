#ifndef __UT_ROBOT_SDK_CHANNEL_FACTORY_HPP__
#define __UT_ROBOT_SDK_CHANNEL_FACTORY_HPP__

#include <unitree/common/dds/dds_factory_model.hpp>

namespace unitree
{
namespace robot
{
template<typename MSG>
using Channel = unitree::common::DdsTopicChannel<MSG>;

template<typename MSG>
using ChannelPtr = unitree::common::DdsTopicChannelPtr<MSG>;

class ChannelFactory
{
public:
    static ChannelFactory* Instance()
    {
        static ChannelFactory inst;
        return &inst;
    }

    void Init(int32_t domainId, const std::string& networkInterface = "");
    void Init(const std::string& configFileName = "");
    void Init(const common::JsonMap& jsonMap);

    void Release();

    template<typename MSG>
    ChannelPtr<MSG> CreateSendChannel(const std::string& name)
    {
        ChannelPtr<MSG> channelPtr = mDdsFactoryPtr->CreateTopicChannel<MSG>(name);
        mDdsFactoryPtr->SetWriter(channelPtr);
        return channelPtr;
    }

    template<typename MSG>
    ChannelPtr<MSG> CreateRecvChannel(const std::string& name, std::function<void(const void*)> callback, int32_t queuelen = 0)
    {
        ChannelPtr<MSG> channelPtr = mDdsFactoryPtr->CreateTopicChannel<MSG>(name);
        mDdsFactoryPtr->SetReader(channelPtr, callback, queuelen);
        return channelPtr;
    }

public:
    ~ChannelFactory();

private:
    ChannelFactory();

private:
    bool mInited;
    common::DdsFactoryModelPtr mDdsFactoryPtr;
    common::Mutex mMutex;
};

}
}

#endif//__UT_ROBOT_SDK_CHANNEL_FACTORY_HPP__
