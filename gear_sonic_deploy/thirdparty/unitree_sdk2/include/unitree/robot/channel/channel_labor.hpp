#ifndef __UT_ROBOT_SDK_CHANNEL_LABOR_HPP__
#define __UT_ROBOT_SDK_CHANNEL_LABOR_HPP__

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_namer.hpp>
#include <unitree/common/time/time_tool.hpp>

namespace unitree
{
namespace robot
{
/*
 * @brief
 * @class: ChannelLabor
 */
template<typename SEND_MSG, typename RECV_MSG>
class ChannelLabor
{
public:
    ChannelLabor()
    {}

    virtual ~ChannelLabor()
    {}

    void InitChannel(const std::string& name, const std::function<void(const void*)>& recvMesageCallback, int32_t queuelen = 0)
    {
        std::string sendChannelName = mNamerPtr->GetSendChannelName(name);
        std::string recvChannelName = mNamerPtr->GetRecvChannelName(name);

        mSendChannlPtr = ChannelFactory::Instance()->CreateSendChannel<SEND_MSG>(sendChannelName);
        mRecvChannlPtr = ChannelFactory::Instance()->CreateRecvChannel<RECV_MSG>(recvChannelName, recvMesageCallback, queuelen);
    }

    bool Send(const SEND_MSG& msg, int64_t waitTimeout)
    {
        return mSendChannlPtr->Write(msg, waitTimeout);
    }

    int64_t GetLastDataAvailableTime() const
    {
        return mRecvChannlPtr->GetLastDataAvailableTime();
    }

protected:
    ChannelNamerPtr mNamerPtr;

private:
    ChannelPtr<SEND_MSG> mSendChannlPtr;
    ChannelPtr<RECV_MSG> mRecvChannlPtr;
};

template<typename SEND_MSG, typename RECV_MSG>
using ChannelLaborPtr = std::shared_ptr<ChannelLabor<SEND_MSG,RECV_MSG>>;

/*
 * @brief
 * @class: ClientChannelLabor
 */
template<typename SEND_MSG, typename RECV_MSG>
class ClientChannelLabor : public ChannelLabor<SEND_MSG,RECV_MSG>
{
public:
    ClientChannelLabor()
    {
        ChannelLabor<SEND_MSG,RECV_MSG>::mNamerPtr = ChannelNamerPtr(new ClientChannelNamer());
    }

    ~ClientChannelLabor()
    {}
};

template<typename SEND_MSG, typename RECV_MSG>
using ClientChannelLaborPtr = std::shared_ptr<ClientChannelLabor<SEND_MSG,RECV_MSG>>;

/*
 * @brief
 * @class: ServerChannelLabor
 */
template<typename SEND_MSG, typename RECV_MSG>
class ServerChannelLabor : public ChannelLabor<SEND_MSG,RECV_MSG>
{
public:
    ServerChannelLabor()
    {
        ChannelLabor<SEND_MSG,RECV_MSG>::mNamerPtr = ChannelNamerPtr(new ServerChannelNamer());
    }

    ~ServerChannelLabor()
    {}
};

template<typename SEND_MSG, typename RECV_MSG>
using ServerChannelLaborPtr = std::shared_ptr<ServerChannelLabor<RECV_MSG,SEND_MSG>>;

}
}

#endif//__UT_ROBOT_SDK_CHANNEL_LABOR_HPP__
