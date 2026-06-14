#ifndef __UT_ROBOT_SDK_SERVER_STUB_HPP__
#define __UT_ROBOT_SDK_SERVER_STUB_HPP__

#include <unitree/robot/internal/internal.hpp>
#include <unitree/robot/channel/channel_labor.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/block_queue.hpp>

namespace unitree
{
namespace robot
{
using ServerRequestHandler = std::function<void(const RequestPtr& requestPtr)>;

class ServerStub
{
public:
    explicit ServerStub();
    ~ServerStub();

    void Init(const std::string& name, const ServerRequestHandler& handler, bool enableProiQueue);
    bool Send(const Response& response, int64_t timeout = 0);

private:
    void Enqueue(const void* message);

    int32_t QueueThreadFunction();
    int32_t ProiQueueThreadFunction();

private:
    bool mEnableProiQueue;
    bool mRunning;
    ServerRequestHandler mRequestHandler;
    ChannelLaborPtr<Response,Request> mChannelLaborPtr;
    common::BlockQueuePtr<RequestPtr> mQueuePtr;
    common::BlockQueuePtr<RequestPtr> mProiQueuePtr;
    common::ThreadPtr mQueueThreadPtr;
    common::ThreadPtr mProiQueueThreadPtr;
};

using ServerStubPtr = std::shared_ptr<ServerStub>;

}
}

#endif//__UT_ROBOT_SDK_SERVER_STUB_HPP__
