#ifndef __UT_ROBOT_SDK_CLIENT_STUB_HPP__
#define __UT_ROBOT_SDK_CLIENT_STUB_HPP__

#include <unitree/robot/future/request_future.hpp>
#include <unitree/robot/channel/channel_labor.hpp>
#include <unitree/common/block_queue.hpp>

namespace unitree
{
namespace robot
{
class ClientStub
{
public:
    explicit ClientStub();
    ~ClientStub();

    void Init(const std::string& name);

    bool Send(const Request& req, int64_t waitTimeout);
    RequestFuturePtr SendRequest(const Request& req, int64_t waitTimeout);

private:
    void ResponseFunc(const void* message);

private:
    ChannelLaborPtr<Request,Response> mChannelLaborPtr;
    RequestFutureQueuePtr mFutureQueuePtr;
};

using ClientStubPtr = std::shared_ptr<ClientStub>;

}
}

#endif//__UT_ROBOT_SDK_CLIENT_STUB_HPP__
