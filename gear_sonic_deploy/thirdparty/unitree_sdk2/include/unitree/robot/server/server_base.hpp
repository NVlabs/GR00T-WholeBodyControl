#ifndef __UT_ROBOT_SDK_SERVER_BASE_HPP__
#define __UT_ROBOT_SDK_SERVER_BASE_HPP__

#include <unitree/robot/server/server_stub.hpp>

namespace unitree
{
namespace robot
{
class ServerBase
{
public:
    explicit ServerBase(const std::string& name);
    virtual ~ServerBase();

    virtual void Init() = 0;
    virtual void Start(bool enableProiQueue = false);

    const std::string& GetName() const;

protected:
    virtual void ServerRequestHandler(const RequestPtr& request) = 0;
    void SendResponse(const Response& response);

protected:
    std::string mName;
    ServerStubPtr mServerStubPtr;
};

using ServerBasePtr = std::shared_ptr<ServerBase>;

}
}

#endif//__UT_ROBOT_SDK_SERVER_BASE_HPP__
