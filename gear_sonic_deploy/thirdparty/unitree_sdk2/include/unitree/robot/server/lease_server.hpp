#ifndef __UT_ROBOT_LEASE_SERVER_HPP__
#define __UT_ROBOT_LEASE_SERVER_HPP__

#include <unitree/robot/server/server_base.hpp>

namespace unitree
{
namespace robot
{
class LeaseCache
{
public:
    LeaseCache();
    ~LeaseCache();

    void Set(int64_t id, const std::string& mName, int64_t lastModified = 0);
    void Renewal(int64_t lastModified = 0);
    void Clear();

    int64_t GetLastModified() const;
    int64_t GetId() const;
    const std::string& GetName() const;

private:
    int64_t mLastModified;
    int64_t mId;
    std::string mName;
};

class LeaseServer : public ServerBase
{
public:
    explicit LeaseServer(const std::string& name, int64_t term);
    ~LeaseServer();

    void Init();

    bool CheckRequestLeaseDenied(int64_t leaseId); 
    
private:
    void ServerRequestHandler(const RequestPtr& request);

    int32_t Apply(const std::string& parameter, std::string& data);
    int32_t Renewal(int64_t leaseId);

    int64_t GenerateId(const std::string& name);

private:
    int64_t mTerm;
    LeaseCache mCache;
    common::Mutex mMutex;
    ServerStubPtr mServerStubPtr;
};

using LeaseServerPtr = std::shared_ptr<LeaseServer>;

}
}

#endif//__UT_ROBOT_LEASE_SERVER_HPP__
