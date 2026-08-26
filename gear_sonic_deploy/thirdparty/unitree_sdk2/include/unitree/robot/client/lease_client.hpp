#ifndef __UT_ROBOT_SDK_LEASE_CLIENT_H__
#define __UT_ROBOT_SDK_LEASE_CLIENT_H__

#include <unitree/robot/client/client_base.hpp>
#include <unitree/common/thread/recurrent_thread.hpp>

namespace unitree
{
namespace robot
{
class LeaseContext
{
public:
    LeaseContext();
    ~LeaseContext();

    void Update(int64_t id, int64_t term);
    void Reset();

    bool Valid() const;

    int64_t GetId() const;
    int64_t GetTerm() const;

private:
    int64_t mId;
    int64_t mTerm;
};

using LeaseContextPtr = std::shared_ptr<LeaseContext>;

class LeaseClient : public ClientBase
{
public:
    explicit LeaseClient(const std::string& name);
    ~LeaseClient();

    void Init();

    void WaitApplied();
    int64_t GetId();
    bool Applied();

private:
    void Apply();
    void Renewal();

    void ThreadFunction();

    int64_t GetWaitMicrosec();

private:
    std::string mName;
    std::string mContextName;
    LeaseContext mContext;
    common::ThreadPtr mThreadPtr;
    common::Mutex mMutex;
};

using LeaseClientPtr = std::shared_ptr<LeaseClient>;

}
}

#endif//__UT_ROBOT_SDK_LEASE_CLIENT_H__
