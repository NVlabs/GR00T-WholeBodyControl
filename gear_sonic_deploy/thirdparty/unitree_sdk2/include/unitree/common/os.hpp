#ifndef __UT_OS_HPP__
#define __UT_OS_HPP__

#include <unitree/common/decl.hpp>
#include <unitree/common/lock/lock.hpp>

namespace unitree
{
namespace common
{
enum UT_SCHED_POLICY
{
    UT_SCHED_POLICY_NORMAL = SCHED_OTHER,
    UT_SCHED_POLICY_FIFO = SCHED_FIFO,
    UT_SCHED_POLICY_RR = SCHED_RR,
    UT_SCHED_POLICY_BATCH = SCHED_BATCH,
    UT_SCHED_POLICY_IDLE = SCHED_IDLE,
    UT_SCHED_POLICY_DEADLINE = SCHED_DEADLINE
};

class OsHelper
{
public:
    static OsHelper* Instance()
    {
        static OsHelper inst;
        return &inst;
    }

    uid_t GetUID() const;
    gid_t GetGID() const;
    std::string GetUser() const;
    const struct passwd& GetPasswd() const;

    bool GetPasswd(const std::string& name, struct passwd& pwd);
    bool GetUIDAndGID(const std::string& name, uid_t& uid, gid_t& gid);

    int32_t GetProcessorNumber() const;
    int32_t GetProcessorNumberConf() const;

    int32_t GetPageSize() const;

    int64_t Align(int64_t len) const;
    bool IsAligned(int64_t len) const;

    const std::string& GetHostname() const;

    uint32_t GetProcessId();
    const std::string& GetProcessFileName();

    std::string GetProcessName();
    std::string GetProcessDirectory(bool withEndDelim = true);

    uint64_t GetThreadId();
    int32_t GetTid();

    bool GetNetworkInterfaceIps(std::map<std::string,std::string>& networkInterfaceIpMap);
    bool GetIps(std::set<std::string>& ips);

    void CpuSet(const std::string& cpuIds);
    void CpuSet(uint64_t threadId, size_t cpuId);

    void SetThreadName(uint64_t threadId, const std::string& name);

    void SetScheduler(int32_t policy, int32_t priority);
    void SetScheduler(uint64_t threadId, int32_t policy, int32_t priority);

    void SingletonProcessInstance();

private:
    OsHelper();

    void __Getpwuid();
    void __GetProcessor();
    void __GetPageSize();
    void __GetProcessFileName();
    void __GetHostname();

private:
    uid_t mUID;
    struct passwd mPasswd;
    int32_t mProcessorNumber;
    int32_t mProcessorNumberConf;
    int32_t mPageSize;
    std::string mHostname;
    std::string mProcessFileName;
    std::shared_ptr<Filelock> mInstanceLockPtr;
};

}
}

#endif//__UT_OS_HPP__
