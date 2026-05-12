#ifndef __UT_LOG_POLICY_HPP__
#define __UT_LOG_POLICY_HPP__

#include <unitree/common/log/log_decl.hpp>

namespace unitree
{
namespace common
{
class LogPolicy
{
public:
    LogPolicy();
    LogPolicy(const std::string& tag, int32_t level, const std::string& store);

    void Dump();

public:
    std::string mTag;
    int32_t mLevel;
    std::string mStore;
};

typedef std::shared_ptr<LogPolicy> LogPolicyPtr;

class LogStorePolicy
{
public:
    LogStorePolicy();
    LogStorePolicy(const std::string& name, int32_t type, int32_t fileNumber, int64_t fileSize,
        int64_t fileWriteInter = UT_LOG_WRITE_INTER, int32_t cpuId = UT_CPU_ID_NONE,
        const std::string& fileName = "", const std::string& directory = "");

    void Dump();

public:
    std::string mName;
    int32_t mType;
    int32_t mFileNumber;
    int64_t mFileSize;
    int64_t mFileWriteInter;
    int32_t mCpuId;
    std::string mFileName;
    std::string mDirectory;
};

typedef std::shared_ptr<LogStorePolicy> LogStorePolicyPtr;

}
}

#endif//__UT_LOG_POLICY_HPP__
