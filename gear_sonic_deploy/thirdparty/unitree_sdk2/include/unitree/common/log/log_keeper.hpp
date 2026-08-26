#ifndef __UT_LOG_FILE_KEEPER_H__
#define __UT_LOG_FILE_KEEPER_H__

#include <unitree/common/log/log_policy.hpp>

namespace unitree
{
namespace common
{
class LogKeeper
{
public:
    LogKeeper(LogStorePolicyPtr storePolicyPtr);
    ~LogKeeper();

    LogStorePolicyPtr GetStorePolicy() const;

    bool Append(const std::string& s, bool rotate);

private:
    void Rotate();

    void AppendFile(const std::string& s);
    void OpenFile();
    void CloseFile();

    void CheckFileSize();

    std::string MakeRegexExpress();

private:
    volatile int64_t mFileSize;
    std::string mFileName;
    std::string mDirectory;
    FilePtr mFilePtr;
    LogStorePolicyPtr mStorePolicyPtr;
};

typedef std::shared_ptr<LogKeeper> LogKeeperPtr;

}
}

#endif//__UT_LOG_FILE_KEEPER_H__
