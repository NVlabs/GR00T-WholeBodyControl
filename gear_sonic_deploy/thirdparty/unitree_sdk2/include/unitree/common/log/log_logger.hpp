#ifndef __UT_LOGGER_HPP__
#define __UT_LOGGER_HPP__

#include <unitree/common/log/log_store.hpp>

namespace unitree
{
namespace common
{
class LogBuilder
{
public:
    LogBuilder()
    {}

    template<typename T>
    LogBuilder& operator()(const std::string& key, const T& value)
    {
        mOs << "\t" << key << ":" << value;
        return *this;
    }

    std::ostringstream mOs;
};

class Logger
{
public:
    explicit Logger(int32_t level, LogStorePtr storePtr) :
        mLevel(level), mStorePtr(storePtr)
    {}

    template<typename ...Args>
    void Log(int32_t level, Args&&... args)
    {
        if (level > mLevel || mStorePtr == NULL)
        {
            return;
        }

        std::ostringstream os;
        LogBegin(os, level);
        LogPend(os, std::forward<Args>(args)...);
        LogEnd(os);

        mStorePtr->Append(os.str());
    }

    void LogFormat(int32_t level, LogBuilder& builder)
    {
        if (level > mLevel || mStorePtr == NULL)
        {
            return;
        }

        std::ostringstream os;
        LogBegin(os, level);
        LogPendBuilder(os, builder);
        LogEnd(os);

        mStorePtr->Append(os.str());
    }

    template<typename ...Args>
    void CritLog(int32_t level, const std::string& key, int32_t code, Args&&... args)
    {
        if (level > mLevel || mStorePtr == NULL)
        {
            return;
        }

        std::ostringstream os;
        LogBegin(os, level);
        LogPendCrit(os, key, code);
        LogPend(os, std::forward<Args>(args)...);
        LogEnd(os);

        mStorePtr->Append(os.str());
    }

    void CritLogFormat(int32_t level, const std::string& key, int32_t code, LogBuilder& builder)
    {
        if (level > mLevel || mStorePtr == NULL)
        {
            return;
        }

        std::ostringstream os;
        LogBegin(os, level);
        LogPendCrit(os, key, code);
        LogPendBuilder(os, builder);
        LogEnd(os);

        mStorePtr->Append(os.str());
    }

private:
    void LogPendCrit(std::ostringstream& os, const std::string& key, int32_t code)
    {
        os << " [__KEY__:" << key << "]";
        os << " [__CODE__:" << code << "]";
    }

    template<typename ...Args>
    void LogPend(std::ostringstream& os, Args&&... args)
    {
        os << " ";
        std::initializer_list<int32_t>{ (os << args, 0)... };
    }

    void LogPendBuilder(std::ostringstream& os, LogBuilder& builder)
    {
        os << builder.mOs.str();
    }

    void LogBegin(std::ostringstream& os, int32_t level)
    {
        os << "[" << GetTimeMillisecondString() << "] ";
        os << "[" << GetLogLevelDesc(level) << "] ";
        os << "[" << OsHelper::Instance()->GetProcessId() << "] ";
        os << "[" << OsHelper::Instance()->GetTid() << "]";
        os << std::setprecision(6) << std::fixed;
    }

    void LogEnd(std::ostringstream& os)
    {
        os << std::endl;
    }

private:
    int32_t mLevel;
    LogStorePtr mStorePtr;
};

typedef std::shared_ptr<Logger> LoggerPtr;
}
}

#endif//__UT_LOGGER_HPP__
