#ifndef __LOG_STORE_HPP__
#define __LOG_STORE_HPP__

#include <unitree/common/log/log_writer.hpp>
#include <unitree/common/log/log_keeper.hpp>

namespace unitree
{
namespace common
{
class LogStore
{
public:
    explicit LogStore()
    {}
    virtual ~LogStore()
    {
        mWriterPtr.reset();
    }

    virtual void Append(const std::string& s) = 0;

protected:
    LogWriterPtr mWriterPtr;
};

typedef std::shared_ptr<LogStore> LogStorePtr;

class LogStdoutStore : public LogStore
{
public:
    explicit LogStdoutStore();
    ~LogStdoutStore();

    void Append(const std::string& s);
};

typedef std::shared_ptr<LogStdoutStore> LogStdoutStorePtr;

class LogStderrStore : public LogStore
{
public:
    explicit LogStderrStore();
    ~LogStderrStore();

    void Append(const std::string& s);
};

typedef std::shared_ptr<LogStderrStore> LogStderrStorePtr;

class LogFileStore : public LogStore
{
public:
    explicit LogFileStore(LogKeeperPtr keeperPtr);
    ~LogFileStore();

    void Append(const std::string& s);
};

typedef std::shared_ptr<LogFileStore> LogFileStorePtr;

class LogFileAsyncStore : public LogStore
{
public:
    explicit LogFileAsyncStore(LogKeeperPtr keeperPtr);
    ~LogFileAsyncStore();

    void Append(const std::string& s);
};

typedef std::shared_ptr<LogFileAsyncStore> LogFileAsyncStorePtr;

}
}

#endif//__LOG_STORE_HPP__
