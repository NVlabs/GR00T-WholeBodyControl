#ifndef __LOG_WRITER_HPP__
#define __LOG_WRITER_HPP__

#include <unitree/common/log/log_buffer.hpp>
#include <unitree/common/log/log_keeper.hpp>

namespace unitree
{
namespace common
{
class LogWriter
{
public:
    virtual void Write(const std::string& s) = 0;
};

typedef std::shared_ptr<LogWriter> LogWriterPtr;

class LogDirectWriter : public LogWriter
{
public:
    explicit LogDirectWriter(int32_t fd);
    virtual ~LogDirectWriter();

    void Write(const std::string& s);

private:
    Mutex mLock;
    int32_t mFd;
};

class LogStdoutWriter : public LogDirectWriter
{
public:
    explicit LogStdoutWriter() :
        LogDirectWriter(UT_FD_STDOUT)
    {}
    ~LogStdoutWriter()
    {}
};

class LogStderrWriter : public LogDirectWriter
{
public:
    explicit LogStderrWriter() :
        LogDirectWriter(UT_FD_STDERR)
    {}
    ~LogStderrWriter()
    {}
};

class LogBufferWriter : public LogWriter
{
public:
    explicit LogBufferWriter(LogKeeperPtr keeperPtr);
    ~LogBufferWriter();

    void Write(const std::string& s);

private:
    LogBufferPtr mBufferPtr;
    LogKeeperPtr mKeeperPtr;
    Mutex mLock;
};

class LogAsyncBufferWriter : public LogWriter
{
public:
    explicit LogAsyncBufferWriter(LogKeeperPtr keeperPtr);
    ~LogAsyncBufferWriter();

    void Write(const std::string& s);

private:
    void DoWrite();

private:
    volatile bool mRotate;
    std::string mTempBuf;
    LogBufferPtr mBufferPtr;
    LogKeeperPtr mKeeperPtr;
    ThreadPtr mThreadPtr;
    Mutex mLock;
};
}
}

#endif//__LOG_WRITER_HPP__
