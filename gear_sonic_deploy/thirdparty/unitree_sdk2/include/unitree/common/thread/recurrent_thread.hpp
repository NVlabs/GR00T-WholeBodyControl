#ifndef __UT_RECURRENT_THREAD_HPP__
#define __UT_RECURRENT_THREAD_HPP__

#include <unitree/common/thread/thread.hpp>

#define UT_THREAD_TIME_INTERVAL_MICROSEC 1000000

namespace unitree
{
namespace common
{
class RecurrentThread : public Thread
{
public:
    __UT_THREAD_DECL_TMPL_FUNC_ARG__
    explicit RecurrentThread(uint64_t intervalMicrosec, __UT_THREAD_TMPL_FUNC_ARG__)
        : mQuit(false), mIntervalMicrosec(intervalMicrosec)
    {
        //recurrent function
        mFunc = std::bind(__UT_THREAD_BIND_FUNC_ARG__);

        //Call Thread::Run for runing thread
        if (mIntervalMicrosec == 0)
        {
            Run(&RecurrentThread::ThreadFunc_0, this);
        }
        else
        {
            Run(&RecurrentThread::ThreadFunc, this);
        }
    }

    __UT_THREAD_DECL_TMPL_FUNC_ARG__
    explicit RecurrentThread(const std::string& name, int32_t cpuId, uint64_t intervalMicrosec,
        __UT_THREAD_TMPL_FUNC_ARG__)
        : Thread(name, cpuId), mQuit(false), mIntervalMicrosec(intervalMicrosec)
    {
        //recurrent function
        mFunc = std::bind(__UT_THREAD_BIND_FUNC_ARG__);

        //Call Thread::Run for runing thread
        if (mIntervalMicrosec == 0)
        {
            Run(&RecurrentThread::ThreadFunc_0, this);
        }
        else
        {
            Run(&RecurrentThread::ThreadFunc, this);
        }
    }

    virtual ~RecurrentThread();

    int32_t ThreadFunc();
    int32_t ThreadFunc_0();

    bool Wait(int64_t microsec = 0);

private:
    volatile bool mQuit;
    uint64_t mIntervalMicrosec;
    std::function<void()> mFunc;
};

typedef std::shared_ptr<RecurrentThread> RecurrentThreadPtr;

__UT_THREAD_DECL_TMPL_FUNC_ARG__
ThreadPtr CreateRecurrentThread(uint64_t intervalMicrosec, __UT_THREAD_TMPL_FUNC_ARG__)
{
    return ThreadPtr(new RecurrentThread(intervalMicrosec, __UT_THREAD_BIND_FUNC_ARG__));
}

__UT_THREAD_DECL_TMPL_FUNC_ARG__
ThreadPtr CreateRecurrentThreadEx(const std::string& name, int32_t cpuId, uint64_t intervalMicrosec,
    __UT_THREAD_TMPL_FUNC_ARG__)
{
    return ThreadPtr(new RecurrentThread(name, cpuId, intervalMicrosec,
        __UT_THREAD_BIND_FUNC_ARG__));
}

}
}

#endif//__UT_RECURRENT_THREAD_HPP__
