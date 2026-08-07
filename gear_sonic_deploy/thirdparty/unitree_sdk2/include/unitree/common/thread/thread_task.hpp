#ifndef __UT_THREAD_TASK_HPP__
#define __UT_THREAD_TASK_HPP__

#include <unitree/common/thread/future.hpp>

namespace unitree
{
namespace common
{
class ThreadTask
{
public:
    __UT_THREAD_DECL_TMPL_FUNC_ARG__
    explicit ThreadTask(__UT_THREAD_TMPL_FUNC_ARG__)
    {
        mFunc = std::bind(__UT_THREAD_BIND_FUNC_ARG__);
    }

    virtual void Execute();

    void SetEnqueueTime();
    uint64_t GetEnqueueTime() const;

protected:
    uint64_t mEnqueueTimeMicrosec;
    std::function<Any()> mFunc;
};

typedef std::shared_ptr<ThreadTask> ThreadTaskPtr;

class ThreadTaskFuture : public ThreadTask, public FutureWrapper
{
public:
    __UT_THREAD_DECL_TMPL_FUNC_ARG__
    explicit ThreadTaskFuture(__UT_THREAD_TMPL_FUNC_ARG__)
        : ThreadTask(__UT_THREAD_BIND_FUNC_ARG__)
    {}

    void Execute();
};

typedef std::shared_ptr<ThreadTaskFuture> ThreadTaskFuturePtr;

}
}

#endif//__UT_THREAD_TASK_HPP__
