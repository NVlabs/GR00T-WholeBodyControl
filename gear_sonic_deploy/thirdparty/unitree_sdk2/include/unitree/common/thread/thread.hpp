#ifndef __UT_THREAD_HPP__
#define __UT_THREAD_HPP__

#include <unitree/common/thread/future.hpp>

namespace unitree
{
namespace common
{
class Thread : public FutureWrapper
{
public:
    Thread()
        : mThreadId(0), mCpuId(UT_CPU_ID_NONE)
    {}

    Thread(const std::string& name, int32_t cpuId)
        : mThreadId(0), mName(name), mCpuId(cpuId)
    {}

    __UT_THREAD_DECL_TMPL_FUNC_ARG__
    explicit Thread(__UT_THREAD_TMPL_FUNC_ARG__)
        : Thread()
    {
        Run(__UT_THREAD_BIND_FUNC_ARG__);
    }

    __UT_THREAD_DECL_TMPL_FUNC_ARG__
    explicit Thread(const std::string& name, int32_t cpuId, __UT_THREAD_TMPL_FUNC_ARG__)
        : Thread(name, cpuId)
    {
        Run(__UT_THREAD_BIND_FUNC_ARG__);
    }

    virtual ~Thread();

    uint64_t GetThreadId() const;

    void SetCpu();
    void SetName();
    void SetPriority(int32_t priority);

    void Wrap();

protected:
    __UT_THREAD_DECL_TMPL_FUNC_ARG__
    void Run(__UT_THREAD_TMPL_FUNC_ARG__)
    {
        mFunc = std::bind(__UT_THREAD_BIND_FUNC_ARG__);
        CreateThreadNative();
    }

    void CreateThreadNative();

protected:
    pthread_t mThreadId;
    std::string mName;
    int32_t mCpuId;
    std::function<Any()> mFunc;
};

typedef std::shared_ptr<Thread> ThreadPtr;

__UT_THREAD_DECL_TMPL_FUNC_ARG__
static inline ThreadPtr CreateThread(__UT_THREAD_TMPL_FUNC_ARG__)
{
    return ThreadPtr(new Thread(__UT_THREAD_BIND_FUNC_ARG__));
}

__UT_THREAD_DECL_TMPL_FUNC_ARG__
static inline ThreadPtr CreateThreadEx(const std::string& name, int32_t cpuId, __UT_THREAD_TMPL_FUNC_ARG__)
{
    return ThreadPtr(new Thread(name, cpuId, __UT_THREAD_BIND_FUNC_ARG__));
}

}
}

#endif//__UT_THREAD_HPP__
