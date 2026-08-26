#ifndef __UT_FUTURE_HPP__
#define __UT_FUTURE_HPP__

#include <unitree/common/thread/thread_decl.hpp>

namespace unitree
{
namespace common
{
class Future
{
public:
    enum
    {
       DEFER = 0,
       READY = 1,
       FAULT = 2
    };

    Future()
    {}

    virtual ~Future()
    {}

    bool IsDeferred()
    {
        return GetState() == DEFER;
    }

    bool IsReady()
    {
        return GetState() == READY;
    }

    bool IsFault()
    {
        return GetState() == FAULT;
    }

    virtual int32_t GetState() = 0;
    virtual bool Wait(int64_t microsec = 0) = 0;
    virtual const Any& GetValue(int64_t microsec = 0) = 0;
    virtual const Any& GetFaultMessage() = 0;

public:
    virtual void Ready(const Any& value) = 0;
    virtual void Fault(const Any& message) = 0;
};

typedef std::shared_ptr<Future> FuturePtr;

class FutureWrapper : public Future
{
public:
    FutureWrapper();
    virtual ~FutureWrapper();

    virtual int32_t GetState()
    {
        return mFuturePtr->GetState();       
    }

    virtual bool Wait(int64_t microsec = 0)
    {
        return mFuturePtr->Wait(microsec);
    }

    virtual const Any& GetValue(int64_t microsec = 0)
    {
        return mFuturePtr->GetValue(microsec);
    }

    virtual const Any& GetFaultMessage()
    {
        return mFuturePtr->GetFaultMessage();
    }

    std::shared_ptr<Future> GetFuture()
    {
        return mFuturePtr;
    }

public:
    virtual void Ready(const Any& value)
    {
        return mFuturePtr->Ready(value);
    }

    virtual void Fault(const Any& message)
    {
        return mFuturePtr->Fault(message);
    }

protected:
    std::shared_ptr<Future> mFuturePtr;
};

}
}

#endif//__UT_FUTURE_HPP__
