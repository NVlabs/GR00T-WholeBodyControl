#ifndef __UT_BLOCK_QUEUE_HPP__
#define __UT_BLOCK_QUEUE_HPP__

#include <unitree/common/exception.hpp>
#include <unitree/common/lock/lock.hpp>

namespace unitree
{
namespace common
{
template<typename T>
class BlockQueue
{
public:
    BlockQueue(uint64_t maxSize = UT_QUEUE_MAX_LEN) :
        mMaxSize(maxSize), mCurSize(0)
    {
        if (mMaxSize == 0)
        {
            mMaxSize = UT_QUEUE_MAX_LEN;
        }
    }

    bool Put(const T& t, bool replace = false, bool putfront = false)
    {
        /*
         * if queue is full or full-replaced occured return false
         */
        bool noneReplaced = true;

        LockGuard<MutexCond> guard(mMutexCond);
        if (mCurSize >= mMaxSize)
        {
            if (!replace)
            {
                return false;
            }

            noneReplaced = false;

            mQueue.pop_front();
            mCurSize --;
        }

        if (putfront)
        {
            mQueue.emplace_front(t);
        }
        else
        {
            mQueue.emplace_back(t);
        }

        mCurSize ++;
        mMutexCond.Notify();

        return noneReplaced;
    }

    bool Get(T& t, uint64_t microsec = 0)
    {
        LockGuard<MutexCond> guard(mMutexCond);
        return GetTimeout(t, microsec);
    }

    T Get(uint64_t microsec = 0)
    {
        LockGuard<MutexCond> guard(mMutexCond);
        T t;
        if (GetTimeout(t, microsec))
        {
            return std::move(t);
        }

        UT_THROW(TimeoutException, "block queue get timeout or interrupted");
    }

    bool Empty()
    {
        return mCurSize == 0;
    }

    uint64_t Size()
    {
        return mCurSize;
    }

    void Interrupt(bool all = false)
    {
        LockGuard<MutexCond> guard(mMutexCond);
        if (all)
        {
            mMutexCond.NotifyAll();
        }
        else
        {
            mMutexCond.Notify();
        }
    }

private:
    bool GetTimeout(T& t, uint64_t microsec = 0)
    {
        if (mQueue.empty())
        {
            if (!mMutexCond.Wait(microsec))
            {
                return false;
            }
        
            if (mQueue.empty())
            {
                return false;
            }
        }

        t = mQueue.front();
        mQueue.pop_front();

        mCurSize--;

        return true;
    }

private:
    uint64_t mMaxSize;
    uint64_t mCurSize;
    std::list<T> mQueue;
    MutexCond mMutexCond;
};

template <typename T>
using BlockQueuePtr = std::shared_ptr<BlockQueue<T>>;

}
}
#endif//__UT_BLOCK_QUEUE_HPP__
