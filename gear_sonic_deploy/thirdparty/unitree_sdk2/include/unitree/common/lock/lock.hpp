#ifndef __UT_LOCK_HPP__
#define __UT_LOCK_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
namespace common
{
enum
{
    UT_LOCK_MODE_READ   = 0,
    UT_LOCK_MODE_WRITE  = 1,
    UT_LOCK_MODE_UNLCK  = 2
};

enum
{
    UT_LOCK_ENO_INTR     = EINTR,
    UT_LOCK_ENO_BUSY     = EBUSY,
    UT_LOCK_ENO_TIMEDOUT = ETIMEDOUT
};

class Spinlock
{
public:
    explicit Spinlock();
    ~Spinlock();

    void Lock();
    void Unlock();

    bool Trylock();

    pthread_spinlock_t & GetNative();

private:
    pthread_spinlock_t mNative;
};

class CaspinLock
{
public:
    explicit CaspinLock();
    ~CaspinLock();

    void Lock();
    void Unlock();

    bool Trylock();

private:
    volatile int32_t mData;
};

class Mutex
{
public:
    explicit Mutex();
    ~Mutex();

    void Lock();
    void Unlock();

    bool Trylock();

    pthread_mutex_t & GetNative();

private:
    pthread_mutex_t mNative;
};

class Cond
{
public:
    explicit Cond();
    ~Cond();

    void Wait(Mutex& mutex);
    bool Wait(Mutex& mutex, uint64_t microsec);

    void Notify();
    void NotifyAll();

private:
    pthread_cond_t mNative;
};

class MutexCond
{
public:
    explicit MutexCond();
    ~MutexCond();

    void Lock();
    void Unlock();

    bool Wait(int64_t microsec = 0);
    void Notify();
    void NotifyAll();

private:
    Mutex mMutex;
    Cond mCond;
};

class Rwlock
{
public:
    explicit Rwlock();
    ~Rwlock();

    void Lock(int32_t mode);
    void Unlock();

    bool Trylock(int32_t mode);

    pthread_rwlock_t & GetNative();

private:
    pthread_rwlock_t mNative;
};

class Filelock
{
public:
    explicit Filelock(const std::string& fileName);
    explicit Filelock(int32_t fd);
    ~Filelock();

    void Lock(int32_t mode, int64_t start = 0, int64_t len = 0);
    void Unlock();

    bool Trylock(int32_t mode, int64_t start = 0, int64_t len = 0);

private:
    void SetLockMember(int32_t mode, int64_t start, int64_t len);
    void SetLockMode(int32_t mode);

private:
    int32_t mFd;
    bool mCloseFd;
    struct flock mLock;
};

template<typename LOCK_TYPE>
class LockGuard
{
public:
    explicit LockGuard(LOCK_TYPE& lockPtr)
    {
        mLockPtr = &lockPtr;
        mLockPtr->Lock();
    }

    explicit LockGuard(LOCK_TYPE* lockPtr)
    {
        mLockPtr = lockPtr;
        mLockPtr->Lock();
    }

    ~LockGuard()
    {
        mLockPtr->Unlock();
    }

private:
    LOCK_TYPE* mLockPtr;
};

template<typename LOCK_TYPE>
class RwLockGuard
{
public:
    explicit RwLockGuard(LOCK_TYPE& lockPtr, int32_t mode)
    {
        mLockPtr = &lockPtr;
        lockPtr.Lock(mode);
    }

    explicit RwLockGuard(LOCK_TYPE* lockPtr, int32_t mode)
    {
        mLockPtr = lockPtr;
        lockPtr->Lock(mode);
    }

    ~RwLockGuard()
    {
        mLockPtr->Unlock();
    }

private:
    LOCK_TYPE* mLockPtr;
};
}
}

#endif//__UT_LOCK_HPP__
