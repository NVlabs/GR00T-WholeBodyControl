/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */


/**
 * @file
 */

#ifndef CYCLONEDDS_CORE_SCOPEDLOCK_HPP_
#define CYCLONEDDS_CORE_SCOPEDLOCK_HPP_

#include <org/eclipse/cyclonedds/core/Mutex.hpp>
#include <org/eclipse/cyclonedds/core/ObjectDelegate.hpp>

#include <cassert>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{

template <typename LOCKABLE>
class OMG_DDS_API ScopedLock
{
public:
    ScopedLock(const LOCKABLE& obj, bool lock = true)
        : lockable(obj),
          owner(lock)
    {
        if (lock) {
            lockable.lock();
        }
    }

    virtual ~ScopedLock()
    {
        if (owner) {
            try {
                lockable.unlock();
            } catch (...) {
                /* Don't know what to do anymore (it should have never failed)... */
                assert(false);
            }
        }
    }

    void lock()
    {
        assert(!owner);
        lockable.lock();
        owner = true;
    }

    bool try_lock()
    {
        bool locked;
        locked = lockable.try_lock();
        if (locked) {
            owner = true;
        }
        return locked;
    }

    void unlock()
    {
        assert(owner);
        owner = false;
        lockable.unlock();
    }

    bool own() const
    {
        return owner;
    }

private:
    const LOCKABLE& lockable;
    bool owner;
};

typedef ScopedLock<Mutex> ScopedMutexLock;
typedef ScopedLock<ObjectDelegate> ScopedObjectLock;

}
}
}
}


#endif /* CYCLONEDDS_CORE_SCOPEDLOCK_HPP_ */
