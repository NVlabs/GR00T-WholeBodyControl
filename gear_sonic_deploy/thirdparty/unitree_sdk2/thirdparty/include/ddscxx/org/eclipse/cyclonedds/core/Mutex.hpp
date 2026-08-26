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

#ifndef CYCLONEDDS_CORE_MUTEX_HPP_
#define CYCLONEDDS_CORE_MUTEX_HPP_

#include <dds/core/macros.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{

class OMG_DDS_API Mutex
{
public:
    Mutex();
    virtual ~Mutex();

    void lock() const;
    bool try_lock() const;
    void unlock() const;
private:
    void* mtx;
};

}
}
}
}


#endif /* CYCLONEDDS_CORE_MUTEX_HPP_ */
