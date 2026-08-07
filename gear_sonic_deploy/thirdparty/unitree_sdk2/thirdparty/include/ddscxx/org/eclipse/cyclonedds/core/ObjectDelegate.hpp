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

#ifndef CYCLONEDDS_CORE_OBJECT_DELEGATE_HPP_
#define CYCLONEDDS_CORE_OBJECT_DELEGATE_HPP_

#include "dds/core/macros.hpp"
#include "dds/core/refmacros.hpp"
#include "org/eclipse/cyclonedds/core/Mutex.hpp"

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{

DDSCXX_WARNING_MSVC_OFF(4251)

class OMG_DDS_API ObjectDelegate
{
public:

    typedef ::dds::core::smart_ptr_traits< ObjectDelegate >::ref_type ref_type;
    typedef ::dds::core::smart_ptr_traits< ObjectDelegate >::weak_ref_type weak_ref_type;

    ObjectDelegate ();
    virtual ~ObjectDelegate ();

    virtual void close ();
    void lock() const;
    void unlock() const;

    virtual void init (ObjectDelegate::weak_ref_type weak_ref) = 0;
    ObjectDelegate::weak_ref_type get_weak_ref () const;
    ObjectDelegate::ref_type get_strong_ref () const;

protected:

    void check () const;
    void set_weak_ref (ObjectDelegate::weak_ref_type weak_ref);

    Mutex mutex;
    bool closed;
    ObjectDelegate::weak_ref_type myself;
};

DDSCXX_WARNING_MSVC_ON(4251)

}
}
}
}

#endif /* CYCLONEDDS_CORE_OBJECT_DELEGATE_HPP_ */
