/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
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

#ifndef CYCLONEDDS_CORE_INSTANCE_HANDLE_HPP_
#define CYCLONEDDS_CORE_INSTANCE_HANDLE_HPP_

#include <dds/core/types.hpp>
#include <org/eclipse/cyclonedds/core/config.hpp>
#include <dds/dds.h>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{
class InstanceHandleDelegate;
}
}
}
}

class OMG_DDS_API org::eclipse::cyclonedds::core::InstanceHandleDelegate
{
public:
    InstanceHandleDelegate();
    InstanceHandleDelegate(dds_instance_handle_t h);
public:
    InstanceHandleDelegate(const dds::core::null_type& src);

public:
    bool operator==(const InstanceHandleDelegate& that) const;

    bool operator<(const InstanceHandleDelegate& that) const;

    bool operator>(const InstanceHandleDelegate& that) const;

    InstanceHandleDelegate& operator=(const dds::core::null_type& src);
    bool is_nil() const;

public:
    dds_instance_handle_t handle() const;

private:
    dds_instance_handle_t handle_;
};

inline std::ostream&
operator << (std::ostream& os,
             const org::eclipse::cyclonedds::core::InstanceHandleDelegate& h)
{
    os << h.handle();
    return os;
}
#endif /* CYCLONEDDS_CORE_INSTANCE_HANDLE_HPP_ */
