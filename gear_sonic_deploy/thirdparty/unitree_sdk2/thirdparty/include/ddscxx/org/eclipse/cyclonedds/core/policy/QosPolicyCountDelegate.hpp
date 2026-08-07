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

#ifndef CYCLONEDDS_CORE_POLICY_QOS_POLICY_COUNT_DELEGATE_HPP_
#define CYCLONEDDS_CORE_POLICY_QOS_POLICY_COUNT_DELEGATE_HPP_

#include <dds/core/types.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{
namespace policy
{

class QosPolicyCountDelegate
{
public:
    QosPolicyCountDelegate(dds::core::policy::QosPolicyId id, int32_t count)
        : policy_id_(id),
          count_(count)
    { }

    QosPolicyCountDelegate(const QosPolicyCountDelegate& other)
        : policy_id_(other.policy_id()),
          count_(other.count())
    { }

    QosPolicyCountDelegate& operator=(const QosPolicyCountDelegate& other) = default;

public:
    dds::core::policy::QosPolicyId policy_id() const
    {
        return policy_id_;
    }
    void policy_id(dds::core::policy::QosPolicyId id)
    {
        policy_id_ = id;
    }

    int32_t count() const
    {
        return count_;
    }
    void count(int32_t c)
    {
        count_ = c;
    }

    bool operator ==(const QosPolicyCountDelegate& other) const
    {
        return other.policy_id_ == policy_id_ &&
               other.count_ == count_;
    }


private:
    dds::core::policy::QosPolicyId policy_id_;
    int32_t count_;
};

}
}
}
}
}

#endif /* CYCLONEDDS_CORE_POLICY_QOS_POLICY_COUNT_DELEGATE_HPP_ */
