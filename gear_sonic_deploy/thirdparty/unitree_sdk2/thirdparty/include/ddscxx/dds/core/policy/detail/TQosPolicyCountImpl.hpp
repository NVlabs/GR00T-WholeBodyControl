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
#ifndef CYCLONEDDS_DDS_CORE_POLICY_TQOSPOLICYCOUNT_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_POLICY_TQOSPOLICYCOUNT_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/policy/TQosPolicyCount.hpp>

// Implementation

namespace dds
{
namespace core
{
namespace policy
{

template <typename D>
TQosPolicyCount<D>::TQosPolicyCount(QosPolicyId policy_id, int32_t count) : dds::core::Value<D>(policy_id, count) { }

#if defined(__GNUC__) && (__GNUC__ >= 10)
_Pragma("GCC diagnostic push")
_Pragma("GCC diagnostic ignored \"-Wanalyzer-null-dereference\"")
#endif

template <typename D>
TQosPolicyCount<D>::TQosPolicyCount(const TQosPolicyCount& other) : dds::core::Value<D>(other.policy_id(), other.count()) { }

#if defined(__GNUC__) && (__GNUC__ >= 10)
_Pragma("GCC diagnostic pop")
#endif

template <typename D> QosPolicyId TQosPolicyCount<D>::policy_id() const
{
    return this->delegate().policy_id();
}

template <typename D>
int32_t TQosPolicyCount<D>::count() const
{
    return this->delegate().count();
}

}
}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_POLICY_TQOSPOLICYCOUNT_IMPL_HPP_ */
