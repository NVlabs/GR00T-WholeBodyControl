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
#ifndef CYCLONEDDS_DDS_CORE_TENTITYQOS_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_TENTITYQOS_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/TEntityQos.hpp>

// Implementation
namespace dds
{
namespace core
{

template <typename DELEGATE>
TEntityQos<DELEGATE>::TEntityQos() : dds::core::Value<DELEGATE>() { }

template <typename DELEGATE>
TEntityQos<DELEGATE>::TEntityQos(const TEntityQos& other)
    : dds::core::Value<DELEGATE>(other.delegate()) { }

template <typename DELEGATE>
TEntityQos<DELEGATE>::TEntityQos(TEntityQos&& other)
    : dds::core::Value<DELEGATE>(other.delegate()) { }

template <typename DELEGATE>
template <typename T>
TEntityQos<DELEGATE>::TEntityQos(const TEntityQos<T>& qos) :
    dds::core::Value<DELEGATE>(qos.delegate()) { }

template <typename DELEGATE>
template <typename POLICY>
TEntityQos<DELEGATE>& TEntityQos<DELEGATE>::policy(const POLICY& p)
{
    this->dds::core::Value<DELEGATE>::delegate().policy(p);
    return *this;
}

template <typename DELEGATE>
template <typename POLICY>
const POLICY& TEntityQos<DELEGATE>::policy() const
{
    return this->delegate().template policy<POLICY>();
}

template <typename DELEGATE>
template <typename POLICY>
POLICY& TEntityQos<DELEGATE>::policy()
{
    return this->delegate().template policy<POLICY>();
}

template <typename DELEGATE>
template <typename POLICY>
TEntityQos<DELEGATE>& TEntityQos<DELEGATE>::operator << (const POLICY& p)
{
    this->policy(p);
    return *this;
}

template <typename DELEGATE>
template <typename POLICY>
const TEntityQos<DELEGATE>& TEntityQos<DELEGATE>::operator >> (POLICY& p) const
{
    p = this->policy<POLICY>();
    return *this;
}

template <typename DELEGATE>
template <typename T>
TEntityQos<DELEGATE>& TEntityQos<DELEGATE>::operator = (const TEntityQos<T>& other)
{
    if(this != reinterpret_cast<const TEntityQos<DELEGATE>*>(&other))
    {
        this->d_ = other.delegate();
    }
    return *this;
}

}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_TENTITYQOS_IMPL_HPP_ */
