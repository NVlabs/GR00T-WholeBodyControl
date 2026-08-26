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
#ifndef CYCLONEDDS_DDS_CORE_COND_TREADCONDITION_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_COND_TREADCONDITION_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/cond/TReadCondition.hpp>
#include <org/eclipse/cyclonedds/sub/cond/ReadConditionDelegate.hpp>

// Implementation

namespace dds
{
namespace sub
{
namespace cond
{

template <typename DELEGATE>
TReadCondition<DELEGATE>::TReadCondition(
    const dds::sub::AnyDataReader& dr,
    const dds::sub::status::DataState& status)
{
    this->set_ref(new DELEGATE(dr, status));
	this->delegate()->init(this->impl_);
}

template <typename DELEGATE>
template <typename FUN>
TReadCondition<DELEGATE>::TReadCondition(
        const dds::sub::AnyDataReader& dr,
        const dds::sub::status::DataState& status,
        FUN functor)
{
    this->set_ref(new DELEGATE(dr, status));
	this->delegate()->init(this->impl_);
    this->delegate()->set_handler(functor);
}

template <typename DELEGATE>
const dds::sub::status::DataState TReadCondition<DELEGATE>::state_filter() const
{
    return this->delegate()->state_filter();
}

template <typename DELEGATE>
const AnyDataReader& TReadCondition<DELEGATE>::data_reader() const
{
    return this->delegate()->data_reader();
}

}
}
namespace core
{
namespace cond
{
template <typename DELEGATE>
TCondition<DELEGATE>::TCondition(const dds::sub::cond::TReadCondition<org::eclipse::cyclonedds::sub::cond::ReadConditionDelegate>& h)
{
    if (h.is_nil()) {
        /* We got a null object and are not really able to do a typecheck here. */
        /* So, just set a null object. */
        *this = dds::core::null;
    } else {
        this->::dds::core::Reference<DELEGATE>::impl_ = ::std::dynamic_pointer_cast<DELEGATE_T>(h.delegate());
        if (h.delegate() != this->::dds::core::Reference<DELEGATE>::impl_) {
            throw dds::core::IllegalOperationError(std::string("Attempted invalid cast: ") + typeid(h).name() + " to " + typeid(*this).name());
        }
    }
}

template <typename DELEGATE>
TCondition<DELEGATE>&
TCondition<DELEGATE>::operator=(const dds::sub::cond::TReadCondition<org::eclipse::cyclonedds::sub::cond::ReadConditionDelegate>& rhs)
{
    const TCondition<DELEGATE> &t = rhs;
    if (this != &t) {
        if (rhs.is_nil()) {
            /* We got a null object and are not really able to do a typecheck here. */
            /* So, just set a null object. */
            *this = dds::core::null;
        } else {
            TCondition other(rhs);
            /* Dont have to copy when the delegate is the same. */
            if (other.delegate() != this->::dds::core::Reference<DELEGATE>::impl_) {
                *this = other;
            }
        }
    }
    return *this;
}
}
}
}
// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_COND_TREADCONDITION_IMPL_HPP_ */
