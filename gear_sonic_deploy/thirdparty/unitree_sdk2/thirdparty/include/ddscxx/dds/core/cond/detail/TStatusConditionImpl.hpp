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
#ifndef CYCLONEDDS_DDS_CORE_COND_TSTATUSCONDITION_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_COND_TSTATUSCONDITION_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/cond/TStatusCondition.hpp>
#include <org/eclipse/cyclonedds/core/cond/StatusConditionDelegate.hpp>
#include <org/eclipse/cyclonedds/core/ReportUtils.hpp>

// Implementation
namespace dds
{
namespace core
{
namespace cond
{

template <typename DELEGATE>
TStatusCondition<DELEGATE>::TStatusCondition(const dds::core::Entity& e)
{
    dds::core::Reference<DELEGATE>::impl_=
            ::std::dynamic_pointer_cast<org::eclipse::cyclonedds::core::cond::StatusConditionDelegate>(
                        e.delegate()->get_statusCondition());
}

template <typename DELEGATE>
template <typename FUN>
TStatusCondition<DELEGATE>::TStatusCondition(const dds::core::Entity& e, FUN functor)
{
    dds::core::Reference<DELEGATE>::impl_=
            ::std::dynamic_pointer_cast<org::eclipse::cyclonedds::core::cond::StatusConditionDelegate>(
                          e.delegate()->get_statusCondition());
    this->delegate()->set_handler(functor);
}

template <typename DELEGATE>
void TStatusCondition<DELEGATE>::enabled_statuses(const dds::core::status::StatusMask& status) const
{
    this->delegate()->enabled_statuses(status);
}

template <typename DELEGATE>
const dds::core::status::StatusMask TStatusCondition<DELEGATE>::enabled_statuses() const
{
    return this->delegate()->enabled_statuses();
}

template <typename DELEGATE>
const dds::core::Entity& TStatusCondition<DELEGATE>::entity() const
{
    return this->delegate()->entity();
}


template <typename DELEGATE>
TCondition<DELEGATE>::TCondition(const dds::core::cond::TStatusCondition<org::eclipse::cyclonedds::core::cond::StatusConditionDelegate>& h)
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
TCondition<DELEGATE>::operator=(const dds::core::cond::TStatusCondition<org::eclipse::cyclonedds::core::cond::StatusConditionDelegate>& rhs)
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

#endif /* CYCLONEDDS_DDS_CORE_COND_TSTATUSCONDITION_IMPL_HPP_ */
