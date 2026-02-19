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
#ifndef CYCLONEDDS_DDS_SUB_COND_TQUERYCONDITION_IMPL_HPP_
#define CYCLONEDDS_DDS_SUB_COND_TQUERYCONDITION_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/cond/TQueryCondition.hpp>
#include <org/eclipse/cyclonedds/sub/cond/QueryConditionDelegate.hpp>

// Implementation

namespace dds
{
namespace sub
{
namespace cond
{

template <typename DELEGATE>
TQueryCondition<DELEGATE>::TQueryCondition(
    const dds::sub::Query& query,
    const dds::sub::status::DataState& status)
{
    this->set_ref(new DELEGATE(query.data_reader(),
                               query.expression(), query.delegate()->parameters(), status));
    this->delegate()->init(this->impl_);
}

template <typename DELEGATE>
template <typename FUN>
TQueryCondition<DELEGATE>::TQueryCondition(
    const dds::sub::Query& query,
    const dds::sub::status::DataState& status, FUN functor)
{
    this->set_ref(new DELEGATE(query.data_reader(),
                               query.expression(), query.delegate()->parameters(), status));
    this->delegate()->set_handler(functor);
    this->delegate()->init(this->impl_);
}

template <typename DELEGATE>
TQueryCondition<DELEGATE>::TQueryCondition(
    const dds::sub::AnyDataReader& dr,
    const std::string& expression,
    const std::vector<std::string>& params,
    const dds::sub::status::DataState& status)
{
    this->set_ref(new DELEGATE(dr, expression, params, status));
    this->delegate()->init(this->impl_);
}

template <typename DELEGATE>
TQueryCondition<DELEGATE>::TQueryCondition(
    const dds::sub::AnyDataReader& dr,
    const dds::sub::status::DataState& status)
{
    this->set_ref(new DELEGATE(dr, status));
    this->delegate()->init(this->impl_);
}

template <typename DELEGATE>
template <typename FUN>
TQueryCondition<DELEGATE>::TQueryCondition(
    const dds::sub::AnyDataReader& dr,
    const std::string& expression,
    const std::vector<std::string>& params,
    const dds::sub::status::DataState& status,
    FUN functor)
{
    this->set_ref(new DELEGATE(dr, expression, params, status));
    this->delegate()->set_handler(functor);
    this->delegate()->init(this->impl_);
}

template <typename DELEGATE>
template<typename FWIterator>
void TQueryCondition<DELEGATE>::parameters(const FWIterator& begin, const FWIterator end)
{
    std::vector<std::string> params(begin, end);
    this->delegate()->parameters(params);
}

template <typename DELEGATE>
void TQueryCondition<DELEGATE>::expression(
    const std::string& expr)
{
    this->delegate()->expression(expr);
}

template <typename DELEGATE>
const std::string& TQueryCondition<DELEGATE>::expression()
{
    return this->delegate()->expression();
}

template <typename DELEGATE>
typename TQueryCondition<DELEGATE>::const_iterator TQueryCondition<DELEGATE>::begin() const
{
    return this->delegate()->begin();
}

template <typename DELEGATE>
typename TQueryCondition<DELEGATE>::const_iterator TQueryCondition<DELEGATE>::end() const
{
    return this->delegate()->end();
}

template <typename DELEGATE>
typename TQueryCondition<DELEGATE>::iterator TQueryCondition<DELEGATE>::begin()
{
    return this->delegate()->begin();
}

template <typename DELEGATE>
typename TQueryCondition<DELEGATE>::iterator TQueryCondition<DELEGATE>::end()
{
    return this->delegate()->end();
}

template <typename DELEGATE>
void TQueryCondition<DELEGATE>::add_parameter(
    const std::string& param)
{
    this->delegate()->add_parameter(param);
}

template <typename DELEGATE>
uint32_t TQueryCondition<DELEGATE>::parameters_length() const
{
    return this->delegate()->parameters_length();
}

template <typename DELEGATE>
const AnyDataReader& TQueryCondition<DELEGATE>::data_reader() const
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
TCondition<DELEGATE>::TCondition(const dds::sub::cond::TQueryCondition<org::eclipse::cyclonedds::sub::cond::QueryConditionDelegate>& h)
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
TCondition<DELEGATE>::operator=(const dds::sub::cond::TQueryCondition<org::eclipse::cyclonedds::sub::cond::QueryConditionDelegate>& rhs)
{
    if (this != static_cast<TCondition*>(&rhs)) {
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

#endif /* CYCLONEDDS_DDS_SUB_COND_TQUERYCONDITION_IMPL_HPP_ */
