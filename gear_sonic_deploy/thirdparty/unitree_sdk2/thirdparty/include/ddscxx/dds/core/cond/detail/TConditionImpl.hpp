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
#ifndef CYCLONEDDS_DDS_CORE_COND_TCONDITION_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_COND_TCONDITION_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/detail/ReferenceImpl.hpp>
#include <dds/core/cond/TCondition.hpp>
#include <org/eclipse/cyclonedds/core/ReportUtils.hpp>

// Implementation
namespace dds
{
namespace core
{
namespace cond
{

template <typename DELEGATE>
template <typename Functor>
void TCondition<DELEGATE>::handler(Functor func)
{
    this->delegate()->set_handler(func);
}

template <typename DELEGATE>
void TCondition<DELEGATE>::reset_handler()
{
    this->delegate()->reset_handler();
}

template <typename DELEGATE>
void TCondition<DELEGATE>::dispatch()
{
    this->delegate()->dispatch();
}

template <typename DELEGATE>
bool TCondition<DELEGATE>::trigger_value() const
{
    return this->delegate()->trigger_value();
}

}
}
}
// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_COND_TCONDITION_IMPL_HPP_ */
