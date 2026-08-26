#ifndef OMG_TDDS_DDS_CORE_COND_CONDITION_HPP_
#define OMG_TDDS_DDS_CORE_COND_CONDITION_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Corp.
 * Copyright 2010, Real-Time Innovations, Inc.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dds/core/Reference.hpp>
#include <dds/core/cond/detail/GuardCondition.hpp>
#include <dds/core/cond/detail/StatusCondition.hpp>
#include <dds/sub/cond/detail/ReadCondition.hpp>
#include <dds/sub/cond/detail/QueryCondition.hpp>

namespace dds
{
namespace core
{
namespace cond
{
template <typename DELEGATE>
class TCondition;
}
}
}

/**
 * @brief
 * This class is the base class for all the conditions that may be attached to a dds::core::cond::WaitSet.
 *
 * This base class is specialized in three classes by the Data Distribution Service:
 * - dds::core::cond::GuardCondition
 * - dds::core::cond::StatusCondition
 * - dds::sub::cond::ReadCondition
 *      - dds::sub::cond::QueryCondition
 *
 * Each Condition has a trigger_value that can be TRUE or FALSE and is set by
 * the Data Distribution Service (except a GuardCondition) depending on the
 * evaluation of the Condition.
 *
 * @see for more information: @ref DCPS_Modules_Infrastructure_Status  "Status concept"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "WaitSet concept"
 */
template <typename DELEGATE>
class dds::core::cond::TCondition : public virtual dds::core::Reference<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TCondition, dds::core::Reference, DELEGATE)
    OMG_DDS_EXPLICIT_REF_BASE_DECL(TCondition, dds::core::cond::detail::StatusCondition)
    OMG_DDS_EXPLICIT_REF_BASE_DECL(TCondition, dds::core::cond::detail::GuardCondition)
    OMG_DDS_EXPLICIT_REF_BASE_DECL(TCondition, dds::sub::cond::detail::ReadCondition)
    OMG_DDS_EXPLICIT_REF_BASE_DECL(TCondition, dds::sub::cond::detail::QueryCondition)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_EXPLICIT(TCondition)

public:
    /**
     * Registers a functor as custom handler with this Condition.
     *
     * The supplied functor will be called when this Condition is triggered
     * and either the dds::core::cond::Condition::dispatch() is called or the
     * dds::core::cond::WaitSet::dispatch() on the WaitSet to which this
     * Condition is attached to.
     *
     * @param func The functor to be called when the StatusCondition triggers.
     * @throw  dds::core::Exception
     */
    template <typename Functor>
    void handler(Functor func);

    /**
     * Resets the handler for this Condition.
     *
     * After the invocation of this function no handler will be registered with
     * this Condition.
     *
     * @throw  dds::core::Exception
     */
    void reset_handler();

    /**
     * Dispatches the functor that have been registered with the Condition.
     *
     * The Condition has to have been triggered for the functor will be called
     * by this function.
     *
     * @throw  dds::core::Exception
     */
    void dispatch();

    /**
     * This operation retrieves the trigger_value of the Condition.
     *
     * A Condition has a trigger_value that can be TRUE or FALSE and is set by the
     * Data Distribution Service (except a GuardCondition). This operation returns the
     * trigger_value of the Condition.
     *
     * @return bool The boolean value to which the Condition is set.
     * @throw  dds::core::Exception
     */
    bool trigger_value() const;

};

#endif /* OMG_TDDS_DDS_CORE_CONDITION_HPP_ */
