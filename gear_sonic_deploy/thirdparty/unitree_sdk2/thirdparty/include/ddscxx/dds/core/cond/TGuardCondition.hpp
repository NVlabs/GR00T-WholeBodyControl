#ifndef OMG_TDDS_CORE_COND_GUARD_CONDITION_HPP_
#define OMG_TDDS_CORE_COND_GUARD_CONDITION_HPP_

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

#include <dds/core/cond/Condition.hpp>


namespace dds
{
namespace core
{
namespace cond
{
template <typename DELEGATE>
class TGuardCondition;
}
}
}

/**
 * @brief
 * A GuardCondition object is a specific Condition whose trigger_value is
 * completely under the control of the application.
 *
 * When a GuardCondition is initially created, the trigger_value is FALSE.
 *
 * The purpose of the GuardCondition is to provide the means for the
 * application to manually triggering a WaitSet to stop waiting. This is accomplished by
 * attaching the GuardCondition to the WaitSet and then setting the
 * trigger_value by means of the set trigger_value operation.
 *
 * @code{.cpp}
 * dds::core::cond::GuardCondition guard;
 * dds::core::cond::WaitSet waitset;
 * waitset.attach_condition(guard);
 * waitset.wait();
 * ...
 * // To wakeup waitset, do in another thread:
 * guard.trigger_value(true);
 * @endcode
 * See the @ref anchor_dds_core_cond_waitset_examples "WaitSet examples" for more examples.<br>
 * Although the WaitSet examples use the StatusCondition, the basic usage of this Condition
 * with a WaitSet is the same.
 *
 * @see dds::core::cond::Condition
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "WaitSet concept"
 * @see for more information: @ref anchor_dds_core_cond_waitset_examples "WaitSet examples"
 */
template <typename DELEGATE>
class dds::core::cond::TGuardCondition : public dds::core::cond::TCondition<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_NO_DC(TGuardCondition, dds::core::cond::TCondition, DELEGATE)
    OMG_DDS_EXPLICIT_REF_BASE(TGuardCondition, dds::core::cond::Condition)

public:
    /**
     * Create a dds::core::cond::GuardCondition.
     *
     * The GuardCondition can then be added to a dds::core::cond::WaitSet so that the
     * application can manually wake up a thread that is blocked on that WaitSet.
     *
     * @throw  dds::core::Exception
     */
    TGuardCondition();

    /**
     * Create a dds::core::cond::GuardCondition.
     *
     * The GuardCondition can then be added to a dds::core::cond::WaitSet so that the
     * application can manually wake up a thread that is blocked on that WaitSet.
     *
     * The supplied functor will be called when this GuardCondition is triggered
     * and either the inherited dds::core::cond::Condition::dispatch() is called or the
     * dds::core::cond::WaitSet::dispatch() on the WaitSet to which this GuardCondition is
     * attached to.
     *
     * @param functor The functor to be called when the GuardCondition triggers.
     * @throw  dds::core::Exception
     */
    template <typename FUN>
    TGuardCondition(FUN functor);

public:

    /**
     * This operation sets the trigger_value of the GuardCondition.
     *
     * A GuardCondition object is a specific Condition which trigger_value is
     * completely under the control of the application. This operation must be used by the
     * application to manually wake-up a WaitSet. This operation sets the
     * trigger_value of the GuardCondition to the parameter value. The
     * GuardCondition is directly created using the GuardCondition constructor.
     * When a GuardCondition is initially created, the trigger_value is FALSE.
     *
     * @param value The boolean value to which the GuardCondition is set.
     * @throw dds::core::Exception
     */
    void trigger_value(bool value);

    /**
     * @copydoc dds::core::cond::TCondition::trigger_value()
     */
    bool trigger_value();
};

#endif /* OMG_TDDS_CORE_GUARD_CONDITION_HPP_ */
