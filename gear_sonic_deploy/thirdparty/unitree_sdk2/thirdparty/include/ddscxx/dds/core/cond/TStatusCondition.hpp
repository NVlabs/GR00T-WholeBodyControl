#ifndef OMG_DDS_CORE_T_STATUS_CONDITION_HPP_
#define OMG_DDS_CORE_T_STATUS_CONDITION_HPP_

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

#include <dds/core/status/State.hpp>
#include <dds/core/cond/Condition.hpp>
#include <dds/core/cond/detail/StatusCondition.hpp>
#include <dds/core/Entity.hpp>

namespace dds
{
namespace core
{
namespace cond
{
template <typename DELEGATE>
class TStatusCondition;
}
}
}

/**
 * @brief
 * A StatusCondition object is a specific Condition that is associated with each Entity.
 *
 * Entity objects that have status attributes also have a StatusCondition, access is
 * provided to the application by the get_statuscondition operation.
 * The communication statuses whose changes can be communicated to the application
 * depend on the Entity. The following table shows the relevant statuses for each
 * Entity.
 *
 * Entity               | Status Name
 * -------------------- | --------------------
 * dds::topic::Topic    | dds::core::status::StatusMask::inconsistent_topic()
 * dds::sub::Subscriber | dds::core::status::StatusMask::data_on_readers()
 * dds::sub::DataReader | dds::core::status::StatusMask::sample_rejected() <br> dds::core::status::StatusMask::liveliness_changed() <br> dds::core::status::StatusMask::requested_deadline_missed() <br> dds::core::status::StatusMask::requested_incompatible_qos() <br> dds::core::status::StatusMask::data_available() <br> dds::core::status::StatusMask::sample_lost() <br> dds::core::status::StatusMask::subscription_matched()
 * dds::pub::DataWriter | dds::core::status::StatusMask::liveliness_lost() <br> dds::core::status::StatusMask::offered_deadline_missed() <br> dds::core::status::StatusMask::offered_incompatible_qos() <br> dds::core::status::StatusMask::publication_matched()
 *
 * The inherited dds::core::cond::Condition::trigger_value() of the StatusCondition
 * depends on the communication statuses of that Entity (e.g., missed deadline) and
 * also depends on the value of the dds::core::status::StatusMask.
 *
 * A StatusCondition can be attached to a WaitSet in order to allow an application
 * to suspend until the trigger_value has become TRUE.
 *
 * The trigger_value of a StatusCondition will be TRUE if one of the enabled
 * StatusChangedFlags is set. That is, trigger_value==FALSE only if all the
 * values of the StatusChangedFlags are FALSE.
 *
 * The sensitivity of the StatusCondition to a particular communication status is
 * controlled by the list of enabled_statuses set on the condition by means of
 * dds::core::cond::StatusCondition::enabled_statuses(const ::dds::core::status::StatusMask& status)
 * When the enabled_statuses are not changed by that
 * operation, all statuses are enabled by default.
 *
 * See the @ref anchor_dds_core_cond_waitset_examples "WaitSet examples" for examples
 * how to use this Condition.
 *
 * @see dds::core::cond::Condition
 * @see for more information: @ref DCPS_Modules_Infrastructure_Status  "Status concept"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "WaitSet concept"
 * @see for more information: @ref anchor_dds_core_cond_waitset_examples "WaitSet examples"
 */
template <typename DELEGATE>
class dds::core::cond::TStatusCondition : public dds::core::cond::TCondition<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_DELEGATE_C(TStatusCondition, dds::core::cond::TCondition, DELEGATE)
    OMG_DDS_EXPLICIT_REF_BASE(TStatusCondition, dds::core::cond::Condition)

    /**
     * Create a dds::core::cond::StatusCondition associated with an Entity.
     *
     * The StatusCondition can then be added to a dds::core::cond::WaitSet so that the
     * application can wait for specific status changes that affect the Entity.
     *
     * @param  e The Entity to associate with the StatusCondition.
     * @throw  dds::core::Exception
     */
    TStatusCondition(const dds::core::Entity& e);

    /**
     * Create a dds::core::cond::StatusCondition associated with an Entity.
     *
     * The StatusCondition can then be added to a dds::core::cond::WaitSet so that the
     * application can wait for specific status changes that affect the Entity.
     *
     * The supplied functor will be called when this StatusCondition is triggered
     * and either the inherited dds::core::cond::Condition::dispatch() is called or the
     * dds::core::cond::WaitSet::dispatch() on the WaitSet to which this StatusCondition is
     * attached to.
     *
     * @param  e       The Entity to associate with the StatusCondition.
     * @param functor The functor to be called when the StatusCondition triggers.
     * @throw  dds::core::Exception
     */
    template <typename FUN>
    TStatusCondition(const dds::core::Entity& e, FUN functor);

public:
    /**
     * This operation sets the list of communication statuses that are taken into account to
     * determine the trigger_value of the StatusCondition.
     *
     * The inherited dds::core::cond::Condition::trigger_value() of the StatusCondition
     * depends on the communication status of that Entity (e.g., missed deadline,
     * loss of information, etc.), ‘filtered’ by the set of enabled_statuses on the StatusCondition.
     *
     * This operation sets the list of communication statuses that are taken into account to
     * determine the trigger_value of the StatusCondition. This operation may
     * change the trigger_value of the StatusCondition.
     *
     * dds::core::cond::WaitSet objects behaviour depend on the changes of the trigger_value of
     * their attached Conditions. Therefore, any WaitSet to which the StatusCondition
     * is attached is potentially affected by this operation.
     * If this function is not invoked, the default list of enabled_statuses includes all
     * the statuses.
     *
     * The result value is a bit mask in which each bit shows which value has changed. The
     * relevant bits represent one of the following statuses:
     *  - dds::core::status::StatusMask::inconsistent_topic()
     *  - dds::core::status::StatusMask::offered_deadline_missed()
     *  - dds::core::status::StatusMask::requested_deadline_missed()
     *  - dds::core::status::StatusMask::offered_incompatible_qos()
     *  - dds::core::status::StatusMask::requested_incompatible_qos()
     *  - dds::core::status::StatusMask::sample_lost()
     *  - dds::core::status::StatusMask::sample_rejected()
     *  - dds::core::status::StatusMask::data_on_readers()
     *  - dds::core::status::StatusMask::data_available()
     *  - dds::core::status::StatusMask::liveliness_lost()
     *  - dds::core::status::StatusMask::liveliness_changed()
     *  - dds::core::status::StatusMask::publication_matched()
     *  - dds::core::status::StatusMask::subscription_matched()
     *
     * Each status bit is declared as a constant and can be used in an AND operation to
     * check the status bit against the result of type StatusMask. Not all statuses are
     * relevant to all Entity objects. See the respective Listener interfaces for each
     * Entity for more information.
     *
     * @param  status A bit mask in which each bit sets the status which is taken
     *                into account for the StatusCondition.the enabled statuses.
     * @throw  dds::core::AlreadyClosedError
     * @throw  dds::core::Error
     */
    void
    enabled_statuses(const ::dds::core::status::StatusMask& status) const;

    /**
     * This operation returns the list of enabled communication statuses of the
     * StatusCondition.
     *
     * The inherited dds::core::cond::Condition::trigger_value() of the StatusCondition
     * depends on the communication status of that Entity (e.g., missed deadline,
     * loss of information, etc.), ‘filtered’ by the set of enabled_statuses on the StatusCondition.
     *
     * This operation returns the list of communication statuses that are taken into account
     * to determine the trigger_value of the StatusCondition. This operation
     * returns the statuses that were explicitly set on the last call to
     * dds::core::cond::StatusCondition::enabled_statuses(const ::dds::core::status::StatusMask& status) const
     * or, if enabled_statuses(status) was never called, the default list.
     *
     * The result value is a bit mask in which each bit shows which value has changed. The
     * relevant bits represent one of the following statuses:
     *  - dds::core::status::StatusMask::inconsistent_topic()
     *  - dds::core::status::StatusMask::offered_deadline_missed()
     *  - dds::core::status::StatusMask::requested_deadline_missed()
     *  - dds::core::status::StatusMask::offered_incompatible_qos()
     *  - dds::core::status::StatusMask::requested_incompatible_qos()
     *  - dds::core::status::StatusMask::sample_lost()
     *  - dds::core::status::StatusMask::sample_rejected()
     *  - dds::core::status::StatusMask::data_on_readers()
     *  - dds::core::status::StatusMask::data_available()
     *  - dds::core::status::StatusMask::liveliness_lost()
     *  - dds::core::status::StatusMask::liveliness_changed()
     *  - dds::core::status::StatusMask::publication_matched()
     *  - dds::core::status::StatusMask::subscription_matched()
     *
     * Each status bit is declared as a constant and can be used in an AND operation to
     * check the status bit against the result of type StatusMask. Not all statuses are
     * relevant to all Entity objects. See the respective Listener interfaces for each
     * Entity for more information.
     *
     * @return dds::core::status::StatusMask
     *              A bit mask in which each bit shows which status is taken into
     *              account for the StatusCondition.
     * @throw  dds::core::Exception
     */
    const ::dds::core::status::StatusMask enabled_statuses() const;

    /**
     * This operation returns the Entity associated with the StatusCondition
     *
     * Note that there is exactly one Entity associated with each StatusCondition.
     *
     * @return dds::core::Entity The Entity associated with the StatusCondition.
     * @throw  dds::core::AlreadyClosedError
     */
    const dds::core::Entity& entity() const;
};

#endif  /* OMG_DDS_CORE_T_STATUS_CONDITION_HPP_ */
