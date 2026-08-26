#ifndef OMG_TDDS_CORE_WAIT_SET_HPP_
#define OMG_TDDS_CORE_WAIT_SET_HPP_

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

#include <vector>

#include <dds/core/types.hpp>
#include <dds/core/Duration.hpp>
#include <dds/core/cond/Condition.hpp>

namespace dds
{
namespace core
{
namespace cond
{
template <typename DELEGATE>
class TWaitSet;
}
}
}


/**
 * @brief
 * A WaitSet object allows an application to wait until one or more of
 * the attached Condition objects has a trigger_value of TRUE or else
 * until the timeout expires.
 *
 * A WaitSet is not necessarily associated with a single DomainParticipant
 * and could be used to wait for Condition objects associated with different
 * DomainParticipant objects.
 *
 * @anchor anchor_dds_core_cond_waitset_examples
 * <b><i>Example with wait()</i></b><br>
 * When using the wait() operation, the triggered Conditions are returned in a list.
 * @code{.cpp}
 * // Create a Condition to attach to a Waitset
 * dds::core::cond::StatusCondition readerSC = dds::core::cond::StatusCondition(reader);
 * readerSC.enabled_statuses(dds::core::status::StatusMask::data_available());
 *
 * // Create WaitSet and attach Condition
 * dds::core::cond::WaitSet waitset;
 * waitset.attach_condition(readerSC); // or waitset += readerSC;
 *
 * dds::core::cond::WaitSet::ConditionSeq conditions;
 * while(true) {
 *     // Wait for any Condition to trigger.
 *     conditions = waitset.wait();
 *
 *     // Loop through the triggered conditions.
 *     for (int i=0; i < conditions.size(); i++) {
 *         // Handle data_available when right Condition triggered.
 *         if (conditions[i] == readerSC) {
 *             // Read samples from the DataReader
 *         }
 *     }
 * }
 * @endcode

 * <b><i>Example with dispatch()</i></b><br>
 * When using the dispatch() operation, the Functors of the triggered Conditions
 * will be called.
 * @code{.cpp}
 * // Functor to add to a Condition
 * class FunctorStatusCondition {
 * public:
 *     void operator()(const dds::core::cond::StatusCondition& condition) {
 *         // Possibly get reader from the condition and read some samples.
 *     }
 * };
 * FunctorStatusCondition functor;
 *
 * // Create a Condition with functor to attach to a Waitset
 * dds::core::cond::StatusCondition readerSC = dds::core::cond::StatusCondition(reader, functor);
 * readerSC.enabled_statuses(dds::core::status::StatusMask::data_available());
 *
 * // Create WaitSet and attach Condition
 * dds::core::cond::WaitSet waitset;
 * waitset.attach_condition(readerSC); // or waitset += readerSC;
 *
 * while(true) {
 *     // Wait for any Condition to trigger.
 *     // The functors of the Conditions are automatically called
 *     // when the Condition triggers.
 *     waitset.dispatch();
 * }
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "WaitSet concept"
 */
template <typename DELEGATE>
class dds::core::cond::TWaitSet : public dds::core::Reference<DELEGATE>
{
public:
    typedef std::vector<dds::core::cond::Condition> ConditionSeq;

public:
    OMG_DDS_REF_TYPE_NO_DC(TWaitSet, dds::core::Reference, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TWaitSet)

public:
    /**
     * Create a WaitSet instance.
     *
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    TWaitSet();

public:
    /**
     * This operation allows an application thread to wait for the occurrence
     * of at least one of the conditions that is attached to the WaitSet.
     *
     * This operation allows an application thread to wait for the occurrence
     * of certain Conditions. If none of the Conditions attached to the
     * WaitSet have a trigger_value of TRUE, the wait operation will block
     * suspending the calling thread.
     *
     * The wait operation takes a timeout argument that specifies the maximum
     * duration for the wait. If this duration is exceeded and none of
     * the attached Condition objects is true, a TimeoutError will be thrown.
     *
     * It is not allowed for more than one application thread to be waiting
     * on the same WaitSet. If the wait operation is invoked on a WaitSet that
     * already has a thread blocking on it, the operation will immediately
     * raise a PreconditionNotMetError exception.
     *
     * The result of the wait operation is the list of all the attached
     * Conditions that have a trigger_value of TRUE (i.e., the Conditions
     * that unblocked the wait).
     *
     * @param timeout   The maximum amount of time for which the wait
     *                  should block while waiting for a Condition to be triggered.
     * @return ConditionSeq
     *                  A vector containing the triggered Conditions
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::TimeoutError
     *                  The timeout has elapsed without any of the attached
     *                  conditions becoming TRUE.
     * @throws dds::core::PreconditionNotMetError
     *                  When multiple thread try to invoke the function concurrently.
     */
    const ConditionSeq wait(const dds::core::Duration& timeout);

    /**
     * This operation allows an application thread to wait for the occurrence
     * of at least one of the conditions that is attached to the WaitSet.
     *
     * This operation allows an application thread to wait for the occurrence
     * of certain Conditions. If none of the Conditions attached to the
     * WaitSet have a trigger_value of TRUE, the wait operation will block
     * suspending the calling thread.
     *
     * It is not allowed for more than one application thread to be waiting
     * on the same WaitSet. If the wait operation is invoked on a WaitSet that
     * already has a thread blocking on it, the operation will immediately
     * raise a PreconditionNotMetError exception.
     *
     * The result of the wait operation is the list of all the attached
     * Conditions that have a trigger_value of TRUE (i.e., the Conditions
     * that unblocked the wait).
     *
     * @return ConditionSeq
     *                  A vector containing the triggered Conditions
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::PreconditionNotMetError
     *                  When multiple thread try to invoke the function concurrently.
     */
    const ConditionSeq wait();

    /**
     * This operation allows an application thread to wait for the occurrence
     * of at least one of the conditions that is attached to the WaitSet.
     *
     * This operation allows an application thread to wait for the occurrence
     * of certain Conditions. If none of the Conditions attached to the
     * WaitSet have a trigger_value of TRUE, the wait operation will block
     * suspending the calling thread.
     *
     * The wait operation takes a timeout argument that specifies the maximum
     * duration for the wait. If this duration is exceeded and none of
     * the attached Condition objects is true, a TimeoutError will be thrown.
     *
     * It is not allowed for more than one application thread to be waiting
     * on the same WaitSet. If the wait operation is invoked on a WaitSet that
     * already has a thread blocking on it, the operation will immediately
     * raise a PreconditionNotMetError exception.
     *
     * The result of the wait operation is the list of all the attached
     * Conditions that have a trigger_value of TRUE (i.e., the Conditions
     * that unblocked the wait).
     *
     * @param triggered A ConditionSeq in which to put Conditions that were
     *                  triggered during the wait.
     * @param timeout   The maximum amount of time for which the wait should
     *                  block while waiting for a Condition to be triggered.
     * @return ConditionSeq
     *                  A vector containing the triggered Conditions
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::TimeoutError
     *                  The timeout has elapsed without any of the attached
     *                  conditions becoming TRUE.
     * @throws dds::core::PreconditionNotMetError
     *                  When multiple thread try to invoke the function concurrently.
     */
    ConditionSeq& wait(ConditionSeq& triggered,
                       const dds::core::Duration& timeout);

    /**
     * This operation allows an application thread to wait for the occurrence
     * of at least one of the conditions that is attached to the WaitSet.
     *
     * This operation allows an application thread to wait for the occurrence
     * of certain Conditions. If none of the Conditions attached to the
     * WaitSet have a trigger_value of TRUE, the wait operation will block
     * suspending the calling thread.
     *
     * It is not allowed for more than one application thread to be waiting
     * on the same WaitSet. If the wait operation is invoked on a WaitSet that
     * already has a thread blocking on it, the operation will immediately
     * raise a PreconditionNotMetError exception.
     *
     * The result of the wait operation is the list of all the attached
     * Conditions that have a trigger_value of TRUE (i.e., the Conditions
     * that unblocked the wait).
     *
     * @param triggered A ConditionSeq in which to put Conditions that were
     *                  triggered during the wait.
     * @return ConditionSeq
     *                  A vector containing the triggered Conditions
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::PreconditionNotMetError
     *                  When multiple thread try to invoke the function concurrently.
     */
    ConditionSeq& wait(ConditionSeq& triggered);

public:
    /**
     * Waits for at least one of the attached Conditions to trigger and then
     * dispatches the functor associated with the Condition.
     *
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::PreconditionNotMetError
     *                  When multiple thread try to invoke the function concurrently.
     */
    void dispatch();

    /**
     * Waits for at least one of the attached Conditions to trigger and then
     * dispatches the functor associated with the Condition, or, times
     * out and throws a TimeoutError.
     *
     * @param timeout   The maximum amount of time for which the dispatch should
     *                  block while waiting for a Condition to be triggered.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::TimeoutError
     *                  The timeout has elapsed without any of the attached
     *                  conditions becoming TRUE.
     * @throws dds::core::PreconditionNotMetError
     *                  When multiple thread try to invoke the function concurrently.
     */
    void dispatch(const dds::core::Duration& timeout);

public:
    /** @copydoc dds::core::cond::TWaitSet::attach_condition(const dds::core::cond::Condition& cond) */
    TWaitSet& operator +=(const dds::core::cond::Condition& cond);

    /** @copydoc dds::core::cond::TWaitSet::detach_condition(const dds::core::cond::Condition& cond) */
    TWaitSet& operator -=(const dds::core::cond::Condition& cond);

    /**
     * This operation attaches a Condition to the WaitSet.
     *
     * Attaches a Condition to the WaitSet. It is possible to attach a
     * Condition on a WaitSet that is currently being waited upon
     * (via the wait operation). In this case, if the Condition has a
     * trigger_value of TRUE, then attaching the Condition will unblock
     * the WaitSet. Adding a Condition that is already attached to the WaitSet
     * has no effect.
     *
     * @param cond      The Condition to be attached to this WaitSet.
     * @return WaitSet  The WaitSet itself so that attaching Conditions
     *                  can be chained.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    TWaitSet& attach_condition(const dds::core::cond::Condition& cond);

    /**
     * This operation detaches a Condition to the WaitSet.
     *
     * Detaches a Condition from the WaitSet. If the Condition was not
     * attached to the WaitSet, the operation will return false.
     *
     * @param cond      The Condition to detach from this WaitSet
     * @return bool     True if the Condition was found and detached, False
     *                  if the Condition was not part of the WaitSet.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    bool detach_condition(const dds::core::cond::Condition& cond);

    /**
     * This operation retrieves the list of attached Conditions.
     *
     * The resulting sequence will either be an empty sequence, meaning there were
     * no conditions attached, or will contain a list of ReadCondition,
     * QueryCondition, StatusCondition and GuardCondition.
     *
     * @return ConditionSeq
     *                  The list of attached Conditions.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    const ConditionSeq conditions() const;

    /**
     * This operation retrieves the list of attached Conditions.
     *
     * The resulting sequence will either be an empty sequence, meaning there were
     * no conditions attached, or will contain a list of ReadCondition,
     * QueryCondition, StatusCondition and GuardCondition.
     *
     * @param conds     A ConditionSeq in which to put the attached Conditions.
     * @return ConditionSeq
     *                  The list of attached Conditions.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The WaitSet was not properly created and references to dds::core::null.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    ConditionSeq& conditions(ConditionSeq& conds) const;
};

#endif /* OMG_TDDS_CORE_WAIT_SET_HPP_ */
