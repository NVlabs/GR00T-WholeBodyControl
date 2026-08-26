#ifndef OMG_DDS_SUB_TCOND_READ_CONDITION_HPP_
#define OMG_DDS_SUB_TCOND_READ_CONDITION_HPP_


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

#include <dds/core/cond/TCondition.hpp>
#include <dds/sub/DataReader.hpp>
#include <dds/sub/AnyDataReader.hpp>

namespace dds
{
namespace sub
{
namespace cond
{
template <typename DELEGATE>
class TReadCondition;
}
}
}

/**
 * @brief
 * ReadCondition objects are conditions specifically dedicated
 * to read operations and attached to one DataReader.
 *
 * ReadCondition objects allow an application to specify the data samples
 * it is interested in (by specifying the desired sample states, view states,
 * and instance states). (See the parameter definitions for DataReader's
 * read/take operations.) This allows the middle-ware to enable the condition
 * only when suitable information is available. They are to be used in
 * conjunction with a WaitSet as normal conditions. More than one
 * ReadCondition may be attached to the same DataReader.
 *
 * See the @ref anchor_dds_core_cond_waitset_examples "WaitSet examples" for some examples.<br>
 * Although the WaitSet examples use the StatusCondition, the basic usage of this Condition
 * with a WaitSet is the same.
 *
 * @see dds::core::cond::Condition
 * @see for more information: @ref DCPS_Modules_Infrastructure_Status  "Status concept"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "WaitSet concept"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "Subscription concept"
 * @see for more information: @ref anchor_dds_core_cond_waitset_examples "WaitSet examples"
 */
template <typename DELEGATE>
class dds::sub::cond::TReadCondition : public dds::core::cond::TCondition<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TReadCondition, dds::core::cond::TCondition, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TReadCondition)

public:
    /**
     * Create a dds::sub::cond::ReadCondition associated with a DataReader.
     *
     * The ReadCondition can then be added to a dds::core::cond::WaitSet so that the
     * application can wait for specific status changes that affect the Entity.
     *
     * @anchor anchor_dds_sub_cond_readcondition_state_mask
     * State Masks.<br>
     * The result of the ReadCondition depends on the selection of samples determined by
     * three masks:
     * - DataState::sample_state() is the mask, which selects only those samples with the desired
     *   sample states SampleState::read(), SampleState::not_read() or SampleState::any().
     * - DataState::view_state() is the mask, which selects only those samples with the desired
     *   view states ViewState::new_view(), ViewState::not_new_view() or ViewState::any().
     * - DataState::instance_state() is the mask, which selects only those samples with the desired
     *   view states InstanceState::alive(), InstanceState::not_alive_disposed(),
     *   InstanceState::not_alive_no_writers(), InstanceState::not_alive_mask() or InstanceState::any().
     * See also @ref DCPS_Modules_Infrastructure_Status  "Status Concept".
     *
     * @param  dr       The DataReader to associate with the ReadCondition.
     * @param  status   A mask, which selects only those samples with the desired
     *                  sample/view/instance states.
     * @throw  dds::core::Exception
     */
    TReadCondition(const dds::sub::AnyDataReader& dr, const dds::sub::status::DataState& status);

    /**
     * Create a dds::sub::cond::ReadCondition associated with a DataReader.
     *
     * The ReadCondition can then be added to a dds::core::cond::WaitSet so that the
     * application can wait for specific status changes that affect the Entity.
     *
     * The supplied functor will be called when this ReadCondition is triggered
     * and either the inherited dds::core::cond::Condition::dispatch() is called or the
     * dds::core::cond::WaitSet::dispatch() on the WaitSet to which this ReadCondition is
     * attached to.
     *
     * See @ref anchor_dds_sub_cond_readcondition_state_mask "State mask info".
     *
     * @param  dr       The DataReader to associate with the ReadCondition.
     * @param  status   A mask, which selects only those samples with the desired
     *                  sample/view/instance states.
     * @param  functor The functor to be called when the ReadCondition triggers.
     * @throw  dds::core::Exception
     */
    template <typename FUN>
    TReadCondition(const dds::sub::AnyDataReader& dr, const dds::sub::status::DataState& status, FUN functor);

public:
    /**
     * This operation returns the set of data-states that are taken into
     * account to determine the trigger_value of the ReadCondition.
     *
     * These are data-states specified when the ReadCondition was created.
     *
     * @return dds::sub::status::DataState The data state.
     * @throw  dds::core::Exception
     */
    const dds::sub::status::DataState state_filter() const;

    /**
     * This operation returns the DataReader associated with the ReadCondition.
     *
     * Note that there is exactly one DataReader associated with each ReadCondition.
     *
     * @return dds::sub::AnyDataReader The associated DataReader
     * @throw  dds::core::Exception
     */
    const dds::sub::AnyDataReader& data_reader() const;

};


#endif /* OMG_DDS_SUB_TCOND_READ_CONDITION_HPP_ */
