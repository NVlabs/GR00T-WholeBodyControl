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
#ifndef OMG_DDS_SUB_T_GENERATION_COUNT_HPP_
#define OMG_DDS_SUB_T_GENERATION_COUNT_HPP_

#include <dds/core/Value.hpp>

namespace dds
{
namespace sub
{
template <typename DELEGATE>
class TGenerationCount;
}
}

/**
 * @brief
 * Class to hold sample GenerationCount information and is part of dds::sub::SampleInfo.
 *
 * Generations
 * A generation is defined as: ‘the number of times an instance has become alive (with
 * instance_state==ALIVE) at the time the sample was
 * received’. Note that the generation counters are initialized to zero when a
 * DataReader first detects a never-seen-before instance.
 *
 * For each instance the middleware internally maintains two counts: the
 * disposed_generation_count and no_writers_generation_count, relative to
 * each DataReader:
 *
 * Two types of generations are distinguished: disposed_generation_count and
 * no_writers_generation_count.<br>
 * - The disposed_generation_count and no_writers_generation_count are
 *   initialized to zero when the DataReader first detects the presence of
 *   a never-seen-before instance.
 * - The disposed_generation_count is incremented each time the instance_state
 *   of the corresponding instance changes from not_alive_disposed to alive.
 * - The no_writers_generation_count is incremented each time the instance_state
 *   of the corresponding instance changes from not_alive_no_writers to alive.
 *
 * The disposed_generation_count and no_writers_generation_count associated with
 * the SampleInfo capture a snapshot of the corresponding counters at the time
 * the sample was received.
 *
 * @see for more information: @ref DCPS_Modules_Subscription_SampleInfo "SampleInfo"
 */
template <typename DELEGATE>
class dds::sub::TGenerationCount : public dds::core::Value<DELEGATE>
{
public:
    /** @cond
     * Create an empty GenerationCount.
     * This constructor is required for containers.
     * An application would normally not create a GenerationCount itself.
     * So, do not put the creation in the API documentation for clarity.
     */
    TGenerationCount();

    /**
     * Create a GenerationCount instance.
     * An application would normally not create a GenerationCount itself.
     * So, do not put the creation in the API documentation for clarity.
     */
    TGenerationCount(int32_t disposed_generation_count, int32_t no_writers_generation_count);
    /** @endcond */

public:
    /**
     * Gets the disposed_generation_count.
     *
     * The disposed_generation_count is initialized at zero and is incremented
     * each time the instance_state, of the corresponding instance, changes from
     * not_alive_disposed to alive.
     *
     * @return the disposed_generation_count
     */
    int32_t disposed() const;

    /**
     * Gets the no_writers_generation_count.
     *
     * The no_writers_generation_count is initialized at zero and is incremented
     * each time the instance_state, of the corresponding instance, changes from
     * not_alive_no_writers to alive.
     *
     * @return the no_writers_generation_count
     */
    inline int32_t no_writers() const;

};

#endif /*  OMG_DDS_SUB_T_GENERATION_COUNT_HPP_ */
