#ifndef OMG_DDS_T_TOPIC_MULTI_TOPIC_HPP_
#define OMG_DDS_T_TOPIC_MULTI_TOPIC_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Inc.
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

#include <dds/core/detail/conformance.hpp>
#include <dds/core/types.hpp>
#include <dds/topic/TopicDescription.hpp>

namespace dds
{
namespace topic
{

#ifdef OMG_DDS_MULTI_TOPIC_SUPPORT

/**
 * <b><i>This operation is not yet implemented. It is scheduled for a future release.</i></b>
 *
 * @see for more information: @ref DCPS_Modules_TopicDefinition "Topic Definition"
 */
template <typename T, template <typename Q> class DELEGATE>
class MultiTopic : public TTopicDescription< DELEGATE<T> >
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC_T(MultiTopic, dds::topic::TTopicDescription, T, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(MultiTopic)

public:
    template <typename FWDIterator>
    MultiTopic(const dds::domain::DomainParticipant& dp,
               const std::string& name,
               const std::string expression,
               const FWDIterator& params_begin,
               const FWDIterator& params_end);

    virtual ~MultiTopic();

public:
    const std::string expression() const;

    template <typename FWDIterator>
    void expression_parameters(const FWDIterator& params_begin,
                               const FWDIterator& params_end);

    dds::core::StringSeq void expression_parameters() const;

};

#endif  // OMG_DDS_MULTI_TOPIC_SUPPORT

}
}


#endif /* OMG_DDS_T_TOPIC_CONTENT_FILTERED_TOPIC_HPP_ */
