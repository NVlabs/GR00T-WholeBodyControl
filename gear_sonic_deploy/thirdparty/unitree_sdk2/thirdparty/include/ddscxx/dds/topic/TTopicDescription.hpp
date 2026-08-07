#ifndef OMG_DDS_T_TOPIC_TOPIC_DESCRIPTION_HPP_
#define OMG_DDS_T_TOPIC_TOPIC_DESCRIPTION_HPP_

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

#include <dds/core/TEntity.hpp>
#include <dds/topic/TopicTraits.hpp>
#include <dds/domain/DomainParticipant.hpp>

namespace dds
{
namespace topic
{
template <typename DELEGATE>
class TTopicDescription;
}
}


/**
 * @brief
 * This class is the base for Topic, ContentFilteredTopic and MultiTopic.
 *
 * The TopicDescription attribute type_name defines an unique data type that is
 * made available to the Data Distribution Service when a Topic is created with
 * that type.<br>
 * TopicDescription has also a name that allows it to be retrieved locally.
 *
 * @see for more information: @ref DCPS_Modules_TopicDefinition "Topic Definition"
 */
template <typename DELEGATE>
class dds::topic::TTopicDescription : public virtual dds::core::Reference<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TTopicDescription, dds::core::Reference, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TTopicDescription)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_EXPLICIT(TTopicDescription)

public:
    /**
     * This operation returns the name used to create the TopicDescription.
     *
     * @return the TopicDescription name
     */
    const std::string& name() const;

    /**
     * This operation returns the registered name of the data type associated
     * with the TopicDescription.
     *
     * @return the type_name
     */
    const std::string& type_name() const;

    /**
     * This operation returns the DomainParticipant associated with the
     * TopicDescription.
     *
     * Note that there is exactly one DomainParticipant associated with
     * each TopicDescription.
     *
     * @return the DomainParticipant
     */
    const dds::domain::DomainParticipant& domain_participant() const;

};


#endif /* OMG_DDS_T_TOPIC_TOPIC_DESCRIPTION_HPP_ */
