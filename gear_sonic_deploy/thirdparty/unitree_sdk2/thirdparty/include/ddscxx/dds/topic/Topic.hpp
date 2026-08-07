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
#ifndef OMG_DDS_TOPIC_TOPIC_HPP_
#define OMG_DDS_TOPIC_TOPIC_HPP_

#include <dds/topic/detail/Topic.hpp>

namespace dds
{
namespace topic
{
template <typename T, template <typename Q> class DELEGATE = dds::topic::detail::Topic>
class Topic;
}
}

/**
 * @todo RTF Issue - moved include
 * @note This include was moved here as MSVC appears to ignore any
 * attempt to 're-declare' a template class with a default argument like:
 * dds/pub/detail/DataWriter.hpp(54): error C2976: 'dds::topic::Topic' : too few template arguments
 * dds/topic/TTopic.hpp(50) : see declaration of 'dds::topic::Topic'
 */
#include <dds/topic/TTopic.hpp>

#endif /* OMG_DDS_TOPIC_TOPIC_HPP_ */
