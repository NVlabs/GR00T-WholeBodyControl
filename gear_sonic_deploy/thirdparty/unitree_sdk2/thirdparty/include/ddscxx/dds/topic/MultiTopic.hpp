#ifndef OMG_DDS_TOPIC_MULTI_TOPIC_HPP_
#define OMG_DDS_TOPIC_MULTI_TOPIC_HPP_

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

#include <dds/core/detail/conformance.hpp>
#include <dds/topic/detail/MultiTopic.hpp>
#include <dds/topic/TMultiTopic.hpp>


namespace dds
{
namespace topic
{

#ifdef OMG_DDS_MULTI_TOPIC_SUPPORT
template <typename T, template <typename Q> class DELEGATE = dds::topic::detail::MultiTopic>
class MultiTopic;
#endif  // OMG_DDS_MULTI_TOPIC_SUPPORT

}
}


#endif /* OMG_DDS_TOPIC_MULTI_TOPIC_HPP_ */
