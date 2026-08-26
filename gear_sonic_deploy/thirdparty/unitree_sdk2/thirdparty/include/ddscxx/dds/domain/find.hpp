#ifndef OMG_DDS_DOMAIN_FIND_HPP_
#define OMG_DDS_DOMAIN_FIND_HPP_

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

#include <dds/domain/DomainParticipant.hpp>

namespace dds
{
namespace domain
{


/**
 * This operation retrieves a previously-created DomainParticipant
 * belonging to the specified domain_id. If no such DomainParticipant
 * exists, the operation will return a dds::core::null DomainParticipant.
 *
 * @param id the domain id
 */
OMG_DDS_API
DomainParticipant find(uint32_t id);

}
}

#endif /* OMG_DDS_DOMAIN_FIND_HPP_ */
