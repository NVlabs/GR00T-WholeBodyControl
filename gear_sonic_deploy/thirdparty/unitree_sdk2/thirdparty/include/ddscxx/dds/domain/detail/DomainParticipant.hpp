#ifndef OMG_DDS_DOMAIN_DETAIL_DOMAINPARTICIPANT_HPP_
#define OMG_DDS_DOMAIN_DETAIL_DOMAINPARTICIPANT_HPP_

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

#include <dds/domain/detail/TDomainParticipantImpl.hpp>
//#include <dds/domain/TDomainParticipant.hpp>
//#include <org/eclipse/cyclonedds/domain/DomainParticipantDelegate.hpp>

namespace dds {
	namespace domain {
		namespace detail {
		typedef dds::domain::TDomainParticipant< org::eclipse::cyclonedds::domain::DomainParticipantDelegate >
		DomainParticipant;
		}
	}
}

#endif /* OMG_DDS_DOMAIN_DETAIL_DOMAINPARTICIPANT_HPP_ */
