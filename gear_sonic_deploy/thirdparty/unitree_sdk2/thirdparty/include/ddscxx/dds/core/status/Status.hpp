#ifndef OMG_DDS_CORE_STATUS_STATUS_HPP_
#define OMG_DDS_CORE_STATUS_STATUS_HPP_

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

#include <dds/core/status/detail/Status.hpp>
#include <dds/core/status/State.hpp>

namespace dds
{
namespace core
{
namespace status
{

class DataAvailableStatus
{
    // empty
};


class DataOnReadersStatus
{
    // empty
};

// This trait is used to get the state associated with each status
template <typename STATUS>
StatusMask get_status();

typedef ::dds::core::status::detail::InconsistentTopicStatus
InconsistentTopicStatus;

typedef ::dds::core::status::detail::LivelinessChangedStatus
LivelinessChangedStatus;

typedef ::dds::core::status::detail::LivelinessLostStatus
LivelinessLostStatus;

typedef ::dds::core::status::detail::OfferedDeadlineMissedStatus
OfferedDeadlineMissedStatus;

typedef ::dds::core::status::detail::OfferedIncompatibleQosStatus
OfferedIncompatibleQosStatus;

typedef ::dds::core::status::detail::PublicationMatchedStatus
PublicationMatchedStatus;

class SampleRejectedState;

typedef ::dds::core::status::detail::SampleRejectedStatus
SampleRejectedStatus;

typedef ::dds::core::status::detail::RequestedDeadlineMissedStatus
RequestedDeadlineMissedStatus;

typedef ::dds::core::status::detail::RequestedIncompatibleQosStatus
RequestedIncompatibleQosStatus;

typedef ::dds::core::status::detail::SampleLostStatus
SampleLostStatus;

class StatusMask;

typedef ::dds::core::status::detail::SubscriptionMatchedStatus
SubscriptionMatchedStatus;
}
}
}

#endif /* OMG_DDS_CORE_STATUS_STATUS_HPP_ */
