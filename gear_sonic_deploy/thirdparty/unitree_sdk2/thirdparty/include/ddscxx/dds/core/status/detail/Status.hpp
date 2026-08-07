#ifndef OMG_DDS_CORE_STATUS_DETAIL_STATUS_HPP_
#define OMG_DDS_CORE_STATUS_DETAIL_STATUS_HPP_

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

#include <dds/core/status/detail/TStatusImpl.hpp>
#include <org/eclipse/cyclonedds/core/status/StatusDelegate.hpp>


namespace dds { namespace core { namespace status { namespace detail {
    typedef dds::core::status::TInconsistentTopicStatus< org::eclipse::cyclonedds::core::InconsistentTopicStatusDelegate >
    InconsistentTopicStatus;

    typedef dds::core::status::TLivelinessChangedStatus <org::eclipse::cyclonedds::core::LivelinessChangedStatusDelegate>
    LivelinessChangedStatus;

    typedef dds::core::status::TLivelinessLostStatus<org::eclipse::cyclonedds::core::LivelinessLostStatusDelegate>
    LivelinessLostStatus;

    typedef dds::core::status::TOfferedDeadlineMissedStatus<org::eclipse::cyclonedds::core::OfferedDeadlineMissedStatusDelegate>
    OfferedDeadlineMissedStatus;

    typedef dds::core::status::TOfferedIncompatibleQosStatus<org::eclipse::cyclonedds::core::OfferedIncompatibleQosStatusDelegate>
    OfferedIncompatibleQosStatus;

    typedef dds::core::status::TPublicationMatchedStatus<org::eclipse::cyclonedds::core::PublicationMatchedStatusDelegate>
    PublicationMatchedStatus;

    typedef dds::core::status::TSampleRejectedStatus< org::eclipse::cyclonedds::core::SampleRejectedStatusDelegate >
    SampleRejectedStatus;

    typedef dds::core::status::TRequestedDeadlineMissedStatus<org::eclipse::cyclonedds::core::RequestedDeadlineMissedStatusDelegate>
    RequestedDeadlineMissedStatus;

    typedef dds::core::status::TRequestedIncompatibleQosStatus<org::eclipse::cyclonedds::core::RequestedIncompatibleQosStatusDelegate>
    RequestedIncompatibleQosStatus;

    typedef dds::core::status::TSampleLostStatus<org::eclipse::cyclonedds::core::SampleLostStatusDelegate>
    SampleLostStatus;

    typedef dds::core::status::TSubscriptionMatchedStatus<org::eclipse::cyclonedds::core::SubscriptionMatchedStatusDelegate>
    SubscriptionMatchedStatus;
} } } } // namespace dds::core::status::detail


#endif /* OMG_DDS_CORE_STATUS_DETAIL_STATUS_HPP_ */
