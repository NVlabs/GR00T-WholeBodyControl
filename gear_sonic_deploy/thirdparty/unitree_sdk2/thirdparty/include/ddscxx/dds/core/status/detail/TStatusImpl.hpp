/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_DDS_CORE_STATUS_TSTATUS_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_STATUS_TSTATUS_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/status/TStatus.hpp>
// Implementation

namespace dds
{
namespace core
{
namespace status
{

//TInconsistentTopicStatus

template <typename D>
TInconsistentTopicStatus<D>::TInconsistentTopicStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TInconsistentTopicStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TInconsistentTopicStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

//TSampleLostStatus

template <typename D>
TSampleLostStatus<D>::TSampleLostStatus() : dds::core::Value<D>() {}

template <typename D>
int32_t TSampleLostStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TSampleLostStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

//TSampleRejectedStatus

template <typename D>
TSampleRejectedStatus<D>::TSampleRejectedStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TSampleRejectedStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TSampleRejectedStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

template <typename D>
const dds::core::status::SampleRejectedState TSampleRejectedStatus<D>::last_reason() const
{
    return this->delegate().last_reason();
}

template <typename D>
const dds::core::InstanceHandle TSampleRejectedStatus<D>::last_instance_handle() const
{
    return this->delegate().last_instance_handle();
}
//TLivelinessLostStatus
template <typename D>
TLivelinessLostStatus<D>::TLivelinessLostStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TLivelinessLostStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TLivelinessLostStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

//TLivelinessChangedStatus

template <typename D>
TLivelinessChangedStatus<D>::TLivelinessChangedStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TLivelinessChangedStatus<D>::alive_count() const
{
    return this->delegate().alive_count();
}

template <typename D>
int32_t TLivelinessChangedStatus<D>::not_alive_count() const
{
    return this->delegate().not_alive_count();
}

template <typename D>
int32_t TLivelinessChangedStatus<D>::alive_count_change() const
{
    return this->delegate().alive_count_change();
}

template <typename D>
int32_t TLivelinessChangedStatus<D>::not_alive_count_change() const
{
    return this->delegate().not_alive_count_change();
}

template <typename D>
const dds::core::InstanceHandle TLivelinessChangedStatus<D>::last_publication_handle() const
{
    return this->delegate().last_publication_handle();
}

//TOfferedDeadlineMissedStatus

template <typename D>
TOfferedDeadlineMissedStatus<D>::TOfferedDeadlineMissedStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TOfferedDeadlineMissedStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TOfferedDeadlineMissedStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

template <typename D>
const dds::core::InstanceHandle TOfferedDeadlineMissedStatus<D>::last_instance_handle() const
{
    return this->delegate().last_instance_handle();
}

//TRequestedDeadlineMissedStatus

template <typename D>
TRequestedDeadlineMissedStatus<D>::TRequestedDeadlineMissedStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TRequestedDeadlineMissedStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TRequestedDeadlineMissedStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

template <typename D>
const dds::core::InstanceHandle TRequestedDeadlineMissedStatus<D>::last_instance_handle() const
{
    return this->delegate().last_instance_handle();
}

//TOfferedIncompatibleQosStatus

template <typename D>
TOfferedIncompatibleQosStatus<D>::TOfferedIncompatibleQosStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TOfferedIncompatibleQosStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TOfferedIncompatibleQosStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

template <typename D>
dds::core::policy::QosPolicyId TOfferedIncompatibleQosStatus<D>::last_policy_id() const
{
    return this->delegate().last_policy_id();
}

template <typename D>
const dds::core::policy::QosPolicyCountSeq TOfferedIncompatibleQosStatus<D>::policies() const
{
    return this->delegate().policies();
}

template <typename D>
const dds::core::policy::QosPolicyCountSeq& TOfferedIncompatibleQosStatus<D>::policies(dds::core::policy::QosPolicyCountSeq& dst) const
{
    return this->delegate().policies(dst);
}

//TRequestedIncompatibleQosStatus

template <typename D>
TRequestedIncompatibleQosStatus<D>::TRequestedIncompatibleQosStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TRequestedIncompatibleQosStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TRequestedIncompatibleQosStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

template <typename D>
dds::core::policy::QosPolicyId TRequestedIncompatibleQosStatus<D>::last_policy_id() const
{
    return this->delegate().last_policy_id();
}

template <typename D>
const dds::core::policy::QosPolicyCountSeq TRequestedIncompatibleQosStatus<D>::policies() const
{
    return this->delegate().policies();
}

template <typename D>
const dds::core::policy::QosPolicyCountSeq& TRequestedIncompatibleQosStatus<D>::policies(dds::core::policy::QosPolicyCountSeq& dst) const
{
    return this->delegate().policies(dst);
}

//TPublicationMatchedStatus

template <typename D>
TPublicationMatchedStatus<D>::TPublicationMatchedStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TPublicationMatchedStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TPublicationMatchedStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

template <typename D>
int32_t TPublicationMatchedStatus<D>::current_count() const
{
    return this->delegate().current_count();
}

template <typename D>
int32_t TPublicationMatchedStatus<D>::current_count_change() const
{
    return this->delegate().current_count_change();
}

template <typename D>
const dds::core::InstanceHandle TPublicationMatchedStatus<D>::last_subscription_handle() const
{
    return this->delegate().last_subscription_handle();
}

//TSubscriptionMatchedStatus

template <typename D>
TSubscriptionMatchedStatus<D>::TSubscriptionMatchedStatus() : dds::core::Value<D>() { }

template <typename D>
int32_t TSubscriptionMatchedStatus<D>::total_count() const
{
    return this->delegate().total_count();
}

template <typename D>
int32_t TSubscriptionMatchedStatus<D>::total_count_change() const
{
    return this->delegate().total_count_change();
}

template <typename D>
int32_t TSubscriptionMatchedStatus<D>::current_count() const
{
    return this->delegate().current_count();
}

template <typename D>
int32_t TSubscriptionMatchedStatus<D>::current_count_change() const
{
    return this->delegate().current_count_change();
}

template <typename D>
const dds::core::InstanceHandle TSubscriptionMatchedStatus<D>::last_publication_handle() const
{
    return this->delegate().last_publication_handle();
}

}
}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_STATUS_TSTATUS_IMPL_HPP_ */
