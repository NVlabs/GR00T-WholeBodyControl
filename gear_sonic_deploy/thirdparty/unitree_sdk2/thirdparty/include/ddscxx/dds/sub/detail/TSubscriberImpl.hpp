/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_DDS_SUB_TSUBSCRIBER_IMPL_HPP_
#define CYCLONEDDS_DDS_SUB_TSUBSCRIBER_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/TSubscriber.hpp>

// Implementation

namespace dds
{
namespace sub
{

template <typename DELEGATE>
TSubscriber<DELEGATE>::TSubscriber(const ::dds::domain::DomainParticipant& dp)
    : ::dds::core::Reference<DELEGATE>(new DELEGATE(dp,
                                                    dp.default_subscriber_qos(),
                                                    NULL,
                                                    dds::core::status::StatusMask::none()))
{
    this->delegate()->init(this->impl_);
}

template <typename DELEGATE>
TSubscriber<DELEGATE>::TSubscriber(const ::dds::domain::DomainParticipant& dp,
                                   const dds::sub::qos::SubscriberQos& qos,
                                   dds::sub::SubscriberListener* listener,
                                   const dds::core::status::StatusMask& mask)
    : ::dds::core::Reference<DELEGATE>(new DELEGATE(dp, qos, listener, mask))
{
    this->delegate()->init(this->impl_);
}

template <typename DELEGATE>
void TSubscriber<DELEGATE>::notify_datareaders()
{
    this->delegate()->notify_datareaders();
}

template <typename DELEGATE>
void TSubscriber<DELEGATE>::listener(Listener* listener,
                                     const dds::core::status::StatusMask& event_mask)
{
    return this->delegate()->listener(listener, event_mask);
}

template <typename DELEGATE>
typename TSubscriber<DELEGATE>::Listener* TSubscriber<DELEGATE>::listener() const
{
    return this->delegate()->listener();
}


template <typename DELEGATE>
const dds::sub::qos::SubscriberQos& TSubscriber<DELEGATE>::qos() const
{
    return this->delegate()->qos();
}

template <typename DELEGATE>
void TSubscriber<DELEGATE>::qos(const dds::sub::qos::SubscriberQos& sqos)
{
    this->delegate()->qos(sqos);
}

template <typename DELEGATE>
dds::sub::qos::DataReaderQos TSubscriber<DELEGATE>::default_datareader_qos() const
{
    return this->delegate()->default_datareader_qos();
}

template <typename DELEGATE>
TSubscriber<DELEGATE>& TSubscriber<DELEGATE>::default_datareader_qos(
    const dds::sub::qos::DataReaderQos& qos)
{
    this->delegate()->default_datareader_qos(qos);
    return *this;
}

template <typename DELEGATE>
const dds::domain::DomainParticipant& TSubscriber<DELEGATE>::participant() const
{
    return this->delegate()->participant();
}

template <typename DELEGATE>
TSubscriber<DELEGATE>& TSubscriber<DELEGATE>::operator << (const dds::sub::qos::SubscriberQos& qos)
{
    this->qos(qos);
    return *this;
}

template <typename DELEGATE>
const TSubscriber<DELEGATE>& TSubscriber<DELEGATE>::operator >> (dds::sub::qos::SubscriberQos& qos) const
{
    qos = this->qos();
    return *this;
}

}
}
// End of implementation

#endif /* CYCLONEDDS_DDS_SUB_TSUBSCRIBER_IMPL_HPP_ */
