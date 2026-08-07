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
#ifndef CYCLONEDDS_DDS_CORE_DETAIL_TQOSPROVIDERIMPL_HPP_
#define CYCLONEDDS_DDS_CORE_DETAIL_TQOSPROVIDERIMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/TQosProvider.hpp>

// Implementation
namespace dds
{
namespace core
{

template <typename DELEGATE>
TQosProvider<DELEGATE>::TQosProvider(const std::string& uri, const std::string& profile)
    : Reference<DELEGATE>(new DELEGATE(uri, profile)) { }

template <typename DELEGATE>
TQosProvider<DELEGATE>::TQosProvider(const std::string& uri)
    : Reference<DELEGATE>(new DELEGATE(uri)) { }

template <typename DELEGATE>
dds::domain::qos::DomainParticipantQos
TQosProvider<DELEGATE>::participant_qos()
{
    return this->delegate()->participant_qos(NULL);
}

template <typename DELEGATE>
dds::domain::qos::DomainParticipantQos
TQosProvider<DELEGATE>::participant_qos(const std::string& id)
{
    return this->delegate()->participant_qos(id.c_str());
}

template <typename DELEGATE>
dds::topic::qos::TopicQos
TQosProvider<DELEGATE>::topic_qos()
{
    return this->delegate()->topic_qos(NULL);
}

template <typename DELEGATE>
dds::topic::qos::TopicQos
TQosProvider<DELEGATE>::topic_qos(const std::string& id)
{
    return this->delegate()->topic_qos(id.c_str());
}


template <typename DELEGATE>
dds::sub::qos::SubscriberQos
TQosProvider<DELEGATE>::subscriber_qos()
{
    return this->delegate()->subscriber_qos(NULL);
}

template <typename DELEGATE>
dds::sub::qos::SubscriberQos
TQosProvider<DELEGATE>::subscriber_qos(const std::string& id)
{
    return this->delegate()->subscriber_qos(id.c_str());
}

template <typename DELEGATE>
dds::sub::qos::DataReaderQos
TQosProvider<DELEGATE>::datareader_qos()
{
    return this->delegate()->datareader_qos(NULL);
}

template <typename DELEGATE>
dds::sub::qos::DataReaderQos
TQosProvider<DELEGATE>::datareader_qos(const std::string& id)
{
    return this->delegate()->datareader_qos(id.c_str());
}

template <typename DELEGATE>
dds::pub::qos::PublisherQos
TQosProvider<DELEGATE>::publisher_qos()
{
    return this->delegate()->publisher_qos(NULL);
}

template <typename DELEGATE>
dds::pub::qos::PublisherQos
TQosProvider<DELEGATE>::publisher_qos(const std::string& id)
{
    return this->delegate()->publisher_qos(id.c_str());
}

template <typename DELEGATE>
dds::pub::qos::DataWriterQos
TQosProvider<DELEGATE>::datawriter_qos()
{
    return this->delegate()->datawriter_qos(NULL);
}

template <typename DELEGATE>
dds::pub::qos::DataWriterQos
TQosProvider<DELEGATE>::datawriter_qos(const std::string& id)
{
    return this->delegate()->datawriter_qos(id.c_str());
}
}
}
// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_DETAIL_TQOSPROVIDERIMPL_HPP_ */
