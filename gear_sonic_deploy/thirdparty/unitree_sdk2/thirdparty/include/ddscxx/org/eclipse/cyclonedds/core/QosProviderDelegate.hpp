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


/**
 * @file
 */

#ifndef CYCLONEDDS_CORE_QOSPROVIDERDELEGATE_HPP_
#define CYCLONEDDS_CORE_QOSPROVIDERDELEGATE_HPP_

#include <dds/domain/qos/DomainParticipantQos.hpp>
#include <dds/topic/qos/TopicQos.hpp>
#include <dds/sub/qos/SubscriberQos.hpp>
#include <dds/sub/qos/DataReaderQos.hpp>
#include <dds/pub/qos/PublisherQos.hpp>
#include <dds/pub/qos/DataWriterQos.hpp>

#if 0
C_CLASS(cmn_qosProvider);
C_CLASS(cmn_qosProviderInputAttr);
#endif

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{
class QosProviderDelegate;
}
}
}
}

class OMG_DDS_API org::eclipse::cyclonedds::core::QosProviderDelegate
{
public:
    QosProviderDelegate(const std::string& uri, const std::string& id = "");

    ~QosProviderDelegate();

    dds::domain::qos::DomainParticipantQos
    participant_qos(const char* id);

    dds::topic::qos::TopicQos
    topic_qos(const char* id);

    dds::sub::qos::SubscriberQos
    subscriber_qos(const char* id);

    dds::sub::qos::DataReaderQos
    datareader_qos(const char* id);

    dds::pub::qos::PublisherQos
    publisher_qos(const char* id);

    dds::pub::qos::DataWriterQos
    datawriter_qos(const char* id);

private:
    template <typename FROM, typename TO>
    static void named_qos__copyOut(void *from, void *to);
#if 0
    cmn_qosProvider qosProvider;
    static const C_STRUCT(cmn_qosProviderInputAttr) qosProviderAttr;
#endif
};

#endif /* CYCLONEDDS_CORE_QOSPROVIDERDELEGATE_HPP_ */
