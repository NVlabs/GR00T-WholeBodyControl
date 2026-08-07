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

#ifndef CYCLONEDDS_SUB_BUILTIN_SUBSCRIBER_DELEGATE_HPP_
#define CYCLONEDDS_SUB_BUILTIN_SUBSCRIBER_DELEGATE_HPP_

#include <dds/core/types.hpp>
#include <dds/domain/DomainParticipant.hpp>

#include <org/eclipse/cyclonedds/ForwardDeclarations.hpp>
#include <org/eclipse/cyclonedds/sub/SubscriberDelegate.hpp>
#include <org/eclipse/cyclonedds/sub/AnyDataReaderDelegate.hpp>
#include <org/eclipse/cyclonedds/core/Mutex.hpp>



namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{

class OMG_DDS_API BuiltinSubscriberDelegate : public org::eclipse::cyclonedds::sub::SubscriberDelegate
{
public:
    BuiltinSubscriberDelegate(
            const dds::domain::DomainParticipant& dp,
            const dds::sub::qos::SubscriberQos& qos);

    virtual ~BuiltinSubscriberDelegate() {};

    std::vector<AnyDataReaderDelegate::ref_type>
    find_datareaders(const std::string& topic_name);

public:
    static SubscriberDelegate::ref_type
    get_builtin_subscriber(const dds::domain::DomainParticipant& dp);

    static AnyDataReaderDelegate::ref_type
    get_builtin_reader(SubscriberDelegate& subscriber, const std::string& topic_name);

private:
    static org::eclipse::cyclonedds::core::Mutex builtinLock;

};

}
}
}
}

#endif /* CYCLONEDDS_SUB_BUILTIN_SUBSCRIBER_DELEGATE_HPP_ */
