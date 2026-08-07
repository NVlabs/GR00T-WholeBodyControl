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

#ifndef CYCLONEDDS_DDS_TOPIC_DETAIL_DISCOVER_HPP_
#define CYCLONEDDS_DDS_TOPIC_DETAIL_DISCOVER_HPP_

#include <dds/topic/AnyTopic.hpp>
#include <dds/topic/Topic.hpp>
#include <org/eclipse/cyclonedds/topic/discovery.hpp>

#include <string>


namespace dds
{
namespace topic
{

template <typename TOPIC>
TOPIC
discover(
    const dds::domain::DomainParticipant& dp,
    const std::string& topic_name,
    const dds::core::Duration& timeout)
{
    TOPIC t = org::eclipse::cyclonedds::topic::lookup_topic<TOPIC, typename TOPIC::DELEGATE_T>::discover(dp, topic_name, timeout);

    return t;
}


template <typename ANYTOPIC, typename FwdIterator>
uint32_t
discover(
    const dds::domain::DomainParticipant& dp,
    FwdIterator begin,
    uint32_t max_size)
{
    std::vector<ANYTOPIC> list;

    org::eclipse::cyclonedds::topic::lookup_topic<ANYTOPIC, typename ANYTOPIC::DELEGATE_T>::discover(dp, list, max_size);

    FwdIterator fit = begin;
    for (typename std::vector<ANYTOPIC>::const_iterator it = list.begin(); it != list.end(); ++it) {
       *fit++ = *it;
    }

    return static_cast<uint32_t>(list.size());
}

template <typename ANYTOPIC, typename BinIterator>
uint32_t
discover_all(
    const dds::domain::DomainParticipant& dp,
    BinIterator begin)
{
    std::vector<ANYTOPIC> list;

    org::eclipse::cyclonedds::topic::lookup_topic<ANYTOPIC, typename ANYTOPIC::DELEGATE_T>::discover(dp, list, static_cast<uint32_t>(dds::core::LENGTH_UNLIMITED));

    BinIterator bit = begin;
    for (typename std::vector<ANYTOPIC>::const_iterator it = list.begin(); it != list.end(); ++it) {
       *bit++ = *it;
    }

    return list.size();
}


template <typename FwdIterator>
void
ignore(
    const dds::domain::DomainParticipant& dp,
    FwdIterator begin, FwdIterator end)
{
    (void)dp;
    (void)begin;
    (void)end;
    ISOCPP_THROW_EXCEPTION(ISOCPP_UNSUPPORTED_ERROR, "Function not currently supported");
}

}
}


#endif /* CYCLONEDDS_DDS_TOPIC_DETAIL_DISCOVER_HPP_ */
