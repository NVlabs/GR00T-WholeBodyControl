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

#ifndef CYCLONEDDS_TOPIC_DISCOVER_HPP_
#define CYCLONEDDS_TOPIC_DISCOVER_HPP_

#include <dds/domain/DomainParticipant.hpp>
#include <dds/topic/TopicDescription.hpp>
#include <dds/topic/AnyTopic.hpp>
#include <dds/topic/Topic.hpp>
#include <dds/topic/ContentFilteredTopic.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace topic
{


template <typename T, typename DELEGATE>
struct typed_lookup_topic {
    template <typename TOPIC>
    static inline TOPIC discover(
            const dds::domain::DomainParticipant& dp,
            const std::string& topic_name,
            const dds::core::Duration& timeout);

    template <typename TOPIC>
    static inline void discover(
            const dds::domain::DomainParticipant& dp,
            std::vector<TOPIC>& list,
            uint32_t max_size);
};


template <typename T>
struct typed_lookup_topic<T, dds::topic::detail::ContentFilteredTopic<T> > {
    static inline dds::topic::ContentFilteredTopic<T> discover(
            const dds::domain::DomainParticipant& dp,
            const std::string& topic_name,
            const dds::core::Duration& timeout)
    {
        (void)dp;
        (void)topic_name;
        (void)timeout;
        ISOCPP_THROW_EXCEPTION(ISOCPP_UNSUPPORTED_ERROR, "Function not currently supported");
    }

    static inline void discover(
            const dds::domain::DomainParticipant& dp,
            std::vector<dds::topic::ContentFilteredTopic<T> >& list,
            uint32_t max_size)
    {
        (void)dp;
        (void)list;
        (void)max_size;
        ISOCPP_THROW_EXCEPTION(ISOCPP_UNSUPPORTED_ERROR, "Function not currently supported");
    }
};

template <typename T>
struct typed_lookup_topic<T, dds::topic::detail::Topic<T> > {
    static inline dds::topic::Topic<T> discover(
            const dds::domain::DomainParticipant& dp,
            const std::string& topic_name,
            const dds::core::Duration& timeout)
    {
        return dds::topic::detail::Topic<T>::discover_topic(dp, topic_name, timeout);
    }

    static inline void discover(
             const dds::domain::DomainParticipant& dp,
             std::vector<dds::topic::Topic<T> >& list,
             uint32_t max_size)
     {
         dds::topic::detail::Topic<T>::discover_topics(dp, list, max_size);
     }
};



template <typename TOPIC, typename DELEGATE>
struct lookup_topic {
    static inline TOPIC discover(
            const dds::domain::DomainParticipant& dp,
            const std::string& topic_name,
            const dds::core::Duration& timeout)
    {
        return org::eclipse::cyclonedds::topic::typed_lookup_topic<typename TOPIC::DataType,
                                                          typename TOPIC::DELEGATE_T>::discover(dp, topic_name, timeout);
    }

    static inline void discover(
            const dds::domain::DomainParticipant& dp,
            std::vector<TOPIC>& list,
            uint32_t max_size)
    {
        org::eclipse::cyclonedds::topic::typed_lookup_topic<typename TOPIC::DataType,
                                                   typename TOPIC::DELEGATE_T>::discover(dp, list, max_size);
    }
};

template <>
struct lookup_topic<dds::topic::TopicDescription, org::eclipse::cyclonedds::topic::TopicDescriptionDelegate> {
    static inline dds::topic::TopicDescription discover(
            const dds::domain::DomainParticipant& dp,
            const std::string& topic_name,
            const dds::core::Duration& timeout)
    {
        return org::eclipse::cyclonedds::topic::AnyTopicDelegate::discover_topic(dp, topic_name, timeout);
    }

    static inline void discover(
            const dds::domain::DomainParticipant& dp,
            std::vector<dds::topic::TopicDescription>& list,
            uint32_t max_size)
    {
        std::vector<dds::topic::AnyTopic> anyTopics;
        org::eclipse::cyclonedds::topic::AnyTopicDelegate::discover_topics(dp, anyTopics, max_size);
        for (std::vector<dds::topic::AnyTopic>::iterator it = anyTopics.begin(); it != anyTopics.end(); ++it) {
            list.push_back(*it);
        }
    }
};

template <>
struct lookup_topic<dds::topic::AnyTopic, org::eclipse::cyclonedds::topic::AnyTopicDelegate> {
    static inline dds::topic::AnyTopic discover(
            const dds::domain::DomainParticipant& dp,
            const std::string& topic_name,
            const dds::core::Duration& timeout)
    {
        return org::eclipse::cyclonedds::topic::AnyTopicDelegate::discover_topic(dp, topic_name, timeout);
    }

    static inline void discover(
            const dds::domain::DomainParticipant& dp,
            std::vector<dds::topic::AnyTopic>& list,
            uint32_t max_size)
    {
        org::eclipse::cyclonedds::topic::AnyTopicDelegate::discover_topics(dp, list, max_size);
    }
};



}
}
}
}

#endif /* CYCLONEDDS_TOPIC_DISCOVER_HPP_ */
