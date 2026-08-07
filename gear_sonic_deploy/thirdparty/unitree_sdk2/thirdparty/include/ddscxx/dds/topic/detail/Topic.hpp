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
#ifndef CYCLONEDDS_DDS_TOPIC_DETAIL_TOPIC_HPP_
#define CYCLONEDDS_DDS_TOPIC_DETAIL_TOPIC_HPP_

/**
 * @file
 */

// Implementation

#include <dds/core/ref_traits.hpp>
#include <dds/topic/TopicTraits.hpp>
#include <dds/core/status/Status.hpp>
#include <dds/core/status/State.hpp>
#include <dds/domain/DomainParticipant.hpp>
#include <dds/topic/qos/TopicQos.hpp>
#include <dds/topic/AnyTopic.hpp>

#include <org/eclipse/cyclonedds/core/config.hpp>
#include <org/eclipse/cyclonedds/topic/TopicTraits.hpp>
#include <org/eclipse/cyclonedds/topic/AnyTopicDelegate.hpp>
#include <org/eclipse/cyclonedds/topic/TopicDescriptionDelegate.hpp>

#include <dds/dds.h>

namespace dds
{
namespace topic
{
template <typename T>
class TopicListener;

}
}

namespace dds
{
namespace topic
{
namespace detail
{
template <typename T>
class Topic;
}
}
}



/***************************************************************************
 *
 * dds/topic/detail/Topic<> DELEGATE declaration.
 * Implementation can be found in dds/topic/detail/TTopicImpl.hpp
 *
 ***************************************************************************/
template <typename T>
class dds::topic::detail::Topic : public org::eclipse::cyclonedds::topic::AnyTopicDelegate
{
public:
    typedef typename ::dds::core::smart_ptr_traits< Topic<T> >::ref_type ref_type;
    typedef typename ::dds::core::smart_ptr_traits< Topic<T> >::weak_ref_type weak_ref_type;

    Topic(const dds::domain::DomainParticipant& dp,
          const std::string& name,
          const std::string& type_name,
          const dds::topic::qos::TopicQos& qos,
          dds::topic::TopicListener<T>* listener,
          const dds::core::status::StatusMask& mask);

    Topic(const dds::domain::DomainParticipant& dp,
          const std::string& name,
          const std::string& type_name,
          const dds::topic::qos::TopicQos& qos,
          dds_entity_t ddsc_topic);

    virtual ~Topic();

    virtual void close();

    void init(ObjectDelegate::weak_ref_type weak_ref);

    dds::topic::Topic<T, dds::topic::detail::Topic> wrapper();


public:

    void listener(dds::topic::TopicListener<T>* listener,
                  const ::dds::core::status::StatusMask& mask);

    dds::topic::TopicListener<T>* listener();

    virtual void listener_notify(ObjectDelegate::ref_type source,
                                 uint32_t       triggerMask,
                                 void           *eventData,
                                 void           *listener);

    dds::topic::TTopicDescription<TopicDescriptionDelegate> clone();

    static dds::topic::Topic<T, dds::topic::detail::Topic>
    discover_topic(const dds::domain::DomainParticipant& dp,
                   const std::string& name,
                   const dds::core::Duration& timeout);

    static void
    discover_topics(const dds::domain::DomainParticipant& dp,
                    std::vector<dds::topic::Topic<T, dds::topic::detail::Topic> >& topics,
                    uint32_t max_size);

    //using ::org::eclipse::cyclonedds::topic::AnyTopicDelegate<T>::on_inconsistent_topic;

    void on_inconsistent_topic(
           dds_entity_t topic,
           org::eclipse::cyclonedds::core::InconsistentTopicStatusDelegate &s ) ; // !!!

private:
    T sample_;
};

// End of implementation

#endif /* CYCLONEDDS_DDS_TOPIC_DETAIL_TOPIC_HPP_ */
