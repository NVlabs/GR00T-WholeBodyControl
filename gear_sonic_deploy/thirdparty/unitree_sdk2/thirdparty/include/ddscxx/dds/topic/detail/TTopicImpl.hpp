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
#ifndef CYCLONEDDS_DDS_TOPIC_TTOPIC_HPP_
#define CYCLONEDDS_DDS_TOPIC_TTOPIC_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/topic/TTopic.hpp>
#include "org/eclipse/cyclonedds/topic/TopicTraits.hpp"
#include "org/eclipse/cyclonedds/topic/TopicListener.hpp"

#include <dds/dds.h>
#include <functional>

#define MAX_TOPIC_NAME_LEN 1024

// Implementation

namespace dds
{
namespace topic
{


/***************************************************************************
 *
 * dds/topic/Topic<> WRAPPER implementation.
 * Declaration can be found in dds/topic/TTopic.hpp
 *
 ***************************************************************************/


template <typename T, template <typename Q> class DELEGATE>
Topic<T, DELEGATE>::Topic(const dds::domain::DomainParticipant& dp,
                          const std::string& topic_name) :
      ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(
              dp,
              topic_name,
              "",
              dp.default_topic_qos(),
              NULL,
              dds::core::status::StatusMask::none()))
{
    this->delegate()->init(this->impl_);
}

template <typename T, template <typename Q> class DELEGATE>
Topic<T, DELEGATE>::Topic(const dds::domain::DomainParticipant& dp,
                          const std::string& topic_name,
                          const std::string& type_name) :
      ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(
              dp,
              topic_name,
              type_name,
              dp.default_topic_qos(),
              NULL,
              dds::core::status::StatusMask::none())),
      ::dds::topic::TAnyTopic< DELEGATE<T> >(::dds::core::Reference< DELEGATE<T>  >::delegate())
{
    throw dds::core::UnsupportedError(std::string("Only Topics with default type_names are supported"));
    /* this->delegate()->init(this->impl_); */
}

template <typename T, template <typename Q> class DELEGATE>
Topic<T, DELEGATE>::Topic(const dds::domain::DomainParticipant& dp,
                          const std::string& topic_name,
                          const dds::topic::qos::TopicQos& qos,
                          dds::topic::TopicListener<T>* listener,
                          const dds::core::status::StatusMask& mask) :
      ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(
              dp,
              topic_name,
              "",
              qos,
              listener,
              mask)),
      ::dds::topic::TAnyTopic< DELEGATE<T> >(::dds::core::Reference< DELEGATE<T>  >::delegate())
{
    this->delegate()->init(this->impl_);
}

template <typename T, template <typename Q> class DELEGATE>
Topic<T, DELEGATE>::Topic(const dds::domain::DomainParticipant& dp,
                          const std::string& topic_name,
                          const std::string& type_name,
                          const dds::topic::qos::TopicQos& qos,
                          dds::topic::TopicListener<T>* listener,
                          const dds::core::status::StatusMask& mask) :
      ::dds::core::Reference< DELEGATE<T> >(new DELEGATE<T>(
              dp,
              topic_name,
              type_name,
              qos,
              listener,
              mask)),
      ::dds::topic::TAnyTopic< DELEGATE<T> >(::dds::core::Reference< DELEGATE<T>  >::delegate())
{
    throw dds::core::UnsupportedError(std::string("Only Topics with default type_names are supported"));
    /* this->delegate()->init(this->impl_); */
}

/** @internal  @todo Relates to OMG_DDS_X_TYPE_DYNAMIC_TYPE_SUPPORT OSPL-1736 no implementation */
template <typename T, template <typename Q> class DELEGATE>
void Topic<T, DELEGATE>::listener(Listener* listener,
                                  const ::dds::core::status::StatusMask& event_mask)
{
    this->delegate()->listener(listener, event_mask);
}

/** @internal @todo Relates to OMG_DDS_X_TYPE_DYNAMIC_TYPE_SUPPORT OSPL-1736 no implementation */
template <typename T, template <typename Q> class DELEGATE>
typename Topic<T, DELEGATE>::Listener* Topic<T, DELEGATE>::listener() const
{
    return this->delegate()->listener();
}


}
}




/***************************************************************************
 *
 * dds/topic/detail/Topic<> DELEGATE implementation.
 * Declaration can be found in dds/topic/detail/Topic.hpp
 *
 * Implementation and declaration have been separated because some circular
 * dependencies, like with TopicListener and AnyTopic.
 *
 ***************************************************************************/

#include <dds/topic/detail/Topic.hpp>
#include <dds/topic/AnyTopic.hpp>
#include <dds/topic/TopicListener.hpp>
//#include <dds/domain/DomainParticipantListener.hpp>
#include <org/eclipse/cyclonedds/core/ScopedLock.hpp>
#include <org/eclipse/cyclonedds/core/ListenerDispatcher.hpp>

#include "dds/ddsi/ddsi_sertype.h"

template <typename T>
dds::topic::detail::Topic<T>::Topic(const dds::domain::DomainParticipant& dp,
      const std::string& name,
      const std::string& type_name,
      const dds::topic::qos::TopicQos& qos,
      dds::topic::TopicListener<T>* listener,
      const dds::core::status::StatusMask& mask)
    : org::eclipse::cyclonedds::topic::TopicDescriptionDelegate(dp, name, type_name),
      org::eclipse::cyclonedds::topic::AnyTopicDelegate(qos, dp, name, type_name)
{
    // Set the correct (IDL) type_name in the TopicDescription.
    org::eclipse::cyclonedds::topic::TopicDescriptionDelegate::myTypeName = org::eclipse::cyclonedds::topic::TopicTraits<T>::getTypeName();

    // get and validate the ddsc qos
    org::eclipse::cyclonedds::topic::qos::TopicQosDelegate tQos = qos.delegate();
    tQos.check();
    dds_qos_t* ddsc_qos = tQos.ddsc_qos();
    dds_entity_t ddsc_par = dp.delegate()->get_ddsc_entity();

    ser_type_ = org::eclipse::cyclonedds::topic::TopicTraits<T>::getSerType();

    dds_entity_t ddsc_topic = dds_create_topic_sertype(
      ddsc_par, name.c_str(), &ser_type_, ddsc_qos, NULL, NULL);

    dds_delete_qos(ddsc_qos);

    if (ddsc_topic < 0) {
      ddsi_sertype_unref(ser_type_);
      ISOCPP_DDSC_RESULT_CHECK_AND_THROW(ddsc_topic, "Could not create topic.");
    }

    this->set_ddsc_entity(ddsc_topic);

    this->listener(listener, mask);

    this->AnyTopicDelegate::set_sample(&this->sample_);
}

template <typename T>
dds::topic::detail::Topic<T>::Topic(const dds::domain::DomainParticipant& dp,
      const std::string& name,
      const std::string& type_name,
      const dds::topic::qos::TopicQos& qos,
      dds_entity_t ddsc_topic)
    : org::eclipse::cyclonedds::topic::TopicDescriptionDelegate(dp, name, type_name),
      org::eclipse::cyclonedds::topic::AnyTopicDelegate(qos, dp, name, type_name)
{
    this->set_ddsc_entity(ddsc_topic);
    this->listener(NULL, dds::core::status::StatusMask::none());

    this->AnyTopicDelegate::set_sample(&this->sample_);
}


template <typename T>
dds::topic::detail::Topic<T>::~Topic/*<T>*/()
{
    if (!closed) {
        try {
            close();
        } catch (...) {

        }
    }
}

template <typename T>
void
dds::topic::detail::Topic<T>::close()
{
    org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);

    if (this->hasDependents()) {
        ISOCPP_THROW_EXCEPTION(ISOCPP_PRECONDITION_NOT_MET_ERROR, "Topic still has unclosed dependencies (e.g. Readers/Writers/ContentFilteredTopics)");
    }

    this->listener_set(NULL, dds::core::status::StatusMask::none());

    this->myParticipant.delegate()->remove_topic(*this);

    org::eclipse::cyclonedds::core::EntityDelegate::close();
}

template <typename T>
void
dds::topic::detail::Topic<T>::init(ObjectDelegate::weak_ref_type weak_ref)
{
    /* Set weak_ref before passing ourselves to other isocpp objects. */
    this->set_weak_ref(weak_ref);
    /* Add weak_ref to the map of entities */
    this->add_to_entity_map(weak_ref);
    /* Register topic at participant. */
    this->myParticipant.delegate()->add_topic(*this);

    /* Enable when needed. */
    if (this->myParticipant.delegate()->is_auto_enable()) {
        this->enable();
    }
}


template <typename T>
void
dds::topic::detail::Topic<T>::listener(TopicListener<T>* listener,
                                       const ::dds::core::status::StatusMask& mask)
{
    org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);
    this->listener_set(listener, mask);
    scopedLock.unlock();
}


template <typename T>
dds::topic::TopicListener<T>*
dds::topic::detail::Topic<T>::listener()
{
    this->check();
    return reinterpret_cast<dds::topic::TopicListener<T>*>(this->listener_get());
}

template <typename T>
dds::topic::Topic<T, dds::topic::detail::Topic>
dds::topic::detail::Topic<T>::wrapper()
{

    typename Topic::ref_type ref =
            ::std::dynamic_pointer_cast<Topic<T> >(this->get_strong_ref());
    dds::topic::Topic<T, dds::topic::detail::Topic> topic(ref);

    return topic;
}

template <typename T>
void
dds::topic::detail::Topic<T>::listener_notify(
        ObjectDelegate::ref_type source,
        uint32_t                 triggerMask,
        void                    *eventData,
        void                    *l)
{
    (void)source;
    (void)triggerMask;
    (void)eventData;
    (void)l;
}

template <typename T>
dds::topic::Topic<T, dds::topic::detail::Topic>
dds::topic::detail::Topic<T>::discover_topic(
        const dds::domain::DomainParticipant& dp,
        const std::string& name,
        const dds::core::Duration& timeout)
{
    dds::topic::Topic<T> found = dds::core::null;
    std::unique_ptr<dds_typeinfo_t, std::function<void(dds_typeinfo_t *)> >
      type_info(org::eclipse::cyclonedds::topic::TopicTraits<T>::getTypeInfo(nullptr),
                [](dds_typeinfo_t *ti) { static_cast<void>(dds_free_typeinfo(ti)); });
    dds_entity_t ddsc_topic = dp.delegate()->lookup_topic(name, type_info.get(), timeout);

    if (ddsc_topic <= 0) {
        return dds::core::null;
    }

#if 0
    /* Add type_name here when non-default ones are supported. */
    size_t slen = MAX_TOPIC_NAME_LEN;
    char *ddsc_type_name;
    ddsc_type_name = (char *)dds_alloc(slen);
    dds_get_type_name(ddsc_topic, ddsc_type_name, slen);
    std::string type_name = ddsc_type_name;
    dds_free(ddsc_type_name);
#endif

    dds_return_t ret;
    dds_qos_t* ddsc_qos = dds_create_qos();
    ret = dds_get_qos(ddsc_topic, ddsc_qos);
    dds::topic::qos::TopicQos qos;
    if (ret == DDS_RETCODE_OK) {
        qos.delegate().ddsc_qos(ddsc_qos);
    }
    dds_delete_qos(ddsc_qos);
    ISOCPP_DDSC_RESULT_CHECK_AND_THROW(ret, "Failed to get the qos from discovered topic");

    /*
     * The found topic could be of the wrong type. This will be indicated
     * with a PreconditionNotMetError when we try to create it.
     */
    try {
        found = dds::topic::Topic<T>(dp, name, qos);
    } catch (dds::core::PreconditionNotMetError&) {
        /* Ignore; just return dds::core::null */
    }

    return found;
}


template <typename T>
void
dds::topic::detail::Topic<T>::discover_topics(
        const dds::domain::DomainParticipant& dp,
        std::vector<dds::topic::Topic<T, dds::topic::detail::Topic> >& topics,
        uint32_t max_size)
{
    (void)dp;
    (void)topics;
    std::vector<dds_entity_t> ddsc_topics;
    topics.clear();
    /*
     * Unfortunately, DomainParticipantDelegate::lookup_topics() is not
     * supported yet and will throw an exception.
     */
    dp.delegate()->lookup_topics(topic_type_name<T>::value(), ddsc_topics, max_size);
}


template <typename T>
void dds::topic::detail::Topic<T>::on_inconsistent_topic(
           dds_entity_t topic,
           org::eclipse::cyclonedds::core::InconsistentTopicStatusDelegate &sd )
{
   dds::core::status::InconsistentTopicStatus s ;
   s.delegate() = sd ;
   (void)topic;
   dds::topic::Topic<T, dds::topic::detail::Topic> t = wrapper() ;

   dds::topic::TopicListener<T> *l =
      reinterpret_cast<dds::topic::TopicListener<T> *>(this->listener_get());
   if( (l != NULL) &&
       (this->get_listener_mask().to_ulong() &
             dds::core::status::StatusMask::inconsistent_topic().to_ulong()) )
   {
      l->on_inconsistent_topic( t, s ) ;
   }
}

// End of implementation

#endif /* CYCLONEDDS_DDS_TOPIC_TTOPIC_HPP_ */
