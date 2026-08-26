#ifndef OMG_DDS_TOPIC_DISCOVER_HPP_
#define OMG_DDS_TOPIC_DISCOVER_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Inc.
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

#include <dds/domain/DomainParticipant.hpp>

namespace dds
{
namespace topic
{

/**
 * This operation gives access to an specific existing (or ready to exist) enabled
 * Topic, ContentFilteredTopic, MultiTopic, AnyTopic or TopicDescription based
 * on its topic_name.
 *
 * Topics that the application has indicated should be 'ignored' (by means of the
 * dds::topic::ignore operation) will not appear in this list (note: the ignore
 * feature is not yet supported).
 *
 * If a Topic of the same topic_name already exists, it gives access to this Topic.
 * Otherwise it waits (blocks the caller) until another mechanism creates it. This other
 * mechanism can be another thread, a configuration tool, or some other Data
 * Distribution Service utility. If after the specified timeout the Topic can still not be
 * found, the caller gets unblocked and the returned Topic will be dds::core::null.
 *
 * A Topic that is obtained by means of find_topic in a specific
 * DomainParticipant can only be used to create DataReaders and
 * DataWriters in that DomainParticipant.
 *
 * This operation usually results in network look-ups.
 *
 * For finding only locally created Topics, look
 * @link find(const dds::domain::DomainParticipant& dp, const std::string& topic_name)
 * here.@endlink
 *
 * @param dp the DomainParticipant
 * @param name the topic name to discover
 * @param timeout the time out
 * @throws dds::core::Error
 *                  An internal error has occurred.
 * @throws dds::core::NullReferenceError
 *                  The DomainParticipant was not properly created and references to dds::core::null.
 * @throws dds::core::AlreadyClosedError
 *                  The DomainParticipant has already been closed.
 * @throws dds::core::NotEnabledError
 *                  The DomainParticipant has not yet been enabled.
 * @throws dds::core::OutOfResourcesError
 *                  The Data Distribution Service ran out of resources to
 *                  complete this operation.
 * @throws dds::core::TimeoutError
 *                  No Topics of the given name found within the timeout.
 */
template <typename TOPIC>
TOPIC discover(const dds::domain::DomainParticipant& dp,
               const std::string& name,
               const dds::core::Duration& timeout = dds::core::Duration::infinite());

/**
 * This operation retrieves a list of Topics that have been discovered in the domain.
 *
 * If the max_size of the given list is large enough, all discovered Topics will be
 * present in that list. Otherwise, a random sub-set of max_size elements is returned.
 *
 * Topics that the application has indicated should be 'ignored' (by means of the
 * dds::topic::ignore operation) will not appear in this list (note: the ignore
 * feature is not yet supported).
 *
 * Because Topics of various different kinds can be retrieved, the list contains
 * AnyTopics.
 *
 * This operation usually results in network look-ups.
 *
 * @param dp the DomainParticipant
 * @param begin a forward iterator pointing to the beginning of a container
 *        in which to insert the topics
 * @param max_size the maximum number of topics to return
 */
template <typename ANYTOPIC, typename FwdIterator>
uint32_t discover(const dds::domain::DomainParticipant& dp, FwdIterator begin, uint32_t max_size);

/**
 * This operation retrieves a list of all Topics that have been discovered in the domain.
 *
 * Topics that the application has indicated should be 'ignored' (by means of the
 * dds::topic::ignore operation) will not appear in this list (note: the ignore
 * feature is not yet supported).
 *
 * Because Topics of various different kinds can be retrieved, the list contains
 * AnyTopics.
 *
 * This operation usually results in network look-ups.
 *
 * @param dp the DomainParticipant
 * @param begin a back inserting iterator pointing to the beginning of a container
 *        in which to insert the topics
 */
template <typename ANYTOPIC, typename BinIterator>
uint32_t discover_all(const dds::domain::DomainParticipant& dp, BinIterator begin);

/**
 * This operation allows an application to instruct the Service to locally ignore
 * a remote domain participant. From that point onwards the Service will locally
 * behave as if the remote participant did not exist. This means it will ignore any
 * Topic, publication, or subscription that originates on that domain participant.
 *
 * <b><i>This operation is not yet implemented. It is scheduled for a future release.</i></b>
 *
 * @param dp the DomainParticipant
 * @param handle the handle of the DomainParticipant to ignore
 */
void OMG_DDS_API ignore(const dds::domain::DomainParticipant& dp, const dds::core::InstanceHandle& handle);

/**
 * This operation allows an application to instruct the Service to locally ignore
 * a remote domain participant. From that point onwards the Service will locally
 * behave as if the remote participant did not exist. This means it will ignore any
 * Topic, publication, or subscription that originates on that domain participant.
 *
 * <b><i>This operation is not yet implemented. It is scheduled for a future release.</i></b>
 *
 * @param dp the DomainParticipant
 * @param begin a forward iterator pointing to the beginning of a sequence of
 *        InstanceHandles to ignore
 * @param end a forward iterator pointing to the end of a sequence of
 *        InstanceHandles to ignore
 */
template <typename FwdIterator>
void ignore(const dds::domain::DomainParticipant& dp, FwdIterator begin, FwdIterator end);

}
}

#include <dds/topic/detail/discovery.hpp>

#endif /* OMG_DDS_TOPIC_DISCOVER_HPP_ */
