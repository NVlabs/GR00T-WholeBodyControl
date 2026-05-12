#ifndef OMG_DDS_SUB_FIND_HPP_
#define OMG_DDS_SUB_FIND_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Corp.
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

#include <string>

#include <dds/sub/detail/find.hpp>

namespace dds
{
namespace sub
{

/**
 * This operation returns the built-in Subscriber associated with the given
 * given DomainParticipant.
 *
 * Each DomainParticipant contains several built-in
 * Topic objects. The built-in Subscriber contains the corresponding DataReader
 * objects to access them. All these DataReader objects belong to a single built-in
 * Subscriber. Note that there is exactly one built-in Subscriber associated with
 * each DomainParticipant.
 *
 * See @ref DCPS_Builtin_Topics "Builtin Topics" for more information.
 *
 * @param dp the domain participant
 * @return the built-in Subscriber
 */
const dds::sub::Subscriber OMG_DDS_API
builtin_subscriber(const dds::domain::DomainParticipant& dp);

/**
 * This function retrieves a previously-created DataReader
 * belonging to the Subscriber that is attached to a Topic with a
 * matching topic_name. If no such DataReader exists, the operation
 * will return an empty container.
 *
 * The use of this
 * operation on the built-in Subscriber allows access to the
 * built-in DataReader entities for the built-in topics
 * (see @ref DCPS_Builtin_Topics "Builtin Topics").
 *
 * @param sub the Subscriber for which to find a DataReader
 * @param topic_name the topic name to find
 * @param begin a forward iterator pointing to the start of a
 *        container in which to put the DataReaders
 * @param max_size the number of DataReaders to return
 * @return the total number of elements returned. Notice that
 *        at most max_size will be copied using the provided iterator
 *
 */
template <typename READER, typename FwdIterator>
uint32_t
find(const dds::sub::Subscriber& sub,
     const std::string &topic_name,
     FwdIterator begin, uint32_t max_size);

/**
 * This function retrieves a previously-created DataReader
 * belonging to the Subscriber that is attached to a Topic with a
 * matching topic_name. If no such DataReader exists, the operation
 * will return an empty container.
 *
 * The use of this
 * operation on the built-in Subscriber allows access to the
 * built-in DataReader entities for the built-in topics
 * (see @ref DCPS_Builtin_Topics "Builtin Topics").
 *
 * @param sub the Subscriber for which to find a DataReader
 * @param topic_name the topic name to find
 * @param begin a back inserting iterator pointing to the start of a
 *        container in which to put the DataReaders
 * @return the total number of elements returned
 */
template <typename READER, typename BinIterator>
uint32_t
find(const dds::sub::Subscriber& sub,
     const std::string &topic_name,
     BinIterator begin);

/**
 * This function retrieves a previously-created DataReader
 * belonging to the Subscriber that is attached to a Topic with a
 * matching topic_name. If no such DataReader exists, the operation
 * will return an empty container.
 *
 * The use of this
 * operation on the built-in Subscriber allows access to the
 * built-in DataReader entities for the built-in topics
 * (see @ref DCPS_Builtin_Topics "Builtin Topics").
 *
 * @param sub the Subscriber for which to find a DataReader
 * @param topic_description the topic description to find
 * @param begin a forward iterator pointing to the start of a
 *        container in which to put the DataReaders
 * @param max_size the number of DataReaders to return
 * @return the total number of elements returned. Notice that
 *        at most max_size will be copied using the provided iterator
 */
template <typename READER, typename T, typename FwdIterator>
uint32_t
find(const dds::sub::Subscriber& sub,
     const dds::topic::TopicDescription& topic_description,
     FwdIterator begin, uint32_t max_size);

/**
 * This function retrieves a previously-created DataReader
 * belonging to the Subscriber that is attached to a Topic with a
 * matching topic_name. If no such DataReader exists, the operation
 * will return an empty container.
 *
 * The use of this
 * operation on the built-in Subscriber allows access to the
 * built-in DataReader entities for the built-in topics
 * (see @ref DCPS_Builtin_Topics "Builtin Topics").
 *
 * @param sub the Subscriber for which to find a DataReader
 * @param topic_description the topic description to find
 * @param begin a back inserting iterator pointing to the start of a
 *        container in which to put the DataReaders
 * @return the total number of elements returned
 */
template <typename READER, typename T, typename BinIterator>
uint32_t
find(const dds::sub::Subscriber& sub,
     const dds::topic::TopicDescription& topic_description,
     BinIterator begin);

/**
 * This operation allows the application to access the DataReader objects that contain
 * samples with the specified sample_states, view_states, and
 * instance_states.
 *
 * If the Presentation QosPolicy of the Subscriber to which the DataReader
 * belongs has the access_scope set to ‘GROUP’, this operation should only be
 * invoked inside a begin_access/end_access block. Otherwise it will throw
 * error PreconditionNotMetError.
 *
 * Depending on the setting of the dds::core::policy::Presentation QoSPolicy,
 * the returned collection of DataReader objects may be:
 * - a ‘set’ containing each DataReader at most once in no specified order,
 * - a ‘list’ containing each DataReader one or more times in a specific order.
 *
 * This difference is due to the fact that, in the second situation it is required to access
 * samples belonging to different DataReader objects in a particular order. In this case,
 * the application should process each DataReader in the same order it appears in the
 * ‘list’ and read or take exactly one sample from each DataReader. The patterns that
 * an application should use to access data is fully described in dds::core::policy::Presentation.
 *
 * @param sub the Subscriber for which to find a DataReader
 * @param data_state the data_state to find
 * @param begin a forward iterator pointing to the start of a
 *        container in which to put the DataReaders
 * @param max_size the number of DataReaders to return
 * @return the total number of elements returned. Notice that
 *        at most max_size will be copied using the provided iterator
 */
template <typename READER, typename FwdIterator>
uint32_t
find(const dds::sub::Subscriber& sub,
     const dds::sub::status::DataState& data_state,
     FwdIterator begin, uint32_t max_size);

/**
 * This operation allows the application to access the DataReader objects that contain
 * samples with the specified sample_states, view_states, and
 * instance_states.
 *
 * If the Presentation QosPolicy of the Subscriber to which the DataReader
 * belongs has the access_scope set to ‘GROUP’, this operation should only be
 * invoked inside a begin_access/end_access block. Otherwise it will throw
 * error PreconditionNotMetError.
 *
 * Depending on the setting of the dds::core::policy::Presentation QoSPolicy,
 * the returned collection of DataReader objects may be:
 * - a ‘set’ containing each DataReader at most once in no specified order,
 * - a ‘list’ containing each DataReader one or more times in a specific order.
 *
 * This difference is due to the fact that, in the second situation it is required to access
 * samples belonging to different DataReader objects in a particular order. In this case,
 * the application should process each DataReader in the same order it appears in the
 * ‘list’ and read or take exactly one sample from each DataReader. The patterns that
 * an application should use to access data is fully described in dds::core::policy::Presentation.
 *
 * @param sub the Subscriber for which to find a DataReader
 * @param rs the data_state to find
 * @param begin a back inserting iterator pointing to the start
 *        of a container in which to put the DataReaders
 * @return the total number of elements returned
 */
template <typename READER, typename BinIterator>
uint32_t
find(const dds::sub::Subscriber& sub,
     const dds::sub::status::DataState& rs,
     BinIterator begin);

}
}

#endif /* OMG_DDS_SUB_FIND_HPP_ */
