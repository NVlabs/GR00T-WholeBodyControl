#ifndef OMG_DDS_SUB_DISCOVERY_HPP_
#define OMG_DDS_SUB_DISCOVERY_HPP_

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

#include <dds/domain/DomainParticipant.hpp>
#include <dds/sub/DataReader.hpp>
#include <dds/topic/BuiltinTopic.hpp>
#include <dds/sub/detail/discovery.hpp>

namespace dds
{
namespace sub
{

/**
 * Ignore subscriptions.
 *
 * <b><i>This operation is not yet implemented. It is scheduled for a future release.</i></b>
 *
 * @param dp      the DomainParticipant for which the remote
 *                entity will be ignored
 *
 * @param handle  the InstanceHandle of the remote entity that
 *                has to be ignored
 */
void OMG_DDS_API ignore(const dds::domain::DomainParticipant& dp,
                        const dds::core::InstanceHandle& handle);

/**
 * Ignore subscriptions.
 *
 * <b><i>This operation is not yet implemented. It is scheduled for a future release.</i></b>
 *
 * @param dp      the DomainParticipant for which the remote
 *                entity will be ignored
 *
 * @param begin  the start of the range to be ignored
 *
 * @param end    the end of the range to be ignored
 */
template <typename FwdIterator>
void ignore(const dds::domain::DomainParticipant& dp, FwdIterator begin, FwdIterator end);


//==========================================================================
//== Discovery Management
/**
 * This operation retrieves the list of publications currently "associated" with the
 * DataReader. That is, subscriptions that have a matching Topic and compatible
 * QoS that the application has not indicated should be “ignored” by means of the
 * ignore_publication operation on the DomainParticipant.
 *
 * The handles returned in the dds::core::InstanceHandleSeq are the ones that are
 * used by the DDS implementation to locally identify the corresponding matched
 * DataWriter entities. You can access more detailed information about a particular
 * publication by passing its publication_handle to either the
 * dds::sub::matched_publication_data operation or to the read with instance
 * operation on the built-in reader for the “DCPSPublication” topic.
 *
 * See @ref DCPS_Builtin_Topics "Builtin Topics" for more information.
 *
 * @param dr        the DataReader
 * @return          a sequence of handles
 * @throws dds::core::Error
 *                  An internal error has occurred.
 * @throws dds::core::NullReferenceError
 *                  The entity was not properly created and references to dds::core::null.
 * @throws dds::core::AlreadyClosedError
 *                  The entity has already been closed.
 * @throws dds::core::NotEnabledError
 *                  The DataReader has not yet been enabled.
 * @throws dds::core::UnsupportedError
 *                  OpenSplice is configured not to maintain the
 *                  information about “associated” subscriptions.
 * @throws dds::core::OutOfResourcesError
 *                  The Data Distribution Service ran out of resources to
 *                  complete this operation.
 */
template <typename T>
::dds::core::InstanceHandleSeq
matched_publications(const dds::sub::DataReader<T>& dr);

/**
 * This operation retrieves the list of publications currently "associated" with the
 * DataReader. That is, subscriptions that have a matching Topic and compatible
 * QoS that the application has not indicated should be “ignored” by means of the
 * ignore_publication operation on the DomainParticipant.
 *
 * The handles returned in the dds::core::InstanceHandleSeq are the ones that are
 * used by the DDS implementation to locally identify the corresponding matched
 * DataWriter entities. You can access more detailed information about a particular
 * publication by passing its publication_handle to either the
 * dds::sub::matched_publication_data operation or to the read with instance
 * operation on the built-in reader for the “DCPSPublication” topic.
 *
 * See @ref DCPS_Builtin_Topics "Builtin Topics" for more information.
 *
 * @param dr        the DataReader
 * @param begin     an iterator indicating the beginning of a sequence of
 *                  instance handles in which to put the matched subscriptions
 * @param max_size  the maximum number of matched subscriptions to return
 * @return          the number of matched subscriptions returned
 * @throws dds::core::Error
 *                  An internal error has occurred.
 * @throws dds::core::NullReferenceError
 *                  The entity was not properly created and references to dds::core::null.
 * @throws dds::core::AlreadyClosedError
 *                  The entity has already been closed.
 * @throws dds::core::NotEnabledError
 *                  The DataReader has not yet been enabled.
 * @throws dds::core::UnsupportedError
 *                  OpenSplice is configured not to maintain the
 *                  information about “associated” subscriptions.
 * @throws dds::core::OutOfResourcesError
 *                  The Data Distribution Service ran out of resources to
 *                  complete this operation.
 */
template <typename T, typename FwdIterator>
uint32_t
matched_publications(const dds::sub::DataReader<T>& dr,
                     FwdIterator begin, uint32_t max_size);

/**
 * This operation retrieves information on the specified publication that is currently
 * “associated” with the DataReader. That is, a publication with a matching Topic
 * and compatible QoS that the application has not indicated should be “ignored” by
 * means of the ignore_publication operation on the DomainParticipant.
 * The publication_handle must correspond to a publication currently associated
 * with the DataReader, otherwise the operation will fail and throw
 * InvalidArgumentError. The operation dds::sub::matched_publications can
 * be used to find the publications that are currently matched with the DataReader.
 *
 * The operation may fail if the infrastructure does not locally maintain the
 * connectivity information. This is the case when OpenSplice is configured not to
 * maintain discovery information in the Networking Service. (See the description for
 * the NetworkingService/Discovery/enabled property in the Deployment
 * Manual for more information about this subject.) In such cases the operation will
 * throw UnsupportedError.
 *
 * See also @ref DCPS_Builtin_Topics and @ref DCPS_Builtin_Topics_PublicationData.
 *
 * @param dr        the DataReader
 * @param h         the InstanceHandle
 * @return          the PublicationBuiltinTopicData
 * @throws dds::core::Error
 *                  An internal error has occurred.
 * @throws dds::core::NullReferenceError
 *                  The entity was not properly created and references to dds::core::null.
 * @throws dds::core::AlreadyClosedError
 *                  The entity has already been closed.
 * @throws dds::core::NotEnabledError
 *                  The DataReader has not yet been enabled.
 * @throws dds::core::UnsupportedError
 *                  OpenSplice is configured not to maintain the
 *                  information about “associated” subscriptions.
 * @throws dds::core::InvalidArgumentError
 *                  Publication not associated with the DataReader.
 * @throws dds::core::OutOfResourcesError
 *                  The Data Distribution Service ran out of resources to
 *                  complete this operation.
 */
template <typename T>
const dds::topic::PublicationBuiltinTopicData
matched_publication_data(const dds::sub::DataReader<T>& dr,
                          const ::dds::core::InstanceHandle& h);

}
}
#endif /* OMG_DDS_SUB_DISCOVERY_HPP_ */
