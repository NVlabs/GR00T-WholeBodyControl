#ifndef OMG_DDS_CORE_QOS_TPROVIDER_HPP_
#define OMG_DDS_CORE_QOS_TPROVIDER_HPP_


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

#include <dds/core/Reference.hpp>

#include <dds/domain/qos/DomainParticipantQos.hpp>

#include <dds/topic/qos/TopicQos.hpp>

#include <dds/sub/qos/SubscriberQos.hpp>
#include <dds/sub/qos/DataReaderQos.hpp>

#include <dds/pub/qos/PublisherQos.hpp>
#include <dds/pub/qos/DataWriterQos.hpp>

namespace dds
{
namespace core
{
template <typename DELEGATE>
class TQosProvider;
}
}

/**
 * @brief
 * The QosProvider API allows users to specify the QoS settings of their DCPS
 * entities outside of application code in XML.
 *
 * The QosProvider is delivered as part of the DCPS API of Vortex OpenSplice.
 *
 * @see for more information: @ref DCPS_QoS_Provider "QoS Provider extensive information."
 */
template <typename DELEGATE>
class dds::core::TQosProvider : public dds::core::Reference<DELEGATE>
{
public:
    /**
     * Constructs a new QosProvider based on the provided uri and profile.
     *
     * A QosProvider instance that is instantiated with all profiles and/or QoS’s loaded
     * from the location specified by the provided uri.
     *
     * Initialization of the QosProvider will fail under the following conditions:<br>
     * - No uri is provided.
     * - The resource pointed to by uri cannot be found.
     * - The content of the resource pointed to by uri is malformed (e.g., malformed XML).
     * When initialization fails (for example, due to a parse error or when the resource
     * identified by uri cannot be found), then PreconditionNotMetError will be thrown.
     *
     * Look @ref DCPS_QoS_Provider "here" for more information.
     *
     * @param uri       A Uniform Resource Identifier (URI) that points to the location
     *                  where the QoS profile needs to be loaded from. Currently only URI’s with a
     *                  ‘file’ scheme that point to an XML file are supported. If profiles and/or QoS
     *                  settings are not uniquely identifiable by name within the resource pointed to by
     *                  uri, a random one of them will be stored.
     * @param profile   The name of the QoS profile within the xml file that serves as the default QoS
     *                  profile for the get qos operations.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  When multiple thread try to invoke the function concurrently.
     */
    explicit TQosProvider(const std::string& uri, const std::string& profile);

    /**
     * Constructs a new QosProvider based on the provided uri.
     *
     * A QosProvider instance that is instantiated with all profiles and/or QoS’s loaded
     * from the location specified by the provided uri.
     *
     * Initialization of the QosProvider will fail under the following conditions:<br>
     * - No uri is provided.
     * - The resource pointed to by uri cannot be found.
     * - The content of the resource pointed to by uri is malformed (e.g., malformed XML).
     * When initialization fails (for example, due to a parse error or when the resource
     * identified by uri cannot be found), then PreconditionNotMetError will be thrown.
     *
     * Look @ref DCPS_QoS_Provider "here" for more information.
     *
     * @param uri       A Uniform Resource Identifier (URI) that points to the location
     *                  where the QoS profile needs to be loaded from. Currently only URI’s with a
     *                  ‘file’ scheme that point to an XML file are supported. If profiles and/or QoS
     *                  settings are not uniquely identifiable by name within the resource pointed to by
     *                  uri, a random one of them will be stored.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  When multiple thread try to invoke the function concurrently.
     */
    explicit TQosProvider(const std::string& uri);

    /**
     * Resolves the DomainParticipantQos from the uri this QosProvider is associated with.
     *
     * @return DomainParticipantQos from the given URI (and profile)
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no DomainParticipantQos can be found within the uri associated
     *                  with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::domain::qos::DomainParticipantQos participant_qos();

    /**
     * Resolves the DomainParticipantQos identified by the id from the uri this
     * QosProvider is associated with.
     *
     * @param id        The fully-qualified name that identifies a QoS within the uri
     *                  associated with the QosProvider or a name that identifies a QoS within the
     *                  uri associated with the QosProvider instance relative to its default QoS
     *                  profile. Id’s starting with ‘::’ are interpreted as fully-qualified names and all
     *                  others are interpreted as names relative to the default QoS profile of the
     *                  QosProvider instance. When id is NULL it is interpreted as a non-named QoS
     *                  within the default QoS profile associated with the QosProvider.
     * @return DomainParticipantQos from the given URI (and profile) using the id
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no DomainParticipantQos that matches the provided id can be
     *                  found within the uri associated with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::domain::qos::DomainParticipantQos participant_qos(const std::string& id);

    /**
     * Resolves the TopicQos from the uri this QosProvider is associated with.
     *
     * @return TopicQos from the given URI (and profile)
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no TopicQos can be found within the uri associated
     *                  with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::topic::qos::TopicQos topic_qos();

    /**
     * Resolves the TopicQos identified by the id from the uri this
     * QosProvider is associated with.
     *
     * @param id        The fully-qualified name that identifies a QoS within the uri
     *                  associated with the QosProvider or a name that identifies a QoS within the
     *                  uri associated with the QosProvider instance relative to its default QoS
     *                  profile. Id’s starting with ‘::’ are interpreted as fully-qualified names and all
     *                  others are interpreted as names relative to the default QoS profile of the
     *                  QosProvider instance. When id is NULL it is interpreted as a non-named QoS
     *                  within the default QoS profile associated with the QosProvider.
     * @return TopicQos from the given URI (and profile) using the id
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no TopicQos that matches the provided id can be
     *                  found within the uri associated with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::topic::qos::TopicQos topic_qos(const std::string& id);

    /**
     * Resolves the SubscriberQos from the uri this QosProvider is associated with.
     *
     * @return SubscriberQos from the given URI (and profile)
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no SubscriberQos can be found within the uri associated
     *                  with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::sub::qos::SubscriberQos subscriber_qos();

    /**
     * Resolves the SubscriberQos identified by the id from the uri this
     * QosProvider is associated with.
     *
     * @param id        The fully-qualified name that identifies a QoS within the uri
     *                  associated with the QosProvider or a name that identifies a QoS within the
     *                  uri associated with the QosProvider instance relative to its default QoS
     *                  profile. Id’s starting with ‘::’ are interpreted as fully-qualified names and all
     *                  others are interpreted as names relative to the default QoS profile of the
     *                  QosProvider instance. When id is NULL it is interpreted as a non-named QoS
     *                  within the default QoS profile associated with the QosProvider.
     * @return SubscriberQos from the given URI (and profile) using the id
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no SubscriberQos that matches the provided id can be
     *                  found within the uri associated with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::sub::qos::SubscriberQos subscriber_qos(const std::string& id);

    /**
     * Resolves the DataReaderQos from the uri this QosProvider is associated with.
     *
     * @return DataReadertQos from the given URI (and profile)
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no DataReaderQos can be found within the uri associated
     *                  with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::sub::qos::DataReaderQos datareader_qos();

    /**
     * Resolves the DataReaderQos identified by the id from the uri this
     * QosProvider is associated with.
     *
     * @param id        The fully-qualified name that identifies a QoS within the uri
     *                  associated with the QosProvider or a name that identifies a QoS within the
     *                  uri associated with the QosProvider instance relative to its default QoS
     *                  profile. Id’s starting with ‘::’ are interpreted as fully-qualified names and all
     *                  others are interpreted as names relative to the default QoS profile of the
     *                  QosProvider instance. When id is NULL it is interpreted as a non-named QoS
     *                  within the default QoS profile associated with the QosProvider.
     * @return DataReaderQos from the given URI (and profile) using the id
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no DataReaderQos that matches the provided id can be
     *                  found within the uri associated with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::sub::qos::DataReaderQos datareader_qos(const std::string& id);

    /**
     * Resolves the PublisherQos from the uri this QosProvider is associated with.
     *
     * @return PublisherQos from the given URI (and profile)
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no PublisherQos can be found within the uri associated
     *                  with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::pub::qos::PublisherQos publisher_qos();

    /**
     * Resolves the PublisherQos identified by the id from the uri this
     * QosProvider is associated with.
     *
     * @param id        The fully-qualified name that identifies a QoS within the uri
     *                  associated with the QosProvider or a name that identifies a QoS within the
     *                  uri associated with the QosProvider instance relative to its default QoS
     *                  profile. Id’s starting with ‘::’ are interpreted as fully-qualified names and all
     *                  others are interpreted as names relative to the default QoS profile of the
     *                  QosProvider instance. When id is NULL it is interpreted as a non-named QoS
     *                  within the default QoS profile associated with the QosProvider.
     * @return PublisherQos from the given URI (and profile) using the id
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no PublisherQos that matches the provided id can be
     *                  found within the uri associated with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::pub::qos::PublisherQos publisher_qos(const std::string& id);

    /**
     * Resolves the DataWriterQos from the uri this QosProvider is associated with.
     *
     * @return DataWriterQos from the given URI (and profile)
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no DataWriterQos can be found within the uri associated
     *                  with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::pub::qos::DataWriterQos datawriter_qos();

    /**
     * Resolves the DataWriterQos identified by the id from the uri this
     * QosProvider is associated with.
     *
     * @param id        The fully-qualified name that identifies a QoS within the uri
     *                  associated with the QosProvider or a name that identifies a QoS within the
     *                  uri associated with the QosProvider instance relative to its default QoS
     *                  profile. Id’s starting with ‘::’ are interpreted as fully-qualified names and all
     *                  others are interpreted as names relative to the default QoS profile of the
     *                  QosProvider instance. When id is NULL it is interpreted as a non-named QoS
     *                  within the default QoS profile associated with the QosProvider.
     * @return DataWriterQos from the given URI (and profile) using the id
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::PreconditionNotMetError
     *                  If no DataWriterQos that matches the provided id can be
     *                  found within the uri associated with the QosProvider.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    dds::pub::qos::DataWriterQos datawriter_qos(const std::string& id);
};

#endif /* OMG_DDS_CORE_QOS_TPROVIDER_HPP_ */
