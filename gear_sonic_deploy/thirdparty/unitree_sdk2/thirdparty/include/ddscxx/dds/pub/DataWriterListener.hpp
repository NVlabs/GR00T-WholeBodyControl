#ifndef OMG_DDS_PUB_DATA_WRITER_LISTENER_HPP_
#define OMG_DDS_PUB_DATA_WRITER_LISTENER_HPP_

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


#include <dds/pub/DataWriter.hpp>

namespace dds
{
namespace pub
{

/**
 *  * @brief
 * DataWriter events Listener
 *
 * Since a DataWriter is an Entity, it has the ability to have a Listener
 * associated with it. In this case, the associated Listener should be of type
 * DataWriterListener. This interface must be implemented by the
 * application. A user-defined class must be provided by the application which must
 * extend from the DataWriterListener class.
 *
 * <b><i>
 * All operations for this interface must be implemented in the user-defined class, it is
 * up to the application whether an operation is empty or contains some functionality.
 * </i></b>
 *
 * The DataWriterListener provides a generic mechanism (actually a
 * callback function) for the Data Distribution Service to notify the application of
 * relevant asynchronous status change events, such as a missed deadline, violation of
 * a QosPolicy setting, etc. The DataWriterListener is related to
 * changes in communication status StatusConditions.
 *
 * @code{.cpp}
 * // Application example listener
 * class ExampleListener :
 *                public virtual dds::pub::DataWriterListener<Foo::Bar>
 * {
 * public:
 *     virtual void on_offered_deadline_missed (
 *         dds::pub::DataWriter<Foo::Bar>& writer,
 *         const dds::core::status::OfferedDeadlineMissedStatus& status)
 *     {
 *         std::cout << "on_offered_deadline_missed" << std::endl;
 *     }
 *
 *     virtual void on_offered_incompatible_qos (
 *         dds::pub::DataWriter<Foo::Bar>& writer,
 *         const dds::core::status::OfferedIncompatibleQosStatus& status)
 *     {
 *         std::cout << "on_offered_incompatible_qos" << std::endl;
 *     }
 *
 *     virtual void on_liveliness_lost (
 *         dds::pub::DataWriter<Foo::Bar>& writer,
 *         const dds::core::status::LivelinessLostStatus& status)
 *     {
 *         std::cout << "on_liveliness_lost" << std::endl;
 *     }
 *
 *     virtual void on_publication_matched (
 *         dds::pub::DataWriter<Foo::Bar>& writer,
 *         const dds::core::status::PublicationMatchedStatus& status)
 *     {
 *         std::cout << "on_publication_matched" << std::endl;
 *     }
 * };
 *
 * // Create DataWriter with the listener
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
 * dds::pub::Publisher publisher(participant);
 * dds::pub::DataWriter<Foo::Bar> writer(publisher,
 *                                       topic,
 *                                       publisher.default_datawriter_qos(),
 *                                       new ExampleListener(),
 *                                       dds::core::status::StatusMask::all());
 *
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Publication_DataWriter "Data Writer"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Listener "Listener information"
 */
template <typename T>
class DataWriterListener
{
public:
    /** @cond */
    virtual ~DataWriterListener() { }
    /** @endcond */

public:
    /**
     * This operation is called by the Data Distribution Service when the
     * OfferedDeadlineMissedStatus changes.
     *
     * This operation will only be called when
     * the relevant DataWriterListener is installed and enabled for the offered
     * deadline missed status (StatusMask::offered_deadline_missed()). The
     * offered deadline missed status will change when the
     * deadline that the DataWriter has committed through its DeadlineQosPolicy
     * was not respected for a specific instance.
     *
     * @param writer contain a pointer to the DataWriter on which
     *               the OfferedDeadlineMissedStatus has changed (this is an input to the
     *               application)
     * @param status contain the
     *               OfferedDeadlineMissedStatus object (this is an input to
     *               the application).
     */
    virtual void on_offered_deadline_missed(
        dds::pub::DataWriter<T>& writer,
        const dds::core::status::OfferedDeadlineMissedStatus& status) = 0;

    /**
     * This operation called by the Data Distribution Service when the
     * OfferedIncompatibleQosStatus changes.
     *
     * This operation will only be called when
     * the relevant DataWriterListener is installed and enabled for the
     * StatusMask::offered_incompatible_qos(). The incompatible Qos status will
     * change when a DataReader object has been discovered by the DataWriter with
     * the same Topic and a requested DataReaderQos that was incompatible with the
     * one offered by the DataWriter.
     *
     * @param writer contain a pointer to the DataWriter on which
     *               the OfferedIncompatibleQosStatus has changed (this is an input to
     *               the application).
     * @param status contain the OfferedIncompatibleQosStatus object (this is
     *               an input to the application).
     */
    virtual void on_offered_incompatible_qos(
        dds::pub::DataWriter<T>& writer,
        const dds::core::status::OfferedIncompatibleQosStatus&  status) = 0;

    /**
     * This operation is called by the Data Distribution Service when the
     * LivelinessLostStatus changes.
     *
     * This operation will only be called when the relevant
     * DataWriterListener is installed and enabled for the liveliness lost status
     * (StatusMask::liveliness_lost()).
     * The liveliness lost status will change when the liveliness that the DataWriter has
     * committed through its LivelinessQosPolicy was not respected. In other words,
     * the DataWriter failed to actively signal its liveliness within the offered liveliness
     * period. As a result, the DataReader objects will consider the DataWriter as no
     * longer “alive”.
     *
     * @param writer contains a pointer to the DataWriter on which
     *               the LivelinessLostStatus has changed (this is an input to
     *               the application).
     * @param status contains the LivelinessLostStatus object (this is an input
     *               to the application).
     */
    virtual void on_liveliness_lost(
        dds::pub::DataWriter<T>& writer,
        const dds::core::status::LivelinessLostStatus& status) = 0;

    /**
     * This operation is called by the Data
     * Distribution Service when a new match has been discovered for the current
     * publication, or when an existing match has ceased to exist.
     *
     * Usually this means that a
     * new DataReader that matches the Topic and that has compatible Qos as the current
     * DataWriter has either been discovered, or that a previously discovered
     * DataReader has ceased to be matched to the current DataWriter. A DataReader
     * may cease to match when it gets deleted, when it changes its Qos to a value that is
     * incompatible with the current DataWriter or when either the DataWriter or the
     * DataReader has chosen to put its matching counterpart on its ignore-list using the
     * dds::sub::ignore or dds::pub::ignore operations.
     *
     * it will only be called when the relevant DataWriterListener is installed and enabled
     * for the StatusMask::publication_matched().
     *
     * @param writer contains a pointer to the DataWriter for which
     *               a match has been discovered (this is an input to the application provided by the
     *               Data Distribution Service).
     * @param status contains the
     *               PublicationMatchedStatus object (this is an input to the application
     *               provided by the Data Distribution Service).
     */
    virtual void on_publication_matched(
        dds::pub::DataWriter<T>& writer,
        const dds::core::status::PublicationMatchedStatus& status) = 0;
};


/**
 * @brief
 * DataWriter events Listener
 *
 * This listener is just like DataWriterListener, except
 * that the application doesn't have to implement all operations.
 *
 * @code{.cpp}
 * class ExampleListener : public virtual dds::pub::NoOpDataWriterListener<Foo::Bar>
 * {
 *    // Not necessary to implement any Listener operations.
 * };
 * @endcode
 *
 * @see dds::pub::DataWriterListener
 */
template <typename T>
class NoOpDataWriterListener : public virtual DataWriterListener<T>
{
/** @cond
 * All these functions have already been documented in the non-NoOp listener.
 * Ignore these functions for the doxygen API documentation for clarity.
 */
public:
    virtual ~NoOpDataWriterListener() { }

public:
    virtual void
    on_offered_deadline_missed(
        dds::pub::DataWriter<T>&,
        const dds::core::status::OfferedDeadlineMissedStatus&) { }

    virtual void
    on_offered_incompatible_qos(
        dds::pub::DataWriter<T>&,
        const dds::core::status::OfferedIncompatibleQosStatus&) { }

    virtual void
    on_liveliness_lost(
        dds::pub::DataWriter<T>&,
        const dds::core::status::LivelinessLostStatus&) { }

    virtual void
    on_publication_matched(
        dds::pub::DataWriter<T>&,
        const dds::core::status::PublicationMatchedStatus&) { }
/** @endcond */
};

}
}

#endif /* OMG_DDS_PUB_DATA_WRITER_LISTENER_HPP_ */
