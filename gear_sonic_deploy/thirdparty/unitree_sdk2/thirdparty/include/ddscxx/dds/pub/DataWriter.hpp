#ifndef OMG_DDS_PUB_DATA_WRITER_HPP_
#define OMG_DDS_PUB_DATA_WRITER_HPP_

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

#include <dds/core/InstanceHandle.hpp>
#include <dds/topic/Topic.hpp>
#include <dds/topic/TopicInstance.hpp>
//#include <dds/topic/BuiltinTopic.hpp>
#include <dds/pub/Publisher.hpp>
#include <dds/pub/AnyDataWriter.hpp>

#include <dds/pub/detail/DataWriter.hpp>

/** @cond */
namespace dds
{
namespace pub
{
template <typename T,
          template <typename Q> class DELEGATE = dds::pub::detail::DataWriter >
class DataWriter;

template <typename T> class DataWriterListener;
}
}
/** @endcond */

/**
 * @brief
 * DataWriter allows the application to set the value of the sample to be published
 * under a given Topic.
 *
 * A DataWriter is attached to exactly one Publisher.
 *
 * A DataWriter is bound to exactly one Topic and therefore to exactly one data
 * type. The Topic must exist prior to the DataWriter's creation.
 * DataWriter is an abstract class. It must be specialized for each particular
 * application data type. For a fictional application data type Bar (defined in the
 * module Foo) the specialized class would be dds::pub::DataWriter<Foo::Bar>.
 *
 * The pre-processor generates from IDL type descriptions the application
 * DataWriter<type> classes. For each application data type that is used as Topic
 * data type, a typed class DataWriter<type> is derived from the AnyDataWriter
 * class.
 *
 * For instance, for an application, the definitions are located in the Foo.idl file.
 * The pre-processor will generate a ccpp_Foo.h include file.
 *
 * <b>General note:</b> The name ccpp_Foo.h is derived from the IDL file Foo.idl,
 * that defines Foo::Bar, for all relevant DataWriter<Foo::Bar> operations.
 *
 * @note Apart from idl files, Google protocol buffers are also supported. For the
 *       API itself, it doesn't matter if the type header files were generated from
 *       idl or protocol buffers. The resulting API usage and includes remain the same.
 *
 * @code{.cpp}
 * // Default creation of a DataWriter
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
 * dds::pub::Publisher publisher(participant);
 * dds::pub::DataWriter<Foo::Bar> writer(publisher, topic);
 *
 * // Default write of a sample on the DataWriter
 * Foo::Bar sample;
 * writer.write(sample);
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Publication "Publication concept"
 * @see for more information: @ref DCPS_Modules_Publication_DataWriter "DataWriter concept"
 */
template <typename T, template <typename Q> class DELEGATE>
class dds::pub::DataWriter : public ::dds::pub::TAnyDataWriter< DELEGATE<T> >
{

public:
    /**
     * Local convenience typedef for dds::pub::DataWriterListener.
     */
    typedef dds::pub::DataWriterListener<T>              Listener;

public:
    OMG_DDS_REF_TYPE_PROTECTED_DC_T(DataWriter, dds::pub::TAnyDataWriter, T, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(DataWriter)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_DEFAULT(DataWriter)

public:

    /**
     * Create a new DataWriter for the desired Topic, using the given Publisher.
     *
     * The DataWriter will be created with the QoS values specified on the last
     * successful call to @link dds::pub::Publisher::default_datawriter_qos(const dds::pub::qos::DataWriterQos& qos)
     * pub.default_datawriter_qos(qos) @endlink or, if the call was never made, the
     * @ref anchor_dds_pub_datawriter_qos_defaults "default" values.
     *
     * @param pub the Publisher that will contain this DataWriter
     * @param topic the Topic associated with this DataWriter
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    DataWriter(const dds::pub::Publisher& pub,
               const ::dds::topic::Topic<T>& topic);

    /**
     * Create a new DataWriter for the desired Topic, using the given Publisher and
     * DataWriterQos and attaches the optionally specified DataWriterListener to it.
     *
     * <i>QoS</i><br>
     * A possible application pattern to construct the DataWriterQos for the
     * DataWriter is to:
     * @code{.cpp}
     * // 1) Retrieve the QosPolicy settings on the associated Topic
     * dds::topic::qos::TopicQos topicQos = topic.qos();
     * // 2) Retrieve the default DataWriterQos from the related Publisher
     * dds::pub::qos::DataWriterQos writerQos = publisher.default_datawriter_qos();
     * // 3) Combine those two lists of QosPolicy settings by overwriting DataWriterQos
     * //    policies that are also present TopicQos
     * writerQos = topicQos;
     * // 4) Selectively modify QosPolicy settings as desired.
     * writerQos << dds::core::policy::WriterDataLifecycle::ManuallyDisposeUnregisteredInstances();
     * // 5) Use the resulting QoS to construct the DataWriter.
     * dds::pub::DataWriter<Foo::Bar> writer(publisher, topic, writerQos);
     * @endcode
     *
     * <i>Listener</i><br>
     * The following statuses are applicable to the DataWriterListener:
     *  - dds::core::status::StatusMask::offered_deadline_missed()
     *  - dds::core::status::StatusMask::offered_incompatible_qos()
     *  - dds::core::status::StatusMask::liveliness_lost()
     *  - dds::core::status::StatusMask::publication_matched()
     *
     * See @ref DCPS_Modules_Infrastructure_Listener "listener concept",
     * @ref anchor_dds_pub_datawriter_commstatus "communication status" and
     * @ref anchor_dds_pub_datawriter_commpropagation "communication propagation"
     * for more information.
     *
     * @param pub the Publisher that will contain this DataWriter
     * @param topic the Topic associated with this DataWriter
     * @param qos the DataWriter qos.
     * @param listener the DataWriter listener.
     * @param mask the listener event mask.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::InconsistentPolicyError
     *                  The parameter qos contains conflicting QosPolicy settings.
     */
    DataWriter(const dds::pub::Publisher& pub,
               const ::dds::topic::Topic<T>& topic,
               const dds::pub::qos::DataWriterQos& qos,
               dds::pub::DataWriterListener<T>* listener = NULL,
               const dds::core::status::StatusMask& mask = ::dds::core::status::StatusMask::none());


public:
    //==========================================================================
    //== Write API

    /**
     * This operation modifies the value of a data instance.
     *
     * <b>Detailed Description</b><br>
     * This operation modifies the value of a data instance. When this operation is used,
     * the Data Distribution Service will automatically supply the value of the
     * source_timestamp that is made available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * @anchor anchor_dds_pub_datawriter_write_blocking
     * <i>Blocking</i><br>
     * If the dds::core::policy::History QosPolicy is set to KEEP_ALL, the write
     * operation on the DataWriter may block if the modification would cause data to be
     * lost because one of the limits, specified in the dds::core::policy::ResourceLimits, is
     * exceeded. In case the synchronous attribute value of the
     * dds::core::policy::Reliability is set to TRUE for communicating DataWriters and
     * DataReaders then the DataWriter will wait until all synchronous
     * DataReaders have acknowledged the data. Under these circumstances, the
     * max_blocking_time attribute of the dds::core::policy::Reliability configures the
     * maximum time the write operation may block (either waiting for space to become
     * available or data to be acknowledged). If max_blocking_time elapses before the
     * DataWriter is able to store the modification without exceeding the limits and all
     * expected acknowledgements are received, the write operation will fail and throw
     * TimeoutError.
     *
     * @param sample the sample to be written
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    void write(const T& sample);

    /**
     * This operation modifies the value of a data instance and provides a value for the
     * source_timestamp explicitly.
     *
     * <b>Detailed Description</b><br>
     * It modifies the values of the given data instances. When this operation is used,
     * the application provides the value for the parameter source_timestamp that is made
     * available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_write_blocking "write blocking").
     *
     * @param sample the sample to be written
     * @param timestamp the timestamp used for this sample
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    void write(const T& sample, const dds::core::Time& timestamp);

    /**
     * This operation modifies the value of a data instance.
     *
     * <b>Detailed Description</b><br>
     * This operation modifies the value of a data instance. When this operation is used,
     * the Data Distribution Service will automatically supply the value of the
     * source_timestamp that is made available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * Before writing data to an instance, the instance may be registered with the
     * @ref dds::pub::DataWriter::register_instance(const T& key) "\"register_instance\"" or
     * @ref dds::pub::DataWriter::register_instance(const T& key, const dds::core::Time& timestamp) "\"register_instance_w_timestamp\""
     * The handle returned by one of the register_instance operations can be supplied to
     * the parameter handle of the write operation. However, it is also possible to
     * supply a default InstanceHandle (InstanceHandle.is_nil() == true), which means
     * that the identity of the instance is automatically deduced from the instance_data
     * (identified by the key).
     *
     * @anchor anchor_dds_pub_datawriter_write_instance_handle
     * <i>Instance Handle</i><br>
     * The default InstanceHandle (InstanceHandle.is_nil() == true) can be used for the
     * parameter handle. This indicates the identity of the instance is automatically deduced
     * from the instance_data (by means of the key).
     *
     * If handle is not nil, it must correspond to the value returned by
     * @ref dds::pub::DataWriter::register_instance(const T& key) "\"register_instance\"" or
     * @ref dds::pub::DataWriter::register_instance(const T& key, const dds::core::Time& timestamp) "\"register_instance_w_timestamp\""
     * when the instance (identified by its key) was registered. Passing such a registered
     * handle helps the Data Distribution Service to process the sample more efficiently.<br>
     * If there is no correspondence between handle and sample, the result of the operation
     * is unspecified.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_write_blocking "write blocking").
     *
     * @param sample the sample to be written
     * @param instance the handle representing the instance written
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    void write(const T& sample, const ::dds::core::InstanceHandle& instance);

    /**
     * This operation modifies the value of a data instance and provides a value for the
     * source_timestamp explicitly.
     *
     * <b>Detailed Description</b><br>
     * It modifies the values of the given data instances. When this operation is used,
     * the application provides the value for the parameter source_timestamp that is made
     * available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Instance Handle</i><br>
     * See @ref anchor_dds_pub_datawriter_write_instance_handle "write instance handle".
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_write_blocking "write blocking").
     *
     * @param data the sample to be written
     * @param instance the handle representing the instance written
     * @param timestamp the timestamp to use for this sample
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    void write(const T& data,
               const ::dds::core::InstanceHandle& instance,
               const dds::core::Time& timestamp);


    /**
     * This operation modifies the value of a data instance.
     *
     * <b>Detailed Description</b><br>
     * This operation modifies the value of a data instance. When this operation is used,
     * the Data Distribution Service will automatically supply the value of the
     * source_timestamp that is made available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Topic Instance</i><br>
     * A TopicInstance encapsulates a sample and its associated
     * @ref anchor_dds_pub_datawriter_write_instance_handle "instance handle".
     *
     *
     * @param i the instance to write
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    void write(const dds::topic::TopicInstance<T>& i);

    /**
     * This operation modifies the value of a data instance and provides a value for the
     * source_timestamp explicitly.
     *
     * <b>Detailed Description</b><br>
     * It modifies the values of the given data instances. When this operation is used,
     * the application provides the value for the parameter source_timestamp that is made
     * available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Topic Instance</i><br>
     * A TopicInstance encapsulates a sample and its associated
     * @ref anchor_dds_pub_datawriter_write_instance_handle "instance handle".
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_write_blocking "write blocking").
     *
     * @param i the instance to write
     * @param timestamp the timestamp for this sample
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    void write(const dds::topic::TopicInstance<T>& i,
               const dds::core::Time& timestamp);

    /**
     * This operation writes a series of typed Samples or TopicInstances.
     *
     * <b>Detailed Description</b><br>
     * This operation takes a sequence of typed Samples or TopicInstances, which
     * is determined by the template specialization.
     *
     * It modifies the values of the given data instances. When this operation is used,
     * the Data Distribution Service will automatically supply the value of the
     * source_timestamp that is made available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Topic Instance</i><br>
     * A TopicInstance encapsulates a typed Sample and its associated
     * @ref anchor_dds_pub_datawriter_write_instance_handle "instance handle".
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_write_blocking "write blocking").
     *
     * @param begin An iterator pointing to the beginning of a sequence of
     *              Samples or a sequence of TopicInstances
     * @param end   An iterator pointing to the end of a sequence of
     *              Samples or a sequence of TopicInstances
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    template <typename FWIterator>
    void write(const FWIterator& begin, const FWIterator& end);

    /**
     * This operation writes a series of typed Samples or TopicInstances and provides
     * a value for the source_timestamp for these samples explicitly.
     *
     * <b>Detailed Description</b><br>
     * This operation takes a sequence of typed Samples or TopicInstances, which
     * is determined by the template specialization.
     *
     * It modifies the values of the given data instances. When this operation is used,
     * the application provides the value for the parameter source_timestamp that is made
     * available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Topic Instance</i><br>
     * A TopicInstance encapsulates a sample and its associated
     * @ref anchor_dds_pub_datawriter_write_instance_handle "instance handle".
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_write_blocking "write blocking").
     *
     * @param begin an iterator pointing to the beginning of a sequence of
     * TopicInstances
     * @param end an iterator pointing to the end of a sequence of
     * TopicInstances
     * @param timestamp the time stamp
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    template <typename FWIterator>
    void write(const FWIterator& begin, const FWIterator& end,
               const dds::core::Time& timestamp);

    /**
     * This operation writes a series of typed Samples and their parallel instance handles.
     *
     * <b>Detailed Description</b><br>
     * It modifies the values of the given data instances. When this operation is used,
     * the Data Distribution Service will automatically supply the value of the
     * source_timestamp that is made available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Instance Handle</i><br>
     * See @ref anchor_dds_pub_datawriter_write_instance_handle "write instance handle".
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_write_blocking "write blocking").
     *
     * @param data_begin    an iterator pointing to the beginning of a sequence of samples
     * @param data_end      an iterator pointing to the end of a sequence of samples
     * @param handle_begin  an iterator pointing to the beginning of a sequence of InstanceHandles
     * @param handle_end    an iterator pointing to the end of a sequence of InstanceHandles
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    template <typename SamplesFWIterator, typename HandlesFWIterator>
    void write(const SamplesFWIterator& data_begin,
               const SamplesFWIterator& data_end,
               const HandlesFWIterator& handle_begin,
               const HandlesFWIterator& handle_end);

    /**
     * This operation writes a series of typed Samples or TopicInstances and provides
     * a value for the source_timestamp for these samples explicitly.
     *
     * <b>Detailed Description</b><br>
     * It modifies the values of the given data instances. When this operation is used,
     * the application provides the value for the parameter source_timestamp that is made
     * available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Instance Handle</i><br>
     * See @ref anchor_dds_pub_datawriter_write_instance_handle "write instance handle".
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_write_blocking "write blocking").
     *
     * @param data_begin    an iterator pointing to the beginning of a sequence of samples
     * @param data_end      an iterator pointing to the end of a sequence of samples
     * @param handle_begin  an iterator pointing to the beginning of a sequence of InstanceHandles
     * @param handle_end    an iterator pointing to the end of a sequence of InstanceHandles
     * @param timestamp     the time stamp
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    template <typename SamplesFWIterator, typename HandlesFWIterator>
    void write(const SamplesFWIterator& data_begin,
               const SamplesFWIterator& data_end,
               const HandlesFWIterator& handle_begin,
               const HandlesFWIterator& handle_end,
               const dds::core::Time& timestamp);


    /** @copydoc dds::pub::DataWriter::write(const T& data) */
    DataWriter& operator << (const T& data);

    /** @copydoc dds::pub::DataWriter::write(const T& sample, const dds::core::Time& timestamp) */
    DataWriter& operator << (const std::pair<T, dds::core::Time>& data);

    /** @copydoc dds::pub::DataWriter::write(const T& sample, const ::dds::core::InstanceHandle& instance) */
    DataWriter& operator << (const std::pair<T, ::dds::core::InstanceHandle>& data);

    /** @cond
     * This can be useful for the DataReader (see fi MaxSamplesManipulatorFunctor), but not
     * really for the DataWriter. Leave it from the API documentation for clarity.
     */
    DataWriter& operator <<(DataWriter & (*manipulator)(DataWriter&));
    /** @endcond */

    //==========================================================================
    //== Instance Management

    /**
     * This operation informs the Data Distribution Service that the application will be
     * modifying a particular instance.
     *
     * <b>Detailed Description</b><br>
     * This operation informs the Data Distribution Service that the application will be
     * modifying a particular instance. This operation may be invoked prior to calling any
     * operation that modifies the instance, such as write, unregister_instance or
     * dispose_instance.<br>
     * When the application does register the instance before modifying, the Data
     * Distribution Service will handle the instance more efficiently. It takes as a parameter
     * (instance_data) an instance (to get the key value) and returns a handle that can
     * be used in successive DataWriter operations. In case of an error, a HANDLE_NIL
     * handle (InstanceHandle.is_nil() == true) is returned.
     *
     * The explicit use of this operation is optional as the application can directly call the
     * write, unregister_instance or dispose_instance operations without InstanceHandle,
     * which indicate that the sample should be examined to identify the instance.
     *
     * When this operation is used, the Data Distribution Service will automatically supply
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. <br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * @anchor anchor_dds_pub_datawriter_register_blocking
     * <i>Blocking</i><br>
     * If the dds::core::policy::History QosPolicy is set to KEEP_ALL, the register_instance
     * operation on the DataWriter may block if the modification would cause data to be
     * lost because one of the limits, specified in the dds::core::policy::ResourceLimits, is
     * exceeded. In case the synchronous attribute value of the
     * dds::core::policy::Reliability is set to TRUE for communicating DataWriters and
     * DataReaders then the DataWriter will wait until all synchronous
     * DataReaders have acknowledged the data. Under these circumstances, the
     * max_blocking_time attribute of the dds::core::policy::Reliability configures the
     * maximum time the register operation may block (either waiting for space to become
     * available or data to be acknowledged). If max_blocking_time elapses before the
     * DataWriter is able to store the modification without exceeding the limits and all
     * expected acknowledgements are received, the register_instance operation will fail
     * will return a nil InstanceHandle.
     *
     * <i>Multiple Calls</i><br>
     * If this operation is called for an already registered instance, it just returns the already
     * allocated instance handle. This may be used to look up and retrieve the handle
     * allocated to a given instance.
     *
     * <i>Key</i><br>
     * The key is a typed Sample of which the key fields are set so that the instance
     * can be identified.
     *
     * @param key the key of the instance to register
     * @return the instance handle
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     */
    const ::dds::core::InstanceHandle register_instance(const T& key);

    /**
     * This operation will inform the Data Distribution Service that the application will be
     * modifying a particular instance and provides a value for the source_timestamp
     * explicitly.
     *
     * <b>Detailed Description</b><br>
     * This operation informs the Data Distribution Service that the application will be
     * modifying a particular instance. This operation may be invoked prior to calling any
     * operation that modifies the instance, such as write, unregister_instance or
     * dispose_instance.<br>
     * When the application does register the instance before modifying, the Data
     * Distribution Service will handle the instance more efficiently. It takes as a parameter
     * (instance_data) an instance (to get the key value) and returns a handle that can
     * be used in successive DataWriter operations. In case of an error, a HANDLE_NIL
     * handle (InstanceHandle.is_nil() == true) is returned.
     *
     * The explicit use of this operation is optional as the application can directly call the
     * write, unregister_instance or dispose_instance operations without InstanceHandle,
     * which indicate that the sample should be examined to identify the instance.
     *
     * When this operation is used, the application provides the value for the parameter
     * source_timestamp that is made available to connected DataReader objects.<br>
     * This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_register_blocking "register blocking").
     *
     * <i>Multiple Calls</i><br>
     * If this operation is called for an already registered instance, it just returns the already
     * allocated instance handle. The source_timestamp is ignored in that case.
     *
     * <i>Key</i><br>
     * The key is a typed Sample of which the key fields are set so that the instance
     * can be identified.
     *
     * @param key the key of the instance to register
     * @param timestamp the timestamp used for registration
     * @return the instance handle
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     */
    const ::dds::core::InstanceHandle register_instance(const T& key,
            const dds::core::Time& timestamp);

    /**
     * This operation informs the Data Distribution Service that the application will not be
     * modifying a particular instance any more.
     *
     * <b>Detailed Description</b><br>
     * This operation informs the Data Distribution Service that the application will not be
     * modifying a particular instance any more. Therefore, this operation reverses the
     * action of @ref dds::pub::DataWriter::register_instance(const T& key) "\"register_instance\"" or
     * @ref dds::pub::DataWriter::register_instance(const T& key, const dds::core::Time& timestamp) "\"register_instance_w_timestamp\"".
     * register_instance or register_instance_w_timestamp.<br>
     * It should only be called on an instance that is currently registered. This operation
     * should be called just once per instance, regardless of how many times
     * @ref dds::pub::DataWriter::register_instance(const T& key) "register_instance" was called
     * for that instance.<br>
     * This operation also indicates
     * that the Data Distribution Service can locally remove all information regarding that
     * instance. The application should not attempt to use the handle, previously
     * allocated to that instance, after calling this operation.
     *
     * When this operation is used, the Data Distribution Service will automatically supply
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * @anchor anchor_dds_pub_datawriter_unregister_effects
     * <i>Effects</i><br>
     * If, after unregistering, the application wants to modify (write or dispose) the
     * instance, it has to register the instance again, or it has to use the default
     * instance handle (InstanceHandle.is_nil() == true).
     * This operation does not indicate that the instance should be deleted (that is the
     * purpose of the @ref dds::pub::DataWriter::dispose_instance(const T& key) "dispose".
     * This operation just indicates that the DataWriter no longer
     * has “anything to say” about the instance. If there is no other DataWriter that
     * has registered the instance as well, then the dds::sub::status::InstanceState in all
     * connected DataReaders will be changed to not_alive_no_writers, provided this
     * InstanceState was not already set to not_alive_disposed. In the last case the
     * InstanceState will not be effected by the unregister_instance call,
     * see also @ref DCPS_Modules_Subscription_SampleInfo "Sample info concept".
     *
     * This operation can affect the ownership of the data instance. If the
     * DataWriter was the exclusive owner of the instance, calling this operation will
     * release that ownership, meaning ownership may be transferred to another,
     * possibly lower strength, DataWriter.
     *
     * The operation must be called only on registered instances. Otherwise the operation
     * trow PreconditionNotMetError.
     *
     * @anchor anchor_dds_pub_datawriter_unregister_blocking
     * <i>Blocking</i><br>
     * If the dds::core::policy::History QosPolicy is set to KEEP_ALL, the unregister_instance
     * operation on the DataWriter may block if the modification would cause data to be
     * lost because one of the limits, specified in the dds::core::policy::ResourceLimits, is
     * exceeded. In case the synchronous attribute value of the
     * dds::core::policy::Reliability is set to TRUE for communicating DataWriters and
     * DataReaders then the DataWriter will wait until all synchronous
     * DataReaders have acknowledged the data. Under these circumstances, the
     * max_blocking_time attribute of the dds::core::policy::Reliability configures the
     * maximum time the unregister operation may block (either waiting for space to become
     * available or data to be acknowledged). If max_blocking_time elapses before the
     * DataWriter is able to store the modification without exceeding the limits and all
     * expected acknowledgements are received, the unregister_instance operation will fail
     * and throw TimeoutError.
     *
     * @param i the instance to unregister
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    DataWriter& unregister_instance(const ::dds::core::InstanceHandle& i);

    /**
     * This operation will inform the Data Distribution Service that the application will not
     * be modifying a particular instance any more and provides a value for the
     * source_timestamp explicitly.
     *
     * <b>Detailed Description</b><br>
     * This operation informs the Data Distribution Service that the application will not be
     * modifying a particular instance any more. Therefore, this operation reverses the
     * action of @ref dds::pub::DataWriter::register_instance(const T& key) "\"register_instance\"" or
     * @ref dds::pub::DataWriter::register_instance(const T& key, const dds::core::Time& timestamp) "\"register_instance_w_timestamp\"".
     * register_instance or register_instance_w_timestamp.<br>
     * It should only be called on an instance that is currently registered. This operation
     * should be called just once per instance, regardless of how many times
     * @ref dds::pub::DataWriter::register_instance(const T& key) "register_instance" was called
     * for that instance.<br>
     * This operation also indicates
     * that the Data Distribution Service can locally remove all information regarding that
     * instance. The application should not attempt to use the handle, previously
     * allocated to that instance, after calling this operation.
     *
     * When this operation is used, the application itself supplied
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * <i>Effects</i><br>
     * See @ref anchor_dds_pub_datawriter_unregister_effects "here" for the unregister effects.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_unregister_blocking "unregister blocking").
     *
     * @param i the instance to unregister
     * @param timestamp the timestamp
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    DataWriter& unregister_instance(const ::dds::core::InstanceHandle& i,
                                    const dds::core::Time& timestamp);

    /**
     * This operation informs the Data Distribution Service that the application will not be
     * modifying a particular instance any more.
     *
     * <b>Detailed Description</b><br>
     * This operation informs the Data Distribution Service that the application will not be
     * modifying a particular instance any more. Therefore, this operation reverses the
     * action of @ref dds::pub::DataWriter::register_instance(const T& key) "\"register_instance\"" or
     * @ref dds::pub::DataWriter::register_instance(const T& key, const dds::core::Time& timestamp) "\"register_instance_w_timestamp\"".
     * register_instance or register_instance_w_timestamp.<br>
     * It should only be called on an instance that is currently registered. This operation
     * should be called just once per instance, regardless of how many times
     * @ref dds::pub::DataWriter::register_instance(const T& key) "register_instance" was called
     * for that instance.<br>
     * This operation also indicates
     * that the Data Distribution Service can locally remove all information regarding that
     * instance. The application should not attempt to use the handle, previously
     * allocated to that instance, after calling this operation.
     *
     * When this operation is used, the Data Distribution Service will automatically supply
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * <i>Effects</i><br>
     * See @ref anchor_dds_pub_datawriter_unregister_effects "here" for the unregister effects.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_unregister_blocking "unregister blocking").
     *
     * <i>Instance</i><br>
     * The instance is identified by the key fields of the given typed data sample, instead
     * of an InstanceHandle.
     *
     * @param key sample of the instance to dispose
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    DataWriter& unregister_instance(const T& key);

    /**
     * This operation will inform the Data Distribution Service that the application will not
     * be modifying a particular instance any more and provides a value for the
     * source_timestamp explicitly.
     *
     * <b>Detailed Description</b><br>
     * This operation informs the Data Distribution Service that the application will not be
     * modifying a particular instance any more. Therefore, this operation reverses the
     * action of @ref dds::pub::DataWriter::register_instance(const T& key) "\"register_instance\"" or
     * @ref dds::pub::DataWriter::register_instance(const T& key, const dds::core::Time& timestamp) "\"register_instance_w_timestamp\"".
     * register_instance or register_instance_w_timestamp.<br>
     * It should only be called on an instance that is currently registered. This operation
     * should be called just once per instance, regardless of how many times
     * @ref dds::pub::DataWriter::register_instance(const T& key) "register_instance" was called
     * for that instance.<br>
     * This operation also indicates
     * that the Data Distribution Service can locally remove all information regarding that
     * instance. The application should not attempt to use the handle, previously
     * allocated to that instance, after calling this operation.
     *
     * When this operation is used, the application itself supplied
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * <i>Effects</i><br>
     * See @ref anchor_dds_pub_datawriter_unregister_effects "here" for the unregister effects.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_unregister_blocking "unregister blocking").
     *
     * <i>Instance</i><br>
     * The instance is identified by the key fields of the given typed data sample, instead
     * of an InstanceHandle.
     *
     * @param key sample of the instance to dispose
     * @param timestamp the timestamp
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    DataWriter& unregister_instance(const T& key,
                                    const dds::core::Time& timestamp);

    /**
     * This operation requests the Data Distribution Service to mark the instance for
     * deletion.
     *
     * <b>Detailed Description</b><br>
     * This operation requests the Data Distribution Service to mark the instance for
     * deletion. Copies of the instance and its corresponding samples, which are stored in
     * every connected DataReader and, dependent on the QosPolicy settings, also in
     * the Transient and Persistent stores, will be marked for deletion by setting their
     * dds::sub::status::InstanceState to not_alive_disposed state.
     *
     * When this operation is used, the Data Distribution Service will automatically supply
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * @anchor anchor_dds_pub_datawriter_dispose_effect_readers
     * <i>Effects on DataReaders</i><br>
     * Actual deletion of the instance administration in a connected DataReader will be
     * postponed until the following conditions have been met:
     * - the instance must be unregistered (either implicitly or explicitly) by all connected
     *   DataWriters that have previously registered it.
     *      - A DataWriter can register an instance explicitly by using the special
     *        register_instance operations.
     *      - A DataWriter can register an instance implicitly by using no or the default (nil)
     *        InstanceHandle in any of the other DataWriter operations.
     *      - A DataWriter can unregister an instance explicitly by using one of the special
     *        unregister_instance operations.
     *      - A DataWriter will unregister all its contained instances implicitly when it is
     *        deleted.
     *      - When a DataReader detects a loss of liveliness in one of its connected
     *        DataWriters, it will consider all instances registered by that DataWriter as
     *        being implicitly unregistered.
     * - <i>and</i> the application must have consumed all samples belonging to the instance,
     *   either implicitly or explicitly.
     *      - An application can consume samples explicitly by invoking the take operation,
     *        or one of its variants, on its DataReaders.
     *      - The DataReader can consume disposed samples implicitly when the
     *        autopurge_disposed_samples_delay of the ReaderData
     *        Lifecycle QosPolicy has expired.
     *
     * The DataReader may also remove instances that haven’t been disposed first: this
     * happens when the autopurge_nowriter_samples_delay of the
     * ReaderDataLifecycle QosPolicy has expired after the instance is considered
     * unregistered by all connected DataWriters (i.e. when it has a
     * InstanceState of not_alive_no_writers.<br>
     * See also dds::core::policy::ReaderDataLifecycle QosPolicy.
     *
     * @anchor anchor_dds_pub_datawriter_dispose_effect_stores
     * <i>Effects on Transient/Persistent Stores</i><br>
     * Actual deletion of the instance administration in the connected Transient and
     * Persistent stores will be postponed until the following conditions have been met:
     * - the instance must be unregistered (either implicitly or explicitly) by all connected
     *   DataWriters that have previously registered it. (See above.)
     * - <i>and</i> the period of time specified by the service_cleanup_delay attribute in
     *   the DurabilityServiceQosPolicy on the Topic must have elapsed after the
     *   instance is considered unregistered by all connected DataWriters.
     *
     * See also dds::core::policy::Durability QosPolicy.
     *
     * <i>Instance Handle</i><br>
     * If handle is not nil, it must correspond to the value returned by
     * @ref dds::pub::DataWriter::register_instance(const T& key) "\"register_instance\"" or
     * @ref dds::pub::DataWriter::register_instance(const T& key, const dds::core::Time& timestamp) "\"register_instance_w_timestamp\""
     * when the instance (identified by its key) was registered. Passing such a registered
     * handle helps the Data Distribution Service to process the sample more efficiently.
     *
     * @anchor anchor_dds_pub_datawriter_dispose_blocking
     * <i>Blocking</i><br>
     * If the dds::core::policy::History QosPolicy is set to KEEP_ALL, the dispose
     * operation on the DataWriter may block if the modification would cause data to be
     * lost because one of the limits, specified in the dds::core::policy::ResourceLimits, is
     * exceeded. In case the synchronous attribute value of the
     * dds::core::policy::Reliability is set to TRUE for communicating DataWriters and
     * DataReaders then the DataWriter will wait until all synchronous
     * DataReaders have acknowledged the data. Under these circumstances, the
     * max_blocking_time attribute of the dds::core::policy::Reliability configures the
     * maximum time the dispose operation may block (either waiting for space to become
     * available or data to be acknowledged). If max_blocking_time elapses before the
     * DataWriter is able to store the modification without exceeding the limits and all
     * expected acknowledgements are received, the dispose operation will fail and throw
     * TimeoutError.
     *
     * @param i the instance to dispose
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    DataWriter& dispose_instance(const ::dds::core::InstanceHandle& i);

    /**
     * This operation requests the Data Distribution Service to mark the instance for
     * deletion and provides a value for the source_timestamp explicitly.
     *
     * <b>Detailed Description</b><br>
     * This operation requests the Data Distribution Service to mark the instance for
     * deletion. Copies of the instance and its corresponding samples, which are stored in
     * every connected DataReader and, dependent on the QosPolicy settings, also in
     * the Transient and Persistent stores, will be marked for deletion by setting their
     * dds::sub::status::InstanceState to not_alive_disposed state.
     *
     * When this operation is used, the application explicitly supplies
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Effects</i><br>
     * This operation @ref anchor_dds_pub_datawriter_dispose_effect_readers "effects DataReaders"
     * and @ref anchor_dds_pub_datawriter_dispose_effect_stores "effects Transient/Persistent Stores".
     *
     * <i>Instance Handle</i><br>
     * If handle is not nil, it must correspond to the value returned by
     * @ref dds::pub::DataWriter::register_instance(const T& key) "\"register_instance\"" or
     * @ref dds::pub::DataWriter::register_instance(const T& key, const dds::core::Time& timestamp) "\"register_instance_w_timestamp\""
     * when the instance (identified by its key) was registered. Passing such a registered
     * handle helps the Data Distribution Service to process the sample more efficiently.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_dispose_blocking "dispose blocking").
     *
     *
     * @param i the instance to dispose
     * @param timestamp the timestamp
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    DataWriter& dispose_instance(const ::dds::core::InstanceHandle& i,
                                 const dds::core::Time& timestamp);

    /**
     * This operation requests the Data Distribution Service to mark the instance for
     * deletion.
     *
     * <b>Detailed Description</b><br>
     * This operation requests the Data Distribution Service to mark the instance for
     * deletion. Copies of the instance and its corresponding samples, which are stored in
     * every connected DataReader and, dependent on the QosPolicy settings, also in
     * the Transient and Persistent stores, will be marked for deletion by setting their
     * dds::sub::status::InstanceState to not_alive_disposed state.
     *
     * When this operation is used, the Data Distribution Service will automatically supply
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Effects</i><br>
     * This operation @ref anchor_dds_pub_datawriter_dispose_effect_readers "effects DataReaders"
     * and @ref anchor_dds_pub_datawriter_dispose_effect_stores "effects Transient/Persistent Stores".
     *
     * <i>Instance</i><br>
     * The instance is identified by the key fields of the given typed data sample, instead
     * of an InstanceHandle.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_dispose_blocking "dispose blocking").
     *
     * @param key sample of the instance to dispose
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    DataWriter& dispose_instance(const T& key);

    /**
     * This operation requests the Data Distribution Service to mark the instance for
     * deletion and provides a value for the source_timestamp explicitly.
     *
     * <b>Detailed Description</b><br>
     * This operation requests the Data Distribution Service to mark the instance for
     * deletion. Copies of the instance and its corresponding samples, which are stored in
     * every connected DataReader and, dependent on the QosPolicy settings, also in
     * the Transient and Persistent stores, will be marked for deletion by setting their
     * dds::sub::status::InstanceState to not_alive_disposed state.
     *
     * When this operation is used, the application explicitly supplies
     * the value of the source_timestamp that is made available to connected
     * DataReader objects. This timestamp is important for the interpretation of the
     * dds::core::policy::DestinationOrder QosPolicy.
     *
     * As a side effect, this operation asserts liveliness on the DataWriter itself and on
     * the containing DomainParticipant.
     *
     * <i>Effects</i><br>
     * This operation @ref anchor_dds_pub_datawriter_dispose_effect_readers "effects DataReaders"
     * and @ref anchor_dds_pub_datawriter_dispose_effect_stores "effects Transient/Persistent Stores".
     *
     * <i>Instance</i><br>
     * The instance is identified by the key fields of the given typed data sample, instead
     * of an InstanceHandle.
     *
     * <i>Blocking</i><br>
     * This operation can be blocked (see @ref anchor_dds_pub_datawriter_dispose_blocking "dispose blocking").
     *
     * @param key sample of the instance to dispose
     * @param timestamp the timestamp
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     * @throws dds::core::TimeoutError
     *                  Either the current action overflowed the available resources
     *                  as specified by the combination of the Reliability QosPolicy,
     *                  History QosPolicy and ResourceLimits QosPolicy, or the current action
     *                  was waiting for data delivery acknowledgement by synchronous DataReaders.
     *                  This caused blocking of the write operation, which could not be resolved before
     *                  max_blocking_time of the Reliability QosPolicy elapsed.
     */
    DataWriter& dispose_instance(const T& key,
                                 const dds::core::Time& timestamp);

    /**
     * This operation retrieves the key value of a specific instance.
     *
     * This operation can be used to retrieve the instance key that corresponds
     * to an instance_handle. The operation will only fill the fields that form
     * the key inside the sample instance.
     *
     * This operation may raise a InvalidArgumentError exception if the InstanceHandle
     * does not correspond to an existing data-object known to the DataWriter.
     * If the implementation is not able to check invalid handles, then the
     * result in this situation is unspecified.
     *
     * The TopicInstance is added as parameter to be able to overload this operation.
     *
     * @param[out] i A topic instance to set the handle and sample key fields of
     * @param[in] h The instance handle
     * @return The given topic instance with the handle and key fields set
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::InvalidArgumentError
     *                  The InstanceHandle is not a valid handle.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     */
    dds::topic::TopicInstance<T>& key_value(dds::topic::TopicInstance<T>& i,
                                            const ::dds::core::InstanceHandle& h);

    /**
     * This operation retrieves the key value of a specific instance.
     *
     * This operation can be used to retrieve the instance key that corresponds
     * to an instance_handle. The operation will only fill the fields that form
     * the key inside the sample instance.
     *
     * This operation may raise a InvalidArgumentError exception if the InstanceHandle
     * does not correspond to an existing data-object known to the DataWriter.
     * If the implementation is not able to check invalid handles, then the
     * result in this situation is unspecified.
     *
     * The Sample is added as parameter to be able to overload this operation.
     *
     * @param[out] sample A sample to set the key fields of
     * @param[in] h The instance handle
     * @return The given sample with the key fields set
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::InvalidArgumentError
     *                  The InstanceHandle is not a valid handle.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NotEnabledError
     *                  The DataWriter has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataWriter.
     */
    T& key_value(T& sample, const ::dds::core::InstanceHandle& h);

    /**
     * This operation returns the value of the instance handle which corresponds
     * to the instance_data.
     *
     * The instance_data parameter is only used for the purpose of
     * examining the fields that define the key. The instance handle can be used in any
     * write, dispose or unregister operations (or their time stamped variants) that
     * operate on a specific instance. Note that DataWriter instance handles are local,
     * and are not interchangeable with DataReader instance handles nor with instance
     * handles of an other DataWriter.
     *
     * This operation does not register the instance in question. If the instance has not been
     * previously registered or if for any other
     * reason the Service is unable to provide an instance handle, the Service will return
     * the default nil handle (InstanceHandle.is_nil() == true).
     *
     * @param key the sample
     * @return the instance handle
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     */
    dds::core::InstanceHandle lookup_instance(const T& key);

    //==========================================================================
    //== QoS Management

    /** @copydoc dds::pub::TAnyDataWriter::qos(const dds::pub::qos::DataWriterQos& qos) */
    DataWriter& operator <<(const ::dds::pub::qos::DataWriterQos& qos);


    //==========================================================================
    //== Entity Navigation

    /**
     * Get the Topic associated with this DataWriter.
     *
     * @return the Topic
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     */
    const dds::topic::Topic<T>& topic() const;

    //==========================================================================
    //== Listeners Management

    /**
     * Register a listener with the DataWriter.
     *
     * This operation attaches a DataWriterListener to the DataWriter. Only one
     * DataWriterListener can be attached to each DataWriter. If a
     * DataWriterListener was already attached, the operation will replace it with the
     * new one. When the listener is the NULL pointer, it represents a listener that is
     * treated as a NOOP for all statuses activated in the bit mask.
     *
     * Listener un-registration is performed by setting the listener to NULL and mask none().
     *
     * @anchor anchor_dds_pub_datawriter_commstatus
     * <i>Communication Status</i><br>
     * For each communication status, the StatusChangedFlag flag is initially set to
     * FALSE. It becomes TRUE whenever that communication status changes. For each
     * communication status activated in the mask, the associated DataWriterListener
     * operation is invoked and the communication status is reset to FALSE, as the listener
     * implicitly accesses the status which is passed as a parameter to that operation. The
     * status is reset prior to calling the listener, so if the application calls the
     * get_<status_name>_status from inside the listener it will see the status
     * already reset. An exception to this rule is the NULL listener, which does not reset the
     * communication statuses for which it is invoked.
     *
     * The following statuses are applicable to the DataWriterListener:
     *  - dds::core::status::StatusMask::offered_deadline_missed()
     *  - dds::core::status::StatusMask::offered_incompatible_qos()
     *  - dds::core::status::StatusMask::liveliness_lost()
     *  - dds::core::status::StatusMask::publication_matched()
     *
     * Be aware that the PUBLICATION_MATCHED_STATUS is not applicable when the
     * infrastructure does not have the information available to determine connectivity.
     * This is the case when OpenSplice is configured not to maintain discovery
     * information in the Networking Service. (See the description for the
     * NetworkingService/Discovery/enabled property in the Deployment
     * Manual for more information about this subject.) In this case the operation will
     * throw UnsupportedError.
     *
     * Status bits are declared as a constant and can be used by the application in an OR
     * operation to create a tailored mask. The special constant dds::core::status::StatusMask::none()
     * can be used to indicate that the created entity should not respond to any of its available
     * statuses. The DDS will therefore attempt to propagate these statuses to its factory.
     * The special constant dds::core::status::StatusMask::all() can be used to select all applicable
     * statuses specified in the “Data Distribution Service for Real-time Systems Version
     * 1.2” specification which are applicable to the PublisherListener.
     *
     * @anchor anchor_dds_pub_datawriter_commpropagation
     * <i>Status Propagation</i><br>
     * In case a communication status is not activated in the mask of the
     * DataWriterListener, the PublisherListener of the containing Publisher
     * is invoked (if attached and activated for the status that occurred). This allows the
     * application to set a default behaviour in the PublisherListener of the containing
     * Publisher and a DataWriter specific behaviour when needed. In case the
     * communication status is not activated in the mask of the PublisherListener as
     * well, the communication status will be propagated to the
     * DomainParticipantListener of the containing DomainParticipant. In case
     * the DomainParticipantListener is also not attached or the communication
     * status is not activated in its mask, the application is not notified of the change.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @param listener  the listener
     * @param mask      the mask defining the events for which the listener
     *                  will be notified.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     * @throws dds::core::AlreadyClosedError
     *                  The entity has already been closed.
     * @throws dds::core::UnsupportedError
     *                  A status was selected that cannot be supported because
     *                  the infrastructure does not maintain the required connectivity information.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    void listener(DataWriterListener<T>* listener,
                  const ::dds::core::status::StatusMask& mask);

    /**
     * Get the listener of this DataWriter.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @return the listener
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     */
    DataWriterListener<T>* listener() const;
};

#endif /* OMG_DDS_PUB_DATA_WRITER_HPP_ */
