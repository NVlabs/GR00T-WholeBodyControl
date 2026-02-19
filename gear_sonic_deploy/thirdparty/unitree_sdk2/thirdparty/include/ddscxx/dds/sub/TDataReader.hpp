#ifndef OMG_DDS_SUB_TDATA_READER_HPP_
#define OMG_DDS_SUB_TDATA_READER_HPP_

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
#include <dds/core/detail/conformance.hpp>
#include <dds/sub/AnyDataReader.hpp>
#include <dds/topic/ContentFilteredTopic.hpp>
#include <dds/topic/TopicInstance.hpp>
#include <dds/sub/LoanedSamples.hpp>
#include <dds/sub/Subscriber.hpp>


namespace dds
{
namespace sub
{
template <typename T, template <typename Q> class DELEGATE>
class DataReader;

template <typename T>
class DataReaderListener;
}
}

/**
 * @brief
 * DataReader allows the applicatin to access published sample data.
 *
 * A DataReader allows the application:
 * - to declare the data it wishes to receive (i.e., make a subscription)
 * - to access the data received by the attached Subscriber
 *
 * A DataReader refers to exactly one TopicDescription (either a Topic, a
 * ContentFilteredTopic or a MultiTopic) that identifies the samples to be
 * read. The Topic must exist prior to the DataReader creation.
 *
 * A DataReader is attached to exactly one Subscriber which acts as a factory
 * for it.
 *
 * The DataReader may give access to several instances of the data type, which
 * are distinguished from each other by their key.
 *
 * The pre-processor generates from IDL type descriptions the application
 * DataReader<type> classes. For each application data type that is used as Topic
 * data type, a typed class DataReader<type> is derived from the AnyDataReader
 * class.
 *
 * For instance, for an application, the definitions are located in the Foo.idl file.
 * The pre-processor will generate a ccpp_Foo.h include file.
 *
 * <b>General note:</b> The name ccpp_Foo.h is derived from the IDL file Foo.idl,
 * that defines Foo::Bar, for all relevant DataReader<Foo::Bar> operations.
 *
 * @note Apart from idl files, Google protocol buffers are also supported. For the
 *       API itself, it doesn't matter if the type header files were generated from
 *       idl or protocol buffers. The resulting API usage and includes remain the same.
 *
 * @anchor anchor_dds_sub_datareader_example
 * <b>Example</b>
 * @code{.cpp}
 * // Default creation of a DataReader
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
 * dds::sub::Subscriber subscriber(participant);
 * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
 *
 * {
 *     // Default read of a sample on the DataReader
 *     dds::sub::LoanedSamples<Foo::Bar> samples;
 *     samples = reader.read();
 *
 *     // Default way of accessing the loaned samples
 *     dds::sub::LoanedSamples<Foo::Bar>::const_iterator it;
 *     for (it = samples.begin(); it != samples.end(); ++it) {
 *         const dds::sub::Sample<Foo::Bar>& sample = *it;
 *         const Foo::Bar& data = sample.data();
 *         const dds::sub::SampleInfo& info = sample.info();
 *         // Use sample data and meta information.
 *     }
 * }
 * // Like the name says, the read samples are loans from the DataReader. The loan
 * // was automatically returned when the LoanedSamples went out of scope.
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Subscription "Subscription concept"
 * @see for more information: @ref DCPS_Modules_Subscription_DataReader "DataReader concept"
 */
template <typename T, template <typename Q> class DELEGATE>
class dds::sub::DataReader : public dds::sub::TAnyDataReader< DELEGATE<T> >
{

public:
    /**
     * Local convenience typedef for dds::sub::DataReaderListener.
     */
    typedef ::dds::sub::DataReaderListener<T> Listener;

public:

    /**
     * The Selector class is used by the DataReader to compose read operations.
     *
     * A Selector can perform complex data selections, such as per-instance selection,
     * content and status filtering, etc, when reading or taking samples. These settings
     * on a Selector can be concatenated.
     *
     * The DataReader has the select() operation, which can be used to aqcuire the Selector
     * functionality on the reader implicitly.
     * @code{.cpp}
     * // Take a maximum of 3 new samples of a certain instance.
     * samples = reader.select()
     *                     .max_samples(3)
     *                     .state(dds::sub::status::DataState::new_data())
     *                     .instance(someValidInstanceHandle)
     *                     .take();
     * @endcode
     * However, this will create and destroy a Selector for every read, which is not
     * very performance friendly.
     *
     * The performance can be increase by creating a Selector up front and doing the
     * reading on that Selector directly and re-using it.
     * @code{.cpp}
     * // Create a Selector as selective reader up front.
     * dds::sub::DataReader<Foo::Bar>::Selector selectiveReader(reader);
     * // Configure it to take a maximum of 3 new samples of a certain instance
     * selectiveReader.max_samples(3);
     * selectiveReader.state(dds::sub::status::DataState::new_data());
     * selectiveReader.instance(someValidInstanceHandle);
     *
     * // Use the configured Selector to -take- a maximum of 3 new samples of a
     * // certain instance (which it was configured to do).
     * // This can be used in loops for example, reducing the need for creating
     * // implicit Selectors for every take.
     * samples = selectiveReader.take();
     * @endcode
     *
     * <i>Defaults</i>
     * Element         | Default Value
     * --------------- | --------------------
     * state           | dds::sub::status::DataState::any
     * content         | Empty dds::sub::Query
     * max_samples     | dds::core::LENGTH_UNLIMITED
     * instance        | dds::core::InstanceHandle nil
     *
     * @see for more information: @link dds::sub::DataReader::select() DataReader select() @endlink
     */
    class Selector
    {
    public:
        /**
         * Construct a Selector for a DataReader.
         *
         * @param dr DataReader
         */
        Selector(DataReader& dr);

        /**
         * Set InstanceHandle to filter with during the read or take.
         *
         * <i>Example</i><br>
         * Read only samples of the given instance.
         * @code{.cpp}
         * dds::core::InstanceHandle hdl = someValidInstanceHandle;
         *
         * // Implicit use of Selector
         * samples = reader.select().instance(hdl).read();
         *
         * // Explicit use of Selector
         * dds::sub::DataReader<Foo::Bar>::Selector selectiveReader(reader);
         * selectiveReader.instance(hdl);
         * samples = selectiveReader.read();
         * @endcode
         * See also @link dds::sub::DataReader::select() DataReader select() @endlink operation.
         *
         * @param handle the InstanceHandle to read/take for
         * @return a Selector to be able to concatenate Selector settings.
         */
        Selector& instance(const dds::core::InstanceHandle& handle);

        /**
         * Set next InstanceHandle to filter with during the read or take.
         *
         * <i>Example</i><br>
         * Read all samples, instance by instance.
         * @code{.cpp}
         * // Implicit use of Selector
         * {
         *     // Get sample(s) of first instance
         *     dds::core::InstanceHandle hdl; //nil
         *     samples = reader.select().next_instance(hdl).read();
         *     while (samples.length() > 0) {
         *         // Handle the sample(s) of this instance (just the first one in this case)
         *         const dds::sub::Sample<Foo::Bar>& sample = *(samples.begin());
         *         // Get sample(s) of the next instance
         *         hdl = sample.info().instance_handle();
         *         samples = reader.select().next_instance(hdl).read();
         *     }
         * }
         *
         * // Explicit use of Selector
         * {
         *     // Get sample(s) of first instance
         *     dds::sub::DataReader<Foo::Bar>::Selector selectiveReader(reader);
         *     dds::core::InstanceHandle hdl; //nil
         *     selectiveReader.next_instance(hdl);
         *     samples = selectiveReader.read();
         *     while (samples.length() > 0) {
         *         // Handle the sample(s) of this instance (just the first one in this case)
         *         const dds::sub::Sample<Foo::Bar>& sample = *(samples.begin());
         *         // Get sample(s) of the next instance
         *         hdl = sample.info().instance_handle();
         *         selectiveReader.next_instance(hdl);
         *         samples = selectiveReader.read();
         *     }
         * }
         * @endcode
         * See also @link dds::sub::DataReader::select() DataReader select() @endlink operation.
         *
         * @param handle the 'previous' InstanceHandle associated with new the read/take
         * @return a Selector to be able to concatenate Selector settings.
         */
        Selector& next_instance(const dds::core::InstanceHandle& handle);

        /**
         * Set DataState to filter with during the read or take.
         *
         * <i>Example</i><br>
         * Read only new data.
         * @code{.cpp}
         * // DataState to filter only new data
         * dds::sub::status::DataState newData = dds::sub::status::DataState::new_data();
         *
         * // Implicit use of Selector
         * samples = reader.select().state(newData).read();
         *
         * // Explicit use of Selector
         * dds::sub::DataReader<Foo::Bar>::Selector selectiveReader(reader);
         * selectiveReader.state(newData);
         * samples = selectiveReader.read();
         * @endcode
         * See also @link dds::sub::DataReader::select() DataReader select() @endlink operation.
         *
         * @param state the requested DataState of the samples
         * @return a Selector to be able to concatenate Selector settings.
         */
        Selector& state(const dds::sub::status::DataState& state);

        /**
         * Set the Query to filter with during the read or take.
         *
         * <i>Example</i><br>
         * Read only samples that will be filtered according to the given dds::sub::Query.
         * @code{.cpp}
         * // Assume data type has an element called long_1
         * dds::sub::Query query(reader, "long_1 > 1 and long_1 < 7");
         *
         * // Implicit use of Selector
         * samples = reader.select().content(query).read();
         *
         * // Explicit use of Selector
         * dds::sub::DataReader<Foo::Bar>::Selector selectiveReader(reader);
         * selectiveReader.content(query);
         * samples = selectiveReader.read();
         * @endcode
         * See also @link dds::sub::DataReader::select() DataReader select() @endlink operation.
         *
         * @param query the Query to apply to the selector
         * @return a Selector to be able to concatenate Selector settings.
         */
        Selector& content(const dds::sub::Query& query);

        /**
         * Set max_samples to limit the number of sample to get during the read or take.
         *
         * <i>Example</i><br>
         * Read a maximum of three samples.
         * @code{.cpp}
         * // Implicit use of Selector
         * samples = reader.select().max_samples(3).read();
         *
         * // Explicit use of Selector
         * dds::sub::DataReader<Foo::Bar>::Selector selectiveReader(reader);
         * selectiveReader.max_samples(3);
         * samples = selectiveReader.read();
         * @endcode
         * See also @link dds::sub::DataReader::select() DataReader select() @endlink operation.
         *
         * @param maxsamples maximum number of samples to read/take
         * @return a Selector to be able to concatenate Selector settings.
         */
        Selector& max_samples(uint32_t maxsamples);

        /**
         * This operation works the same as the @link DataReader::read() default
         * DataReader read() @endlink, except that it is performed on this Selector
         * with possible filters set.
         *
         * @return          The samples in the LoanedSamples container
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
         *                  The DataReader has not yet been enabled.
         */
        dds::sub::LoanedSamples<T> read();

        /**
         * This operation works the same as the @link DataReader::take() default
         * DataReader take() @endlink, except that it is performed on this Selector
         * with possible filters set.
         *
         * @return          The samples in the LoanedSamples container
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
         *                  The DataReader has not yet been enabled.
         */
        dds::sub::LoanedSamples<T> take();

        // --- Forward Iterators: --- //
        /**
         * This operation works the same as the @link DataReader::read(SamplesFWIterator sfit, uint32_t max_samples)
         * forward iterator DataReader read() @endlink, except that it is performed on this
         * Selector with possible filters set.
         *
         * @param  sfit     Forward-inserting container iterator
         * @param  max_samples Maximum samples to read and copy into the given container
         * @return          The number of samples in the LoanedSamples container
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
         *                  The DataReader has not yet been enabled.
         */
        template <typename SamplesFWIterator>
        uint32_t
        read(SamplesFWIterator sfit, uint32_t max_samples);

        /**
         * This operation works the same as the @link DataReader::take(SamplesFWIterator sfit, uint32_t max_samples)
         * forward iterator DataReader take() @endlink, except that it is performed on this
         * Selector with possible filters set.
         *
         * @param  sfit     Forward-inserting container iterator
         * @param  max_samples Maximum samples to read and copy into the given container
         * @return          The number of samples in the given container
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
         *                  The DataReader has not yet been enabled.
         */
        template <typename SamplesFWIterator>
        uint32_t
        take(SamplesFWIterator sfit,  uint32_t max_samples);

        // --- Back-Inserting Iterators: --- //
        /**
         * This operation works the same as the @link DataReader::read(SamplesBIIterator sbit)
         * backward iterator DataReader read() @endlink, except that it is performed on this
         * Selector with possible filters set.
         *
         * @param  sbit     Back-inserting container iterator
         * @return          The number of samples in the given container
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
         *                  The DataReader has not yet been enabled.
         */
        template <typename SamplesBIIterator>
        uint32_t
        read(SamplesBIIterator sbit);

        /**
         * This operation works the same as the @link DataReader::take(SamplesBIIterator sbit)
         * backward iterator DataReader take() @endlink, except that it is performed on this
         * Selector with possible filters set.
         *
         * @param  sbit     Back-inserting container iterator
         * @return          The number of samples in the given container
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
         *                  The DataReader has not yet been enabled.
         */
        template <typename SamplesBIIterator>
        uint32_t
        take(SamplesBIIterator sbit);

    private:
        typename DELEGATE<T>::Selector impl_;
    };

    /**
     * The ManipulatorSelector class is used by the DataReader to compose streaming
     * read operations.
     *
     * A ManipulatorSelector can perform complex data selections, such as per-instance
     * selection, content and status filtering, etc, when reading or taking samples
     * through the streaming operator.
     *
     * <i>Convenience functors</i><br>
     * The following convenience functors use a ManipulatorSelector implicitly and can be
     * used in the streaming operator:
     * - dds::sub::read
     * - dds::sub::take
     * - dds::sub::max_samples
     * - dds::sub::content
     * - dds::sub::state
     * - dds::sub::instance
     * - dds::sub::next_instance
     *
     * @code{.cpp}
     * // Take a maximum of 3 new samples of a certain instance.
     * reader >> dds::sub::take
     *        >> dds::sub::max_samples(3)
     *        >> dds::sub::state(dds::sub::status::DataState::new_data())
     *        >> dds::sub::instance(someValidInstanceHandle)
     *        >> samples;
     * @endcode
     * However, this will create and destroy ManipulatorSelectors and Functors for
     * every read, which is not very performance friendly.
     *
     * The performance can be increase by creating a ManipulatorSelector up front and
     * doing the reading on that ManipulatorSelector directly and re-using it.
     * @code{.cpp}
     * // Create a ManipulatorSelector as selective reader up front.
     * dds::sub::DataReader<Foo::Bar>::ManipulatorSelector selectiveReader(reader);
     * // Configure it to take a maximum of 3 new samples of a certain instance
     * selectiveReader.max_samples(3);
     * selectiveReader.state(dds::sub::status::DataState::new_data());
     * selectiveReader.instance(someValidInstanceHandle);
     * selectiveReader.read_mode(false); // take
     *
     * // Use the configured ManipulatorSelector to -take- a maximum of 3 samples of a
     * // certain instance (which it was configured to do).
     * // This can be used in loops for example, reducing the need for creating
     * // implicit ManipulatorSelectors for every take.
     * selectiveReader >> samples;
     * @endcode
     *
     * <i>Defaults</i>
     * Element         | Default Value
     * --------------- | --------------------
     * read_mode       | true (read)
     * state           | dds::sub::status::DataState::any
     * content         | Empty dds::sub::Query
     * max_samples     | dds::core::LENGTH_UNLIMITED
     * instance        | dds::core::InstanceHandle nil
     *
     * @see for more information: @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls) DataReader stream operator>> @endlink
     */
    class ManipulatorSelector
    {
    public:
        /**
         * Construct a ManipulatorSelector for a DataReader.
         *
         * See also @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink
         *
         * @param DataReader
         */
        ManipulatorSelector(DataReader& dr);

        /**
         * Get the read_mode.
         *
         * The read_mode specifies if a sample should be read or taken:
         * - true = read (default)
         * - false = take
         *
         * @return true if read_mode is set to read
         */
        bool read_mode();

        /**
         * Set the read_mode.
         *
         * The read_mode specifies if a sample should be read or taken:
         * - true = read (default)
         * - false = take
         *
         * <i>Convenience Functor:</i> dds::sub::read<br>
         * <i>Convenience Functor:</i> dds::sub::take
         *
         * <i>Example</i><br>
         * Determine to read or take samples.
         * @code{.cpp}
         * // No usage of ManipulatorSelector, means a read iso take as default.
         * reader >> samples;
         *
         * // Implicit use of ManipulatorSelector
         * reader >> dds::sub::read >> samples;
         * reader >> dds::sub::take >> samples;
         *
         * // Explicit use of ManipulatorSelector
         * dds::sub::DataReader<Foo::Bar>::ManipulatorSelector readingReader(reader);
         * readingReader.read_mode(true); // Read, which is already the default.
         * readingReader >> samples;
         *
         * dds::sub::DataReader<Foo::Bar>::ManipulatorSelector takingReader(reader);
         * takingReader.read_mode(false); // Take.
         * takingReader >> samples;
         * @endcode
         * See also @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink
         *
         * @param readmode the read mode of the DataReader
         */
        void read_mode(bool readmode);

        /**
         * Set max_samples to limit the number of sample to get during the read or take.
         *
         * <i>Convenience Functor:</i> dds::sub::max_samples
         *
         * <i>Example</i><br>
         * Read a maximum of three samples.
         * @code{.cpp}
         * // Implicit use of ManipulatorSelector
         * reader >> dds::sub::max_samples(3) >> samples;
         *
         * // Explicit use of ManipulatorSelector
         * dds::sub::DataReader<Foo::Bar>::ManipulatorSelector selectiveReader(reader);
         * selectiveReader.max_samples(3);
         * selectiveReader >> samples;
         * @endcode
         * See also @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink
         *
         * @param n maximum number of samples
         */
        ManipulatorSelector& max_samples(uint32_t n);

        /**
         * Set InstanceHandle to filter with during the read or take.
         *
         * <i>Convenience Functor:</i> dds::sub::instance
         *
         * <i>Example</i><br>
         * Read only samples of the given instance.
         * @code{.cpp}
         * dds::core::InstanceHandle hdl = someValidInstanceHandle;
         *
         * // Implicit use of ManipulatorSelector
         * reader >> dds::sub::instance(hdl) >> samples;
         *
         * // Explicit use of ManipulatorSelector
         * dds::sub::DataReader<Foo::Bar>::ManipulatorSelector selectiveReader(reader);
         * selectiveReader.instance(hdl);
         * selectiveReader >> samples;
         * @endcode
         * See also @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink
         *
         * @param handle the InstanceHandle for the read/take
         */
        ManipulatorSelector& instance(const dds::core::InstanceHandle& handle);

        /**
         * Set next InstanceHandle to filter with during the read or take.
         *
         * <i>Convenience Functor:</i> dds::sub::next_instance
         *
         * <i>Example</i><br>
         * Read all samples, instance by instance.
         * @code{.cpp}
         * // Implicit use of ManipulatorSelector
         * {
         *     // Get sample(s) of first instance
         *     dds::core::InstanceHandle hdl; //nil
         *     reader >> dds::sub::next_instance(hdl) >> samples;
         *     while (samples.length() > 0) {
         *         // Handle the sample(s) of this instance (just the first one in this case)
         *         const dds::sub::Sample<Foo::Bar>& sample = *(samples.begin());
         *         // Get sample(s) of the next instance
         *         hdl = sample.info().instance_handle();
         *         reader >> dds::sub::next_instance(hdl) >> samples;
         *     }
         * }
         *
         * // Explicit use of ManipulatorSelector
         * {
         *     // Get sample(s) of first instance
         *     dds::sub::DataReader<Foo::Bar>::ManipulatorSelector selectiveReader(reader);
         *     dds::core::InstanceHandle hdl; //nil
         *     selectiveReader.next_instance(hdl);
         *     selectiveReader >> samples;
         *     while (samples.length() > 0) {
         *         // Handle the sample(s) of this instance (just the first one in this case)
         *         const dds::sub::Sample<Foo::Bar>& sample = *(samples.begin());
         *         // Get sample(s) of the next instance
         *         hdl = sample.info().instance_handle();
         *         selectiveReader.next_instance(hdl);
         *         selectiveReader >> samples;
         *     }
         * }
         * @endcode
         * See also @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink
         *
         * @param handle the 'previous' InstanceHandle associated with new the read/take
         */
        ManipulatorSelector& next_instance(const dds::core::InstanceHandle& handle);

        /**
         * Set DataState to filter with during the read or take.
         *
         * <i>Convenience Functor:</i> dds::sub::state
         *
         * <i>Example</i><br>
         * Read only new data.
         * @code{.cpp}
         * // DataState to filter only new data
         * dds::sub::status::DataState newData = dds::sub::status::DataState::new_data();
         *
         * // Implicit use of ManipulatorSelector
         * reader >> dds::sub::state(newData) >> samples;
         *
         * // Explicit use of ManipulatorSelector
         * dds::sub::DataReader<Foo::Bar>::ManipulatorSelector selectiveReader(reader);
         * selectiveReader.state(newData);
         * selectiveReader.read() >> samples;
         * @endcode
         * See also @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink
         *
         * @param state the required DataState of the samples
         */
        ManipulatorSelector& state(const dds::sub::status::DataState& state);

        /**
         * Set Query to filter with during the read or take.
         *
         * <i>Convenience Functor:</i> dds::sub::content
         *
         * <i>Example</i><br>
         * Read only samples that will be filtered according to the given dds::sub::Query.
         * @code{.cpp}
         * // Assume data type has an element called long_1
         * dds::sub::Query query(reader, "long_1 > 1 and long_1 < 7");
         *
         * // Implicit use of ManipulatorSelector
         * reader >> dds::sub::content(query) >> samples;
         *
         * // Explicit use of ManipulatorSelector
         * dds::sub::DataReader<Foo::Bar>::ManipulatorSelector selectiveReader(reader);
         * selectiveReader.content(query);
         * selectiveReader >> read;
         * @endcode
         * See also @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink
         *
         * @param query The Query to apply to a read/take
         */
        ManipulatorSelector& content(const dds::sub::Query& query);

        /**
         * This operation works the same as the @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink, except that it is performed on this ManipulatorSelector
         * with possible filters set.
         *
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
         *                  The DataReader has not yet been enabled.
         */
        ManipulatorSelector&
        operator >>(dds::sub::LoanedSamples<T>& samples);

        /**
         * This operation works the same as the @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink, except that it is performed on this ManipulatorSelector
         * with possible filters set.
         *
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
         *                  The DataReader has not yet been enabled.
         */
        ManipulatorSelector&
        operator >> (ManipulatorSelector & (manipulator)(ManipulatorSelector&));

        /**
         * This operation works the same as the @link dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls)
         * DataReader stream operator>> @endlink, except that it is performed on this ManipulatorSelector
         * with possible filters set.
         *
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
         *                  The DataReader has not yet been enabled.
         */
        template <typename Functor>
        ManipulatorSelector
        operator >> (Functor f);

    private:
        typename DELEGATE<T>::ManipulatorSelector impl_;

    };

public:
    OMG_DDS_REF_TYPE_PROTECTED_DC_T(DataReader, dds::sub::TAnyDataReader, T, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(DataReader)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_DEFAULT(DataReader)

public:
    /**
     * Create a new DataReader for the desired Topic, ContentFilteredTopic or MultiTopic,
     * using the given Subscriber.
     *
     * <i>QoS</i><br>
     * The DataReader will be created with the QoS values specified on the last
     * successful call to @link dds::sub::Subscriber::default_datareader_qos(const dds::sub::qos::DataReaderQos& qos)
     * sub.default_datareader_qos(qos) @endlink or, if the call was never made,
     * the @ref anchor_dds_sub_datareader_qos_defaults "default" values.
     *
     * @param sub       the Subscriber that will contain this DataReader
     * @param topic     the Topic associated with this DataReader
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     */
    DataReader(const dds::sub::Subscriber& sub,
               const ::dds::topic::Topic<T>& topic);

    /**
     * Create a new DataReader for the desired Topic, ContentFilteredTopic or MultiTopic,
     * using the given Subscriber and DataReaderQos and attaches the optionally specified
     * DataReaderListener to it.
     *
     * <i>QoS</i><br>
     * A possible application pattern to construct the DataReaderQos for the
     * DataReader is to:
     * @code{.cpp}
     * // 1) Retrieve the QosPolicy settings on the associated Topic
     * dds::topic::qos::TopicQos topicQos = topic.qos();
     * // 2) Retrieve the default DataReaderQos from the related Subscriber
     * dds::sub::qos::DataReaderQos readerQos = subscriber.default_datareader_qos();
     * // 3) Combine those two lists of QosPolicy settings by overwriting DataReaderQos
     * //    policies that are also present TopicQos
     * readerQos = topicQos;
     * // 4) Selectively modify QosPolicy settings as desired.
     * readerQos << dds::core::policy::Durability::Transient();
     * // 5) Use the resulting QoS to construct the DataReader.
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic, readerQos);
     * @endcode
     *
     * <i>Listener</i><br>
     * The following statuses are applicable to the DataReaderListener:
     *  - dds::core::status::StatusMask::requested_deadline_missed()
     *  - dds::core::status::StatusMask::requested_incompatible_qos()
     *  - dds::core::status::StatusMask::sample_lost()
     *  - dds::core::status::StatusMask::sample_rejected()
     *  - dds::core::status::StatusMask::data_available()
     *  - dds::core::status::StatusMask::liveliness_changed()
     *  - dds::core::status::StatusMask::subscription_matched()
     *
     * See @ref DCPS_Modules_Infrastructure_Listener "listener concept",
     * @ref anchor_dds_sub_datareader_commstatus "communication status" and
     * @ref anchor_dds_sub_datareader_commpropagation "communication propagation"
     * for more information.
     *
     * @param sub       the Subscriber that will contain this DataReader
     * @param topic     the Topic, ContentFilteredTopic or MultiTopic associated with this DataReader
     * @param qos       the DataReader qos.
     * @param listener  the DataReader listener.
     * @param mask      the listener event mask.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::InconsistentPolicyError
     *                  The parameter qos contains conflicting QosPolicy settings.
     */
    DataReader(const dds::sub::Subscriber& sub,
               const ::dds::topic::Topic<T>& topic,
               const dds::sub::qos::DataReaderQos& qos,
               dds::sub::DataReaderListener<T>* listener = NULL,
               const dds::core::status::StatusMask& mask = ::dds::core::status::StatusMask::none());

    #ifdef OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT

    /** @copydoc dds::sub::DataReader::DataReader(const dds::sub::Subscriber& sub, const ::dds::topic::Topic<T>& topic) */
    DataReader(const dds::sub::Subscriber& sub,
               const ::dds::topic::ContentFilteredTopic<T>& topic);

    /** @copydoc dds::sub::DataReader::DataReader(const dds::sub::Subscriber& sub, const ::dds::topic::Topic<T>& topic, const dds::sub::qos::DataReaderQos& qos, dds::sub::DataReaderListener<T>* listener, const dds::core::status::StatusMask& mask) */
    DataReader(const dds::sub::Subscriber& sub,
               const ::dds::topic::ContentFilteredTopic<T>& topic,
               const dds::sub::qos::DataReaderQos& qos,
               dds::sub::DataReaderListener<T>* listener = NULL,
               const dds::core::status::StatusMask& mask = ::dds::core::status::StatusMask::none());
    #endif /* OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT */

    #ifdef OMG_DDS_MULTI_TOPIC_SUPPORT

    /** @copydoc dds::sub::DataReader::DataReader(const dds::sub::Subscriber& sub, const ::dds::topic::Topic<T>& topic) */
    DataReader(const dds::sub::Subscriber& sub,
               const ::dds::topic::MultiTopic<T>& topic);

    /** @copydoc dds::sub::DataReader::DataReader(const dds::sub::Subscriber& sub, const ::dds::topic::Topic<T>& topic, const dds::sub::qos::DataReaderQos& qos, dds::sub::DataReaderListener<T>* listener, const dds::core::status::StatusMask& mask) */
    DataReader(const dds::sub::Subscriber& sub,
               const ::dds::topic::MultiTopic<T>& topic,
               const dds::sub::qos::DataReaderQos& qos,
               dds::sub::DataReaderListener<T>* listener = NULL,
               const dds::core::status::StatusMask& mask = ::dds::core::status::StatusMask::none());

    #endif /* OMG_DDS_MULTI_TOPIC_SUPPORT */

public:
    // == ReadState Management

    /**
     * Returns the @ref anchor_dds_sub_datareader_defaultstatefilter "default state filter"
     * for read/take operations.
     *
     * The default value of default_filter_state is dds::sub::status::DataState::any().
     *
     * @return the default state to filter for
     */
     dds::sub::status::DataState default_filter_state();

    /**
     * Set the default state filter for read/take operations.
     *
     * @anchor anchor_dds_sub_datareader_defaultstatefilter
     * <i>Default State Filter</i><br>
     * The default_filter_state indicates what the dds::sub::status::DataState of samples
     * should be for the read to filter them out of the total data samples pool.
     *
     * This filter can be overruled by using dds::sub::DataReader::Selector::state or
     * dds::sub::DataReader::ManipulatorSelector::state during the actual read action.
     *
     * @param state the state mask that will be used to read/take samples
     */
    DataReader& default_filter_state(const dds::sub::status::DataState& state);

    //== Streaming read/take

    /**
     * This operation reads a sequence of typed samples from the DataReader by means
     * of the shift operator.
     *
     * What samples are read depends on what @link dds::sub::DataReader::default_filter_state(const dds::sub::status::DataState& state)
     * default state filter @endlink has been set (default any).
     *
     * This operation reads a sequence of typed samples from the DataReader<type>. The
     * data is put into a dds::sub::LoanedSamples, which is basically a sequence of samples,
     * which in turn contains the actual data and meta information.
     *
     * The memory used for storing the sample may be loaned by the middleware thus allowing zero
     * copy operations.
     *
     * If the DataReader has no samples that meet the constraints, the resulting
     * dds::sub::LoanedSamples will be empty.
     * @code{.cpp}
     * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
     * dds::sub::Subscriber subscriber(participant);
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
     *
     * {
     *     dds::sub::LoanedSamples<Foo::Bar> samples;
     *     reader >> samples;
     * }
     * // The LoanedSamples went out of scope, meaning the loan was returned.
     * @endcode
     *
     * <b><i>Take</i></b><br>
     * The default behaviour of the DataReader stream operator>> is to read samples. It can be
     * explicitly stated if the operator should do a read or take.
     * @code{.cpp}
     * reader >> dds::sub::read >> samples;
     * reader >> dds::sub::take >> samples;
     * @endcode
     * These are two of the available convenience manipulators (see next paragraph).
     *
     * <b><i>Manipulators</i></b><br>
     * Manipulators are defined externally to make it possible to control which data is read
     * and whether the streaming operators reads or takes.
     *
     * Available convenience stream manipulators:
     * - dds::sub::read
     * - dds::sub::take
     * - dds::sub::max_samples
     * - dds::sub::content
     * - dds::sub::state
     * - dds::sub::instance
     * - dds::sub::next_instance
     *
     * The manipulators can be concatenated:
     * @code{.cpp}
     * // Take (iso read) a maximum of 3 new samples of a certain instance.
     * reader >> dds::sub::take
     *        >> dds::sub::max_samples(3)
     *        >> dds::sub::state(dds::sub::status::DataState::new_data())
     *        >> dds::sub::instance(someValidInstanceHandle)
     *        >> samples;
     * @endcode
     *
     * <i>Please be aware that using pre-set filters on the DataReader and then
     * call read() or take() on that reader explicitly will perform better than
     * having to create Manipulators for every read (which is what essentially
     * happens in the code example above). Performance can be increase by creating
     * a manipulator up front and using that multiple times
     * (see dds::sub::DataReader::ManipulatorSelector).</i>
     *
     * <b><i>Invalid Data</i></b><br>
     * Some elements in the returned sequence may not have valid data: the valid_data
     * field in the SampleInfo indicates whether the corresponding data value contains
     * any meaningful data. If not, the data value is just a ‘dummy’ sample for which only
     * the keyfields have been assigned. It is used to accompany the SampleInfo that
     * communicates a change in the instance_state of an instance for which there is
     * no ‘real’ sample available.
     *
     * For example, when an application always ‘takes’ all available samples of a
     * particular instance, there is no sample available to report the disposal of that
     * instance. In such a case the DataReader will insert a dummy sample into the
     * data_values sequence to accompany the SampleInfo element in the info_seq
     * sequence that communicates the disposal of the instance.
     *
     * The act of reading a sample sets its sample_state to READ_SAMPLE_STATE. If
     * the sample belongs to the most recent generation of the instance, it also sets the
     * view_state of the instance to NOT_NEW_VIEW_STATE. It does not affect the
     * instance_state of the instance.
     *
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
     *                  The DataReader has not yet been enabled.
     */
    DataReader& operator>>(dds::sub::LoanedSamples<T>& ls);

    /** @copydoc dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls) */
    ManipulatorSelector
    operator>> (ManipulatorSelector & (manipulator)(ManipulatorSelector&));

    /** @copydoc dds::sub::DataReader::operator>>(dds::sub::LoanedSamples<T>& ls) */
    template <typename Functor>
    ManipulatorSelector
    operator>> (Functor f);


    ///////////////////////////////////////////////////////////////////////
public:
    //== Loan Read/Take API ==================================================

    /**
     * This operation reads a sequence of typed samples from the DataReader.
     *
     * What samples are read depends on what @link dds::sub::DataReader::default_filter_state(const dds::sub::status::DataState& state)
     * default state filter @endlink has been set (default any).
     *
     * This operation reads a sequence of typed samples from the DataReader<type>. The
     * data is put into a dds::sub::LoanedSamples, which is basically a sequence of samples,
     * which in turn contains the actual data and meta information.
     *
     * The memory used for storing the sample may be loaned by the middleware thus allowing zero
     * copy operations.
     *
     * If the DataReader has no samples that meet the constraints, the resulting
     * dds::sub::LoanedSamples will be empty.
     * @code{.cpp}
     * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
     * dds::sub::Subscriber subscriber(participant);
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
     *
     * {
     *     // Read the available samples.
     *     dds::sub::LoanedSamples<Foo::Bar> samples;
     *     samples = reader.read();
     *
     *     // Access the information.
     *     dds::sub::LoanedSamples<Foo::Bar>::const_iterator it;
     *     for (it = samples.begin(); it != samples.end(); ++it) {
     *         const dds::sub::Sample<Foo::Bar>& sample = *it;
     *     }
     * }
     * // The LoanedSamples went out of scope, meaning the loan resources were taken care of.
     * @endcode
     *
     * <b><i>Selectors</i></b><br>
     * What data is read already depends on the
     * @ref anchor_dds_sub_datareader_defaultstatefilter "default state filter".
     * But it can be manipulated even more when using dds::sub::DataReader::select()
     * operation.
     *
     * @anchor anchor_dds_sub_datareader_invalidsamples
     * <b><i>Invalid Data</i></b><br>
     * Some elements in the returned sequence may not have valid data: the valid_data
     * field in the SampleInfo indicates whether the corresponding data value contains
     * any meaningful data. If not, the data value is just a ‘dummy’ sample for which only
     * the keyfields have been assigned. It is used to accompany the SampleInfo that
     * communicates a change in the instance_state of an instance for which there is
     * no ‘real’ sample available.
     *
     * For example, when an application always ‘takes’ all available samples of a
     * particular instance, there is no sample available to report the disposal of that
     * instance. In such a case the DataReader will insert a dummy sample into the
     * data_values sequence to accompany the SampleInfo element in the info_seq
     * sequence that communicates the disposal of the instance.
     *
     * The act of reading a sample sets its sample_state to READ_SAMPLE_STATE. If
     * the sample belongs to the most recent generation of the instance, it also sets the
     * view_state of the instance to NOT_NEW_VIEW_STATE. It does not affect the
     * instance_state of the instance.
     *
     * @return          The samples in the LoanedSamples container
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
     *                  The DataReader has not yet been enabled.
     */
    LoanedSamples<T> read();

    /**
     * This operation takes a sequence of typed samples from the DataReader.
     *
     * The behaviour is identical to read except for that the samples are removed
     * from the DataReader.
     *
     * What samples are taken depends on what @link dds::sub::DataReader::default_filter_state(const dds::sub::status::DataState& state)
     * default state filter @endlink has been set (default any).
     *
     * This operation takes a sequence of typed samples from the DataReader<type>. The
     * data is put into a dds::sub::LoanedSamples, which is basically a sequence of samples,
     * which in turn contains the actual data and meta information.
     *
     * The memory used for storing the sample may be loaned by the middleware thus allowing zero
     * copy operations.
     *
     * If the DataReader has no samples that meet the constraints, the resulting
     * dds::sub::LoanedSamples will be empty.
     * @code{.cpp}
     * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
     * dds::sub::Subscriber subscriber(participant);
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
     *
     * {
     *     // Take the available samples.
     *     dds::sub::LoanedSamples<Foo::Bar> samples;
     *     samples = reader.take();
     *
     *     // Access the information.
     *     dds::sub::LoanedSamples<Foo::Bar>::const_iterator it;
     *     for (it = samples.begin(); it != samples.end(); ++it) {
     *         const dds::sub::Sample<Foo::Bar>& sample = *it;
     *     }
     * }
     * // The LoanedSamples went out of scope, meaning the loan resources were taken care of.
     * @endcode
     *
     * <b><i>Selectors</i></b><br>
     * What data is taken, already depends on the
     * @ref anchor_dds_sub_datareader_defaultstatefilter "default state filter".
     * But it can be manipulated even more when using dds::sub::DataReader::select()
     * operation.
     *
     * <b><i>Invalid Data</i></b><br>
     * Some elements in the returned sequence may not have valid data: the valid_data
     * field in the SampleInfo indicates whether the corresponding data value contains
     * any meaningful data.<br>
     * Look @ref anchor_dds_sub_datareader_invalidsamples "here" for more information.
     *
     * @return          The samples in the LoanedSamples container
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
     *                  The DataReader has not yet been enabled.
     */
    LoanedSamples<T> take();

    //== Copy Read/Take API ==================================================

    // --- Forward Iterators: --- //

    /**
     * This operation reads a sequence of typed samples from the DataReader.
     *
     * What samples are read depends on what @link dds::sub::DataReader::default_filter_state(const dds::sub::status::DataState& state)
     * default state filter @endlink has been set (default any).
     *
     * The samples are copied into the application provided container using the
     * forward iterator parameter.
     *
     * If the DataReader has no samples that meet the constraints, the resulting
     * dds::sub::LoanedSamples will be empty.
     * @code{.cpp}
     * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
     * dds::sub::Subscriber subscriber(participant);
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
     *
     * // Prepare container to store samples in
     * const uint32_t MAX_SAMPLES (3)
     * std::vector<dds::sub::Sample<Foo::Bar> > samples(MAX_SAMPLES);
     * std::vector<dds::sub::Sample<Foo::Bar> >::iterator iter = samples.begin();
     *
     * // Read and copy the available samples into the vector by means of the iterator
     * uint32_t len =  reader.read(iter, MAX_SAMPLES);
     *
     * // Access the information.
     * for (iter = samples.begin(); iter != samples.end(); ++iter) {
     *     const dds::sub::Sample<Foo::Bar>& sample = *iter;
     * }
     * @endcode
     *
     * <b><i>Selectors</i></b><br>
     * What data is read already depends on the
     * @ref anchor_dds_sub_datareader_defaultstatefilter "default state filter".
     * But it can be manipulated even more when using dds::sub::DataReader::select()
     * operation.
     *
     * <b><i>Invalid Data</i></b><br>
     * Some elements in the returned sequence may not have valid data: the valid_data
     * field in the SampleInfo indicates whether the corresponding data value contains
     * any meaningful data.<br>
     * Look @ref anchor_dds_sub_datareader_invalidsamples "here" for more information.
     *
     * @param  sfit     Forward-inserting container iterator
     * @param  max_samples Maximum samples to read and copy into the given container
     * @return          The number of samples in the LoanedSamples container
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
     *                  The DataReader has not yet been enabled.
     */
    template <typename SamplesFWIterator>
    uint32_t
    read(SamplesFWIterator sfit,
         uint32_t max_samples);

    /**
     * This operation takes a sequence of typed samples from the DataReader.
     *
     * What samples are take depends on what @link dds::sub::DataReader::default_filter_state(const dds::sub::status::DataState& state)
     * default state filter @endlink has been set (default any).
     *
     * The samples are copied into the application provided container using the
     * forward iterator parameter.
     *
     * If the DataReader has no samples that meet the constraints, the resulting
     * dds::sub::LoanedSamples will be empty.
     * @code{.cpp}
     * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
     * dds::sub::Subscriber subscriber(participant);
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
     *
     * // Prepare container to store samples in
     * const uint32_t MAX_SAMPLES (3)
     * std::vector<dds::sub::Sample<Foo::Bar> > samples(MAX_SAMPLES);
     * std::vector<dds::sub::Sample<Foo::Bar> >::iterator iter = samples.begin();
     *
     * // Take and copy the available samples into the vector by means of the iterator
     * uint32_t len =  reader.take(iter, MAX_SAMPLES);
     *
     * // Access the information.
     * for (iter = samples.begin(); iter != samples.end(); ++iter) {
     *     const dds::sub::Sample<Foo::Bar>& sample = *iter;
     * }
     * @endcode
     *
     * <b><i>Selectors</i></b><br>
     * What data is taken, already depends on the
     * @ref anchor_dds_sub_datareader_defaultstatefilter "default state filter".
     * But it can be manipulated even more when using dds::sub::DataReader::select()
     * operation.
     *
     * <b><i>Invalid Data</i></b><br>
     * Some elements in the returned sequence may not have valid data: the valid_data
     * field in the SampleInfo indicates whether the corresponding data value contains
     * any meaningful data.<br>
     * Look @ref anchor_dds_sub_datareader_invalidsamples "here" for more information.
     *
     * @param  sfit     Forward-inserting container iterator
     * @param  max_samples Maximum samples to take and copy into the given container
     * @return          The number of samples in the given container
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
     *                  The DataReader has not yet been enabled.
     */
    template <typename SamplesFWIterator>
    uint32_t
    take(SamplesFWIterator sfit,
         uint32_t max_samples);


    // --- Back-Inserting Iterators: --- //

    /**
     * This operation reads a sequence of typed samples from the DataReader.
     *
     * What samples are read depends on what @link dds::sub::DataReader::default_filter_state(const dds::sub::status::DataState& state)
     * default state filter @endlink has been set (default any).
     *
     * The samples are copied into the application provided container using a
     * back-inserting iterator. Notice that as a consequence of using a back-inserting
     * iterator, this operation may allocate memory to resize the underlying container.
     *
     * If the DataReader has no samples that meet the constraints, the resulting
     * dds::sub::LoanedSamples will be empty.
     * @code{.cpp}
     * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
     * dds::sub::Subscriber subscriber(participant);
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
     *
     * // Prepare container to store samples in
     * std::vector<dds::sub::Sample<Foo::Bar> > samples;
     * std::back_insert_iterator< std::vector<dds::sub::Sample<Foo::Bar> > > bi(samples);
     *
     * // Read and copy the available samples into the vector by means of the iterator
     * uint32_t len =  reader.read(bi);
     *
     * // Access the information.
     * std::vector<dds::sub::Sample<Space::Type1> >::iterator iter = samples.begin();
     * for (iter = samples.begin(); iter != samples.end(); ++iter) {
     *     const dds::sub::Sample<Foo::Bar>& sample = *iter;
     * }
     * @endcode
     *
     * <b><i>Selectors</i></b><br>
     * What data is read already depends on the
     * @ref anchor_dds_sub_datareader_defaultstatefilter "default state filter".
     * But it can be manipulated even more when using dds::sub::DataReader::select()
     * operation.
     *
     * <b><i>Invalid Data</i></b><br>
     * Some elements in the returned sequence may not have valid data: the valid_data
     * field in the SampleInfo indicates whether the corresponding data value contains
     * any meaningful data.<br>
     * Look @ref anchor_dds_sub_datareader_invalidsamples "here" for more information.
     *
     * @param  sbit     Back-inserting container iterator
     * @return          The number of samples in the given container
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
     *                  The DataReader has not yet been enabled.
     */
    template <typename SamplesBIIterator>
    uint32_t
    read(SamplesBIIterator sbit);

    /**
     * This operation takes a sequence of typed samples from the DataReader.
     *
     * What samples are take depends on what @link dds::sub::DataReader::default_filter_state(const dds::sub::status::DataState& state)
     * default state filter @endlink has been set (default any).
     *
     * The samples are copied into the application provided container using a
     * back-inserting iterator. Notice that as a consequence of using a back-inserting
     * iterator, this operation may allocate memory to resize the underlying container.
     *
     * If the DataReader has no samples that meet the constraints, the resulting
     * dds::sub::LoanedSamples will be empty.
     * @code{.cpp}
     * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
     * dds::sub::Subscriber subscriber(participant);
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
     *
     * // Prepare container to store samples in
     * std::vector<dds::sub::Sample<Foo::Bar> > samples;
     * std::back_insert_iterator< std::vector<dds::sub::Sample<Foo::Bar> > > bi(samples);
     *
     * // Take and copy the available samples into the vector by means of the iterator
     * uint32_t len =  reader.take(bi);
     *
     * // Access the information.
     * std::vector<dds::sub::Sample<Space::Type1> >::iterator iter = samples.begin();
     * for (iter = samples.begin(); iter != samples.end(); ++iter) {
     *     const dds::sub::Sample<Foo::Bar>& sample = *iter;
     * }
     * @endcode
     *
     * <b><i>Selectors</i></b><br>
     * What data is taken, already depends on the
     * @ref anchor_dds_sub_datareader_defaultstatefilter "default state filter".
     * But it can be manipulated even more when using dds::sub::DataReader::select()
     * operation.
     *
     * <b><i>Invalid Data</i></b><br>
     * Some elements in the returned sequence may not have valid data: the valid_data
     * field in the SampleInfo indicates whether the corresponding data value contains
     * any meaningful data.<br>
     * Look @ref anchor_dds_sub_datareader_invalidsamples "here" for more information.
     *
     * @param  sbit     Back-inserting container iterator
     * @return          The number of samples in the given container
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
     *                  The DataReader has not yet been enabled.
     */
    template <typename SamplesBIIterator>
    uint32_t
    take(SamplesBIIterator sbit);
public:
    //========================================================================
    //== DSL Method for dealing with instances, content and status filters.

    /**
     * Get a dds::sub::DataReader::Selector instance that acts on this DataReader.
     *
     * The DataReader::read and DataReader::take read all samples that are
     * available within the DataReader (provided that the
     * @ref anchor_dds_sub_datareader_defaultstatefilter "default state filter"
     * hasn't been changed).
     *
     * <b><i>Selectors</i></b><br>
     * A Selector can perform complex data selections, such as per-instance selection,
     * content and status filtering, etc when reading or taking. Such a selector is
     * returned by this operation. Setting filters on the Selector can be concatenated,
     * resulting in the following example code.
     * @code{.cpp}
     * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
     * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
     * dds::sub::Subscriber subscriber(participant);
     * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
     *
     * // Take a maximum of 3 new samples of a certain instance.
     * {
     *     dds::sub::LoanedSamples<Foo::Bar> samples;
     *     samples = reader.select()
     *                         .max_samples(3)
     *                         .state(dds::sub::status::DataState::new_data())
     *                         .instance(someValidInstanceHandle)
     *                         .take();
     * }
     * @endcode
     *
     * <i>Please be aware that using pre-set filters on the DataReader and then
     * call read() or take() on that reader explicitly will perform better than
     * having to create Selectors for every read (which is what essentially
     * happens in the code example above). Performance can be increase by creating
     * a selector up front and using that multiple times
     * (see dds::sub::DataReader::Selector).</i>
     *
     * @return          A Selector that acts on the DataReader
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
     *                  The DataReader has not yet been enabled.
     */
    Selector select();

    //========================================================================
    //== Instance Management
public:
    /**
     * This operation retrieves the key value of a specific instance.
     *
     * This operation can be used to retrieve the instance key that corresponds
     * to an instance_handle. The operation will only fill the fields that form
     * the key inside the sample instance.
     *
     * This operation may raise a InvalidArgumentError exception if the InstanceHandle
     * does not correspond to an existing data-object known to the DataReader.
     * If the implementation is not able to check invalid handles, then the
     * result in this situation is unspecified.
     *
     * @param  h        The instance handle
     * @return          A topic instance with the handle and key fields set
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
     *                  The DataReader has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataReader.
     */
    dds::topic::TopicInstance<T> key_value(const dds::core::InstanceHandle& h);

    /**
     * This operation retrieves the key value of a specific instance.
     *
     * This operation can be used to retrieve the instance key that corresponds
     * to an instance_handle. The operation will only fill the fields that form
     * the key inside the sample instance.
     *
     * This operation may raise a InvalidArgumentError exception if the InstanceHandle
     * does not correspond to an existing data-object known to the DataReader.
     * If the implementation is not able to check invalid handles, then the
     * result in this situation is unspecified.
     *
     * The Sample is added as parameter to be able to overload this operation.
     *
     * @param[out] sample A sample to set the key fields of
     * @param[in] h     The instance handle
     * @return          The given sample with the key fields set
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
     *                  The DataReader has not yet been enabled.
     * @throws dds::core::PreconditionNotMetError
     *                  The handle has not been registered with this DataReader.
     */
    T& key_value(T& sample, const dds::core::InstanceHandle& h);

    /**
     * This operation returns the value of the instance handle which corresponds
     * to the instance_data.
     *
     * The instance handle can be used in read operations that operate
     * on a specific instance. Note that DataReader instance handles are local, and are
     * not interchangeable with DataWriter instance handles nor with instance handles
     * of an other DataReader.
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
    const dds::core::InstanceHandle
    lookup_instance(const T& key) const;

public:

    /**
     * Register a listener with the DataReader.
     *
     * This operation attaches a DataReaderListener to the DataReader. Only one
     * DataReaderListener can be attached to each DataReader. If a
     * DataReaderListener was already attached, the operation will replace it with the
     * new one. When the listener is the NULL pointer, it represents a listener that is
     * treated as a NOOP for all statuses activated in the bit mask.
     *
     * Listener un-registration is performed by setting the listener to NULL and mask none().
     *
     * @anchor anchor_dds_sub_datareader_commstatus
     * <i>Communication Status</i><br>
     * For each communication status, the StatusChangedFlag flag is initially set to
     * FALSE. It becomes TRUE whenever that communication status changes. For each
     * communication status activated in the mask, the associated DataReaderListener
     * operation is invoked and the communication status is reset to FALSE, as the listener
     * implicitly accesses the status which is passed as a parameter to that operation. The
     * status is reset prior to calling the listener, so if the application calls the
     * get_<status_name>_status from inside the listener it will see the status
     * already reset. An exception to this rule is the NULL listener, which does not reset the
     * communication statuses for which it is invoked.
     *
     * The following statuses are applicable to the DataReaderListener:
     *  - dds::core::status::StatusMask::requested_deadline_missed()
     *  - dds::core::status::StatusMask::requested_incompatible_qos()
     *  - dds::core::status::StatusMask::sample_lost()
     *  - dds::core::status::StatusMask::sample_rejected()
     *  - dds::core::status::StatusMask::data_available()
     *  - dds::core::status::StatusMask::liveliness_changed()
     *  - dds::core::status::StatusMask::subscription_matched()
     *
     * Be aware that the SUBSCRIPTION_MATCHED_STATUS is not applicable when the
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
     * statuses specified in the "Data Distribution Service for Real-time Systems Version
     * 1.2" specification which are applicable to the PublisherListener.
     *
     * @anchor anchor_dds_sub_datareader_commpropagation
     * <i>Status Propagation</i><br>
     * In case a communication status is not activated in the mask of the
     * DataReaderListener, the SubscriberListener of the containing Subscriber
     * is invoked (if attached and activated for the status that occurred). This allows the
     * application to set a default behaviour in the SubscriberListener of the containing
     * Subscriber and a DataReader specific behaviour when needed. In case the
     * communication status is not activated in the mask of the SubscriberListener as
     * well, the communication status will be propagated to the
     * DomainParticipantListener of the containing DomainParticipant. In case
     * the DomainParticipantListener is also not attached or the communication
     * status is not activated in its mask, the application is not notified of the change.
     *
     * The statuses DATA_ON_READERS_STATUS and DATA_AVAILABLE_STATUS are
     * "Read Communication Statuses" and are an exception to all other plain
     * communication statuses: they have no corresponding status structure that can be
     * obtained with a get_<status_name>_status operation and they are mutually
     * exclusive. When new information becomes available to a DataReader, the Data
     * Distribution Service will first look in an attached and activated
     * SubscriberListener or DomainParticipantListener (in that order) for the
     * DATA_ON_READERS_STATUS. In case the DATA_ON_READERS_STATUS can not be
     * handled, the Data Distribution Service will look in an attached and activated
     * DataReaderListener, SubscriberListener or DomainParticipant
     * Listener for the DATA_AVAILABLE_STATUS (in that order).
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @param listener  the listener
     * @param event_mask the mask defining the events for which the listener
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
    void listener(Listener* listener,
                  const dds::core::status::StatusMask& event_mask);

    /**
     * Get the listener of this DataReader.
     *
     * See also @ref DCPS_Modules_Infrastructure_Listener "listener information".
     *
     * @return the listener
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     */
    Listener* listener() const;
};


#endif /* OMG_DDS_SUB_TDATA_READER_HPP_ */
