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
#ifndef OMG_DDS_SUB_TLOANED_SAMPLES_HPP_
#define OMG_DDS_SUB_TLOANED_SAMPLES_HPP_

#include <dds/core/ref_traits.hpp>
#include <dds/sub/SampleRef.hpp>

#include <dds/sub/detail/LoanedSamples.hpp>

/** @cond */
namespace dds
{
namespace sub
{
template <typename T,
          template <typename Q> class DELEGATE = dds::sub::detail::LoanedSamples>
class LoanedSamples;

// Used by C++11 compilers to allow for using LoanedSamples
// and SharedSamples in a range-based for-loop.
template <typename T> typename T::const_iterator cbegin(const T& t);
template <typename T> typename T::const_iterator cend(const T& t);
}

}
/** @endcond */

/**
 * @brief
 * This class encapsulates and automates the management of loaned samples.
 *
 * It is a container which is used to hold samples which have been read
 * or taken by the DataReader. Samples are effectively "loaned" from the
 * DataReader to avoid the need to copy the data. When the LoanedSamples
 * container goes out of scope the loan is automatically returned.
 *
 * LoanedSamples maintains a ref count so that the loan will only be
 * returned once all copies of the same LoanedSamples have been destroyed.
 *
 * @anchor anchor_dds_sub_loanedsamples_example
 * @code{.cpp}
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
 * dds::sub::Subscriber subscriber(participant);
 * dds::sub::DataReader<Foo::Bar> reader(subscriber, topic);
 *
 * // Assume there is data to read
 * {
 *     dds::sub::LoanedSamples<Foo::Bar> samples = reader.read();
 *     dds::sub::LoanedSamples<Foo::Bar>::const_iterator it;
 *     for (it = samples.begin(); it != samples.end(); ++it) {
 *         const dds::sub::Sample<Foo::Bar>& sample = *it;
 *         const Foo::Bar& data = sample.data();
 *         const dds::sub::SampleInfo& info = sample.info();
 *         // Use sample data and meta information.
 *     }
 *
 *     function(samples);
 * }
 * // LoanedSamples out of scope. Whether the loan is returned, depends what the reference
 * // count of the LoanedSamples is. That again, depends on what the function() did with it.
 * // Maybe function() stored the LoanedSamples, maybe not. Whatever the case, LoanedSamples
 * // takes care of the loan and resource handling.
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Subscription_DataSample "DataSample"
 * @see for more information: @ref DCPS_Modules_Subscription_SampleInfo "SampleInfo"
 * @see for more information: @ref DCPS_Modules_Subscription "Subscription"
 */
template <typename T, template <typename Q> class DELEGATE>
class dds::sub::LoanedSamples
{
public:
    /**
     * Convenience typedef for the type of the data sample.
     */
    typedef T DataType;

    /**
     * Convenience typedef for the iterator over the loaned samples.
     */
    typedef typename DELEGATE<T>::const_iterator const_iterator;

    /** @cond */
    typedef typename dds::core::smart_ptr_traits< DELEGATE<T> >::ref_type DELEGATE_REF_T;
    /** @endcond */

public:
    /**
     * Constructs a LoanedSamples instance.
     */
    LoanedSamples();

    /**
     * Implicitly return the loan if this is the last object with a reference to the
     * contained loan.
     */
    ~LoanedSamples();

    /**
     * Copies a LoanedSamples instance.
     *
     * No actual data samples are copied.<br>
     * Just references and reference counts are updated.
     */
    LoanedSamples(const LoanedSamples& other);

    /**
     * Move-constructs a LoanedSamples instance.
     *
     * No actual data samples are copied.<br>
     * Just references and reference counts are updated.
     */
    LoanedSamples(LoanedSamples&& other);

    /**
     * Copy-assigns a LoanedSamples instance to another.
     *
     * No actual data samples are copied.<br>
     * Just references and reference counts are updated.
     */
    LoanedSamples& operator=(const LoanedSamples& other) = default;

    /**
     * Move-assigns a LoanedSamples instance to another.
     *
     * No actual data samples are copied.<br>
     * Just references and reference counts are updated.
     */
    LoanedSamples& operator=(LoanedSamples&& other) = default;


public:
    /**
     * Gets an iterator pointing to the first sample in the LoanedSamples container.
     *
     * See @ref anchor_dds_sub_loanedsamples_example "example".
     *
     * @return an iterator pointing to the first sample
     */
    const_iterator begin() const;

    /**
     * Gets an iterator pointing to the end of the LoanedSamples container.
     *
     * See @ref anchor_dds_sub_loanedsamples_example "example".
     *
     * @return an iterator pointing to the end of the container
     */
    const_iterator end() const;

    /**
     * @cond
     * The delegate part of the LoanedSamples should not be part of the API.
     */
    /**
     * Gets the delegate.
     *
     * @return the delegate
     */
    const DELEGATE_REF_T& delegate() const;

    /**
     * Gets the delegate.
     *
     * @return the delegate
     */
    DELEGATE_REF_T& delegate();
    /** @endcond */

    /**
     * Gets the number of samples within the LoanedSamples container.
     *
     * @return the number of samples
     */
    uint32_t length() const;

private:
    DELEGATE_REF_T delegate_;
};

namespace dds
{
namespace sub
{
/**
 * Move loan and its ownership to a new LoanedSamples object.
 *
 * @return LoanedSampless
 */
template <typename T, template <typename Q> class D>
LoanedSamples<T, D >
move(LoanedSamples<T, D >& a);
}
}

#include <dds/sub/detail/LoanedSamplesImpl.hpp>

#endif /* OMG_DDS_SUB_TLOANED_SAMPLES_HPP_ */
