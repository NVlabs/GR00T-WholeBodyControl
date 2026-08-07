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
#ifndef OMG_DDS_SUB_SHARED_SAMPLES_HPP_
#define OMG_DDS_SUB_SHARED_SAMPLES_HPP_


#include <dds/core/Reference.hpp>
#include <dds/sub/Sample.hpp>
#include <dds/sub/LoanedSamples.hpp>
#include <dds/sub/detail/SharedSamples.hpp>

/** @cond */
namespace dds
{
namespace sub
{
template <typename T,
          template <typename Q> class DELEGATE = detail::SharedSamples>
class SharedSamples;
}
}
/** @endcond */

/**
 * @brief
 * This class encapsulates and automates the management of loaned samples.
 *
 * @note The specification of this class is not yet finished. It will possibly
 *       change in the future. Anyway, this class is basically a copy of LoanedSamples
 *       when considering its functionality. So, anything you would want to do with this
 *       class can also be done by the LoanedSamples.
 */
template <typename T, template <typename Q> class DELEGATE>
class dds::sub::SharedSamples
{
public:
    typedef T                     DataType;
    typedef typename DELEGATE<T>::const_iterator        const_iterator;

    typedef typename dds::core::smart_ptr_traits< DELEGATE<T> >::ref_type DELEGATE_REF_T;

public:
    /**
     * Constructs a SharedSamples instance.
     */
    SharedSamples();

    /**
     * Constructs an instance of SharedSamples and
     * removes the ownership of the loan from the LoanedSamples.
     * As a result the destruction of the LoanedSamples object
     * will have no effect on loaned data. Loaned data will be returned
     * automatically once the delegate of this reference type will have a
     * zero reference count.
     *
     * @param ls the loaned samples
     *
     */
    SharedSamples(dds::sub::LoanedSamples<T> ls);

    /**
     * Copies a SharedSamples instance.
     */
    SharedSamples(const SharedSamples& other);

    ~SharedSamples();


public:
    /**
     * Gets an iterator pointing to the beginning of the samples.
     *
     * @return an iterator pointing to the beginning of the samples
     */
    const_iterator begin() const;

    /**
     * Gets an iterator pointing to the beginning of the samples.
     *
     * @return an iterator pointing to the beginning of the samples
     */
    const_iterator  end() const;

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

    /**
     * Returns the number of samples.
     *
     * @return the number of samples
     */
    uint32_t length() const;

private:
    DELEGATE_REF_T delegate_;
};

#include <dds/sub/detail/SharedSamplesImpl.hpp>

#endif /* OMG_DDS_SUB_SHARED_SAMPLES_HPP_ */
