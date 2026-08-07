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
#ifndef OMG_SUB_DETAIL_LOANED_SAMPLES_IMPL_HPP_
#define OMG_SUB_DETAIL_LOANED_SAMPLES_IMPL_HPP_

namespace dds
{
namespace sub
{
namespace detail
{

template <typename T>
class LoanedSamples
{
public:

    typedef std::vector< dds::sub::SampleRef<T, dds::sub::detail::SampleRef> > LoanedSamplesContainer;
    typedef typename std::vector< dds::sub::SampleRef<T, dds::sub::detail::SampleRef> >::iterator iterator;
    typedef typename std::vector< dds::sub::SampleRef<T, dds::sub::detail::SampleRef> >::const_iterator const_iterator;

public:
    LoanedSamples() { }

    ~LoanedSamples()
    {

    }

public:

    iterator mbegin()
    {
        return samples_.begin();
    }

    const_iterator begin() const
    {
        return samples_.begin();
    }

    const_iterator end() const
    {
        return samples_.end();
    }

    uint32_t length() const
    {
        return static_cast<uint32_t>(samples_.size());
    }

    void reserve(uint32_t s)
    {
        samples_.reserve(s);
    }

    void resize(uint32_t s)
    {
         samples_.resize(s);
    }

    dds::sub::SampleRef<T, dds::sub::detail::SampleRef>& operator[] (uint32_t i)
    {
        return this->samples_[i];
    }

    dds::sub::SampleRef<T, dds::sub::detail::SampleRef> * get_buffer() {
        return this->samples_.data();
    }


private:
    LoanedSamplesContainer samples_;
};

template <>
class LoanedSamples<org::eclipse::cyclonedds::topic::CDRBlob>
{
public:

    typedef std::vector< dds::sub::Sample<org::eclipse::cyclonedds::topic::CDRBlob, dds::sub::detail::Sample> > LoanedSamplesContainer;
    typedef typename std::vector< dds::sub::Sample<org::eclipse::cyclonedds::topic::CDRBlob, dds::sub::detail::Sample> >::iterator iterator;
    typedef typename std::vector< dds::sub::Sample<org::eclipse::cyclonedds::topic::CDRBlob, dds::sub::detail::Sample> >::const_iterator const_iterator;

public:
    LoanedSamples() { }

    ~LoanedSamples()
    {

    }

public:

    iterator mbegin()
    {
        return samples_.begin();
    }

    const_iterator begin() const
    {
        return samples_.begin();
    }

    const_iterator end() const
    {
        return samples_.end();
    }

    uint32_t length() const
    {
        return static_cast<uint32_t>(samples_.size());
    }

    void reserve(uint32_t s)
    {
        samples_.reserve(s);
    }

    void resize(uint32_t s)
    {
         samples_.resize(s);
    }

    dds::sub::Sample<org::eclipse::cyclonedds::topic::CDRBlob, dds::sub::detail::Sample>& operator[] (uint32_t i)
    {
        return this->samples_[i];
    }

    dds::sub::Sample<org::eclipse::cyclonedds::topic::CDRBlob, dds::sub::detail::Sample> * get_buffer() {
        return this->samples_.data();
    }


private:
    LoanedSamplesContainer samples_;
};


}
}
}
#endif /* OMG_SUB_DETAIL_LOANED_SAMPLES_IMPL_HPP_ */
