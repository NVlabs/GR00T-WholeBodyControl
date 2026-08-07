/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_DDS_SUB_DETAIL_SHARED_SAMPLES_HPP_
#define CYCLONEDDS_DDS_SUB_DETAIL_SHARED_SAMPLES_HPP_

/**
 * @file
 */

#include <dds/sub/LoanedSamples.hpp>

// Implementation

namespace dds
{
namespace sub
{
namespace detail
{

template <typename T>
class SharedSamples
{
public:
    typedef typename std::vector< dds::sub::Sample<T, Sample> >::iterator iterator;
    typedef typename std::vector< dds::sub::Sample<T, Sample> >::const_iterator const_iterator;

public:
    SharedSamples() { }

    SharedSamples(dds::sub::LoanedSamples<T> ls) : samples_(ls) { }

    ~SharedSamples()
    {

    }

public:

    iterator mbegin()
    {
        return samples_->begin();
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
        /** @internal @todo Possible RTF size issue ? */
        return static_cast<uint32_t>(samples_.length());
    }

    void resize(uint32_t s)
    {
        samples_.resize(s);
    }

private:
    dds::sub::LoanedSamples<T> samples_;
};

}
}
}


// End of implementation

#endif /* CYCLONEDDS_DDS_SUB_DETAIL_SHARED_SAMPLES_HPP_ */
