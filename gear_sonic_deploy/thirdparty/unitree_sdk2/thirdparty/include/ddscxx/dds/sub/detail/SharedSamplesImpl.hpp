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
#ifndef CYCLONEDDS_DDS_SUB_SHARED_SAMPLES_IMPL_HPP_
#define CYCLONEDDS_DDS_SUB_SHARED_SAMPLES_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */

// Implementation

namespace dds
{
namespace sub
{

template <typename T, template <typename Q> class DELEGATE>
SharedSamples<T, DELEGATE>::SharedSamples() : delegate_(new DELEGATE<T>()) { }

template <typename T, template <typename Q> class DELEGATE>
SharedSamples<T, DELEGATE>::SharedSamples(dds::sub::LoanedSamples<T> ls) : delegate_(new DELEGATE<T>(ls)) { }

template <typename T, template <typename Q> class DELEGATE>
SharedSamples<T, DELEGATE>::~SharedSamples() {  }

template <typename T, template <typename Q> class DELEGATE>
SharedSamples<T, DELEGATE>::SharedSamples(const SharedSamples& other)
{
    delegate_ = other.delegate_;
}

template <typename T, template <typename Q> class DELEGATE>
typename SharedSamples<T, DELEGATE>::const_iterator SharedSamples<T, DELEGATE>::begin() const
{
    return delegate()->begin();
}

template <typename T, template <typename Q> class DELEGATE>
typename SharedSamples<T, DELEGATE>::const_iterator SharedSamples<T, DELEGATE>::end() const
{
    return delegate()->end();
}

template <typename T, template <typename Q> class DELEGATE>
const typename SharedSamples<T, DELEGATE>::DELEGATE_REF_T& SharedSamples<T, DELEGATE>::delegate() const
{
    return delegate_;
}

template <typename T, template <typename Q> class DELEGATE>
typename SharedSamples<T, DELEGATE>::DELEGATE_REF_T& SharedSamples<T, DELEGATE>::delegate()
{
    return delegate_;
}

template <typename T, template <typename Q> class DELEGATE>
uint32_t SharedSamples<T, DELEGATE>::length() const
{
    return delegate_->length();
}

}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_SUB_SHARED_SAMPLES_IMPL_HPP_ */
