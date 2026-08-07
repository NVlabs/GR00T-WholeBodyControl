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
#ifndef CYCLONEDDS_DDS_SUB_TSAMPLE_HPP_
#define CYCLONEDDS_DDS_SUB_TSAMPLE_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/TSample.hpp>

// Implementation
namespace dds
{
namespace sub
{

template <typename T, template <typename Q> class DELEGATE>
Sample<T, DELEGATE>::Sample() : dds::core::Value< DELEGATE<T> >() {}

template <typename T, template <typename Q> class DELEGATE>
Sample<T, DELEGATE>::Sample(const T& data, const SampleInfo& info) : dds::core::Value< DELEGATE<T> >(data, info) { }

template <typename T, template <typename Q> class DELEGATE>
Sample<T, DELEGATE>::Sample(const Sample& other) : dds::core::Value< DELEGATE<T> >(other.delegate()) { }

template <typename T, template <typename Q> class DELEGATE>
const typename Sample<T, DELEGATE>::DataType& Sample<T, DELEGATE>::data() const
{
    return this->delegate().data();
}

template <typename T, template <typename Q> class DELEGATE>
void Sample<T, DELEGATE>::data(const DataType& d)
{
    this->delegate().data(d);
}

template <typename T, template <typename Q> class DELEGATE>
const SampleInfo& Sample<T, DELEGATE>::info() const
{
    return this->delegate().info();
}

template <typename T, template <typename Q> class DELEGATE>
void Sample<T, DELEGATE>::info(const SampleInfo& i)
{
    this->delegate().info(i);
}

}
}
// End of implementation
#endif /* CYCLONEDDS_DDS_SUB_TSAMPLE_HPP_ */
