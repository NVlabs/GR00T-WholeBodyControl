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
#ifndef CYCLONEDDS_DDS_SUB_TSAMPLEREF_HPP_
#define CYCLONEDDS_DDS_SUB_TSAMPLEREF_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/TSampleRef.hpp>

// Implementation
namespace dds
{
namespace sub
{

template <typename T, template <typename Q> class DELEGATE>
SampleRef<T, DELEGATE>::SampleRef() : dds::core::Value< DELEGATE<T> >() {}

template <typename T, template <typename Q> class DELEGATE>
SampleRef<T, DELEGATE>::SampleRef(const T& data, const SampleInfo& info) : dds::core::Value< DELEGATE<T> >(data, info) { }

template <typename T, template <typename Q> class DELEGATE>
SampleRef<T, DELEGATE>::SampleRef(const SampleRef& other) : dds::core::Value< DELEGATE<T> >(other.delegate()) { }

template <typename T, template <typename Q> class DELEGATE>
const typename SampleRef<T, DELEGATE>::DataType& SampleRef<T, DELEGATE>::data() const
{
    return this->delegate().data();
}

template <typename T, template <typename Q> class DELEGATE>
const SampleInfo& SampleRef<T, DELEGATE>::info() const
{
    return this->delegate().info();
}

}
}
// End of implementation
#endif /* CYCLONEDDS_DDS_SUB_TSAMPLEREF_HPP_ */
