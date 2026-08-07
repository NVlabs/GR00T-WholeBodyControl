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
#ifndef CYCLONEDDS_DDS_TOPIC_DETAIL_TBUILTINTOPICKEY_IMPL_HPP_
#define CYCLONEDDS_DDS_TOPIC_DETAIL_TBUILTINTOPICKEY_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/topic/TBuiltinTopicKey.hpp>

// Implementation

namespace dds
{
namespace topic
{

template <typename D>
const int32_t* TBuiltinTopicKey<D>::value() const
{
    return this->delegate().value();
}

template <typename D>
void TBuiltinTopicKey<D>::value(int32_t v[])
{
    return this->delegate().value(v);
}

}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_TOPIC_DETAIL_TBUILTINTOPICKEY_IMPL_HPP_ */
