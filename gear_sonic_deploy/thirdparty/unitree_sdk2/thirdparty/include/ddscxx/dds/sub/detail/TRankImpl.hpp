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
#ifndef CYCLONEDDS_DDS_SUB_DETAIL_TRANK_IMPL_HPP_
#define CYCLONEDDS_DDS_SUB_DETAIL_TRANK_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/TRank.hpp>

// Implementation
namespace dds
{
namespace sub
{

template <typename DELEGATE>
TRank<DELEGATE>::TRank() { }

template <typename DELEGATE>
TRank<DELEGATE>::TRank(int32_t s, int32_t a, int32_t ag)
    : dds::core::Value<DELEGATE>(s, a, ag) { }

template <typename DELEGATE>
int32_t TRank<DELEGATE>::absolute_generation() const
{
    return this->delegate().absolute_generation();
}

template <typename DELEGATE>
inline int32_t TRank<DELEGATE>::generation() const
{
    return this->delegate().generation();
}

template <typename DELEGATE>
inline int32_t TRank<DELEGATE>::sample() const
{
    return this->delegate().sample();
}

}
}
// End of implementation

#endif /* CYCLONEDDS_DDS_SUB_DETAIL_TRANK_IMPL_HPP_ */
