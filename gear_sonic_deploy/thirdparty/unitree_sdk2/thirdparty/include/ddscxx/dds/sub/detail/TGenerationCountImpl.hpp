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
#ifndef CYCLONEDDS_DDS_SUB_DETAIL_TGENERATIONCOUNT_IMPL_HPP_
#define CYCLONEDDS_DDS_SUB_DETAIL_TGENERATIONCOUNT_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/TGenerationCount.hpp>

// Implementation

namespace dds
{
namespace sub
{

template <typename DELEGATE>
TGenerationCount<DELEGATE>::TGenerationCount() { }

template <typename DELEGATE>
TGenerationCount<DELEGATE>::TGenerationCount(int32_t dgc, int32_t nwgc)
    : dds::core::Value<DELEGATE>(dgc, nwgc) { }

template <typename DELEGATE>
int32_t TGenerationCount<DELEGATE>::disposed() const
{
    return this->delegate().disposed();
}

template <typename DELEGATE>
inline int32_t TGenerationCount<DELEGATE>::no_writers() const
{
    return this->delegate().no_writers();
}

}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_SUB_DETAIL_TGENERATIONCOUNT_IMPL_HPP_ */
