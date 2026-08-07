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
#ifndef CYCLONEDDS_DDS_SUB_DETAIL_TSAMPLEINFO_IMPL_HPP_
#define CYCLONEDDS_DDS_SUB_DETAIL_TSAMPLEINFO_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/sub/TSampleInfo.hpp>

// Implementation
namespace dds
{
namespace sub
{

template <typename DELEGATE>
TSampleInfo<DELEGATE>::TSampleInfo() { }

template <typename DELEGATE>
const dds::core::Time TSampleInfo<DELEGATE>::timestamp() const
{
    return this->delegate().timestamp();
}

template <typename DELEGATE>
const dds::sub::status::DataState TSampleInfo<DELEGATE>::state() const
{
    return this->delegate().state();
}

template <typename DELEGATE>
dds::sub::GenerationCount TSampleInfo<DELEGATE>::generation_count() const
{
    return this->delegate().generation_count();
}

template <typename DELEGATE>
dds::sub::Rank TSampleInfo<DELEGATE>::rank() const
{
    return this->delegate().rank();
}

template <typename DELEGATE>
bool TSampleInfo<DELEGATE>::valid() const
{
    return this->delegate().valid();
}

template <typename DELEGATE>
dds::core::InstanceHandle TSampleInfo<DELEGATE>::instance_handle() const
{
    return this->delegate().instance_handle();
}

template <typename DELEGATE>
dds::core::InstanceHandle TSampleInfo<DELEGATE>::publication_handle() const
{
    return this->delegate().publication_handle();
}
}
}
// End of implementation

#endif /* CYCLONEDDS_DDS_SUB_DETAIL_TSAMPLEINFO_IMPL_HPP_ */
