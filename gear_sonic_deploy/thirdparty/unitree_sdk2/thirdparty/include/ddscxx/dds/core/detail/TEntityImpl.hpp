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
#ifndef CYCLONEDDS_DDS_CORE_TENTITY_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_TENTITY_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/detail/ReferenceImpl.hpp>
#include <dds/core/TEntity.hpp>
#include <org/eclipse/cyclonedds/core/ReportUtils.hpp>

// Implementation

template <typename DELEGATE>
void
dds::core::TEntity<DELEGATE>::enable()
{
    this->delegate()->enable();
}

template <typename DELEGATE>
const dds::core::status::StatusMask
dds::core::TEntity<DELEGATE>::status_changes()
{
    return this->delegate()->status_changes();
}

template <typename DELEGATE>
const dds::core::InstanceHandle
dds::core::TEntity<DELEGATE>::instance_handle() const
{
    return this->delegate()->instance_handle();
}

template <typename DELEGATE>
void
dds::core::TEntity<DELEGATE>::close()
{
    this->delegate()->close();
}

template <typename DELEGATE>
void
dds::core::TEntity<DELEGATE>::retain()
{
    this->delegate()->retain();
}

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_TENTITY_IMPL_HPP_ */
