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
#ifndef CYCLONEDDS_DDS_CORE_COND_DETAIL_WAITSET_HPP_
#define CYCLONEDDS_DDS_CORE_COND_DETAIL_WAITSET_HPP_

/**
 * @file
 */

// Implementation

#include <dds/core/cond/detail/TWaitSetImpl.hpp>
#include <org/eclipse/cyclonedds/core/cond/WaitSetDelegate.hpp>

namespace dds
{
namespace core
{
namespace cond
{
namespace detail
{
typedef dds::core::cond::TWaitSet<org::eclipse::cyclonedds::core::cond::WaitSetDelegate> WaitSet;
}
}
}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_COND_DETAIL_WAITSET_HPP_ */
