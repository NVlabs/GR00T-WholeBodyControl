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
#ifndef CYCLONEDDS_DDS_CORE_COND_DETAIL_STATUSCONDITION_HPP_
#define CYCLONEDDS_DDS_CORE_COND_DETAIL_STATUSCONDITION_HPP_

/**
 * @file
 */

// Implementation
namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{
namespace cond
{
class StatusConditionDelegate;
}
}
}
}
}

namespace dds
{
namespace core
{
namespace cond
{

template <typename DELEGATE>
class TStatusCondition;

namespace detail
{
typedef dds::core::cond::TStatusCondition<org::eclipse::cyclonedds::core::cond::StatusConditionDelegate> StatusCondition;
}
}
}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_COND_DETAIL_STATUSCONDITION_HPP_ */
