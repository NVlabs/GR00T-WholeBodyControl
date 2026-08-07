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
#ifndef OMG_DDS_SUB_DETAIL_READ_CONDITION_HPP_
#define OMG_DDS_SUB_DETAIL_READ_CONDITION_HPP_

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{
namespace cond
{
class ReadConditionDelegate;
}
}
}
}
}

namespace dds
{
namespace sub
{
namespace cond
{

template <typename DELEGATE>
class TReadCondition;

namespace detail
{
typedef dds::sub::cond::TReadCondition<org::eclipse::cyclonedds::sub::cond::ReadConditionDelegate> ReadCondition;
}
}
}
}

#endif /* OMG_DDS_SUB_DETAIL_READ_CONDITION_HPP_ */
