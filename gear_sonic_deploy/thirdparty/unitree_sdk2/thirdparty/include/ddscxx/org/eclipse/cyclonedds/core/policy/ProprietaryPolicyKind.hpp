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


/**
 * @file
 */

#ifndef CYCLONEDDS_CORE_POLICY_PROPRIETARYPOLICYKIND_HPP_
#define CYCLONEDDS_CORE_POLICY_PROPRIETARYPOLICYKIND_HPP_


#include <dds/core/SafeEnumeration.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{
namespace policy
{

/*
 * Proprietary policy values
 */
struct InvalidSampleVisibility_def
{
    enum Type
    {
        NO_INVALID_SAMPLES,
        MINIMUM_INVALID_SAMPLES,
        ALL_INVALID_SAMPLES
    };
};
typedef dds::core::safe_enum<InvalidSampleVisibility_def> InvalidSampleVisibility;

struct SchedulingKind_def
{
    enum Type
    {
        SCHEDULE_DEFAULT,
        SCHEDULE_TIMESHARING,
        SCHEDULE_REALTIME
    };
};
typedef dds::core::safe_enum<SchedulingKind_def> SchedulingKind;

struct SchedulingPriorityKind_def
{
    enum Type
    {
        PRIORITY_RELATIVE,
        PRIORITY_ABSOLUTE
    };
};
typedef dds::core::safe_enum<SchedulingPriorityKind_def> SchedulingPriorityKind;

}
}
}
}
}

#endif /* CYCLONEDDS_CORE_POLICY_PROPRIETARYPOLICYKIND_HPP_ */
