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

#ifndef CYCLONEDDS_CORE_MISC_UTILS_HPP_
#define CYCLONEDDS_CORE_MISC_UTILS_HPP_

#include <dds/core/ddscore.hpp>
#include <org/eclipse/cyclonedds/core/Mutex.hpp>
#include "dds/dds.h"

#define STATUS_MASK_CONTAINS(mask,check) ((mask & check) == check)

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{

void
convertByteSeq(
        const dds::core::ByteSeq &from,
        void*& to,
        int32_t  size);

void
convertByteSeq(
        const void* from,
        const int32_t    size,
        dds::core::ByteSeq  &to);

void
convertStringSeq(
        const dds::core::StringSeq &from,
        char **&to);

void
convertStringSeq(
        char **from,
        uint32_t size,
        dds::core::StringSeq &to);

dds::core::Duration
convertDuration(
        const dds_duration_t &from);

dds_duration_t
convertDuration(
        const dds::core::Duration &from);

dds::core::Time
convertTime(
        const dds_time_t &from);

dds_time_t
convertTime(
        const dds::core::Time &from);

dds::core::status::StatusMask
convertStatusMask(
        const uint32_t from);

uint32_t
convertStatusMask(
        const dds::core::status::StatusMask &from);

static Mutex gpb_mutex;

}
}
}
}

#endif //CYCLONEDDS_CORE_MISC_UTILS_HPP_
