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
#ifndef OMG_DDS_SUB_DETAIL_RANK_HPP
#define OMG_DDS_SUB_DETAIL_RANK_HPP

#include <dds/sub/detail/TRankImpl.hpp>
#include <org/eclipse/cyclonedds/sub/RankImpl.hpp>

namespace dds { namespace sub { namespace detail {
  typedef dds::sub::TRank< org::eclipse::cyclonedds::sub::RankImpl > Rank;
} } }

#endif /* OMG_DDS_SUB_DETAIL_RANK_HPP */
