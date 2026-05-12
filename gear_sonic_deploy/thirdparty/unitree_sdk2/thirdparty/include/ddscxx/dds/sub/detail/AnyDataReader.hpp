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
#ifndef CYCLONEDDS_DDS_SUB_DETAIL_ANYDATAREADER_HPP_
#define CYCLONEDDS_DDS_SUB_DETAIL_ANYDATAREADER_HPP_

#include <dds/sub/detail/TAnyDataReaderImpl.hpp>
#include <org/eclipse/cyclonedds/sub/AnyDataReaderDelegate.hpp>

namespace dds { namespace sub { namespace detail {
  typedef dds::sub::TAnyDataReader<org::eclipse::cyclonedds::sub::AnyDataReaderDelegate> AnyDataReader;
} } }

#endif /* CYCLONEDDS_DDS_SUB_DETAIL_ANYDATAREADER_HPP_ */
