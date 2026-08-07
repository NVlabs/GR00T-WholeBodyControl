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
#ifndef OMG_DDS_PUB_DETAIL_ANYDATAWRITER_HPP_
#define OMG_DDS_PUB_DETAIL_ANYDATAWRITER_HPP_

#include <dds/pub/detail/TAnyDataWriterImpl.hpp>
#include <org/eclipse/cyclonedds/pub/AnyDataWriterDelegate.hpp>

namespace dds { namespace pub { namespace detail {
  typedef dds::pub::TAnyDataWriter<org::eclipse::cyclonedds::pub::AnyDataWriterDelegate> AnyDataWriter;
} } }

#endif /* OMG_DDS_PUB_DETAIL_ANYDATAWRITER_HPP_ */
