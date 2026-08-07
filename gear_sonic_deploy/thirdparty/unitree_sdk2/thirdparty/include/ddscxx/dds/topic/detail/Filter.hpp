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
#ifndef DDS_TOPIC_DETAIL_QUERY_HPP_
#define DDS_TOPIC_DETAIL_QUERY_HPP_

#include <dds/topic/detail/TFilterImpl.hpp>
#include <org/eclipse/cyclonedds/topic/FilterDelegate.hpp>

namespace dds {
namespace topic {
namespace detail {
typedef ::dds::topic::TFilter< ::org::eclipse::cyclonedds::topic::FilterDelegate > Filter;
}
}
}
#endif /* DDS_TOPIC_DETAIL_QUERY_HPP_ */
