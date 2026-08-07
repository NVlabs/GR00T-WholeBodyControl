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

#ifndef DDS_SUB_DETAIL_QUERY_HPP_
#define DDS_SUB_DETAIL_QUERY_HPP_

#include <dds/sub/TQuery.hpp>
#include <org/eclipse/cyclonedds/sub/QueryDelegate.hpp>


namespace dds
{
namespace sub
{
namespace detail
{
typedef dds::sub::TQuery<org::eclipse::cyclonedds::sub::QueryDelegate> Query;
}
}
}


#endif /* DDS_SUB_DETAIL_QUERY_HPP_ */
