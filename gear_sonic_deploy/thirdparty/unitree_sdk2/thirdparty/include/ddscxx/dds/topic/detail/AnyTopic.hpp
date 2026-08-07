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
#ifndef OMG_DDS_TOPIC_DETAIL_ANY_TOPIC_HPP_
#define OMG_DDS_TOPIC_DETAIL_ANY_TOPIC_HPP_

/**
 * @file
 */

// Implementation

#include <dds/topic/detail/TAnyTopicImpl.hpp>
#include <org/eclipse/cyclonedds/topic/AnyTopicDelegate.hpp>

namespace dds { namespace topic { namespace detail {
  typedef dds::topic::TAnyTopic<org::eclipse::cyclonedds::topic::AnyTopicDelegate> AnyTopic;
} } }

// End of implementation

#endif /* OMG_DDS_TOPIC_DETAIL_ANY_TOPIC_HPP_ */
