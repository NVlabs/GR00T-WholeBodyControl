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

#ifndef CYCLONEDDS_DDS_TOPIC_DETAIL_FIND_HPP_
#define CYCLONEDDS_DDS_TOPIC_DETAIL_FIND_HPP_


#include <org/eclipse/cyclonedds/topic/find.hpp>


#include <string>


namespace dds
{
namespace topic
{

template <typename TOPIC>
TOPIC
find(const dds::domain::DomainParticipant& dp, const std::string& topic_name)
{
    TOPIC t = org::eclipse::cyclonedds::topic::finder<TOPIC, typename TOPIC::DELEGATE_T>::find(dp, topic_name);

    return t;
}


}
}

#endif /* CYCLONEDDS_DDS_TOPIC_DETAIL_FIND_HPP_ */
