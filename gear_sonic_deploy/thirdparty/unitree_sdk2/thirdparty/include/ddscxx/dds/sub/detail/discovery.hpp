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

#ifndef CYCLONEDDS_DDS_SUB_DETAIL_DISCOVERY_HPP_
#define CYCLONEDDS_DDS_SUB_DETAIL_DISCOVERY_HPP_

#include <dds/sub/DataReader.hpp>

namespace dds
{
namespace sub
{

template <typename FwdIterator>
void ignore(const dds::domain::DomainParticipant& dp, FwdIterator begin, FwdIterator end)
{
    (void)dp;
    (void)begin;
    (void)end;
    ISOCPP_THROW_EXCEPTION(ISOCPP_UNSUPPORTED_ERROR, "Function not currently supported");
}

template <typename T>
::dds::core::InstanceHandleSeq
 matched_publications(const dds::sub::DataReader<T>& dr)
{
    return dr.delegate()->matched_publications();
}

template <typename T, typename FwdIterator>
uint32_t
matched_publications(const dds::sub::DataReader<T>& dr,
                      FwdIterator begin, uint32_t max_size)
{
    return dr.delegate()->matched_publications(begin, max_size);
}

template <typename T>
const dds::topic::PublicationBuiltinTopicData
matched_publication_data(const dds::sub::DataReader<T>& dr,
                          const ::dds::core::InstanceHandle& h)
{
    return dr.delegate()->matched_publication_data(h);
}

}
}
#endif /* CYCLONEDDS_DDS_SUB_DETAIL_DISCOVERY_HPP_ */
