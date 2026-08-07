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

#ifndef CYCLONEDDS_DDS_PUB_DETAIL_FIND_HPP_
#define CYCLONEDDS_DDS_PUB_DETAIL_FIND_HPP_

#include <string>

#include <dds/pub/DataWriter.hpp>
#include <dds/pub/Publisher.hpp>
#include <org/eclipse/cyclonedds/pub/PublisherDelegate.hpp>
#include <org/eclipse/cyclonedds/pub/AnyDataWriterDelegate.hpp>

namespace dds
{
namespace pub
{

template <typename WRITER, typename FwdIterator>
uint32_t
find(const dds::pub::Publisher& pub, const std::string& topic_name,
     FwdIterator begin, uint32_t max_size)
{
    if(max_size > 0) {
        org::eclipse::cyclonedds::pub::AnyDataWriterDelegate::ref_type writer_base = pub.delegate()->find_datawriter(topic_name);
        if (writer_base) {
            /* Cast base writer to typed delegate: */
            typename WRITER::DELEGATE_REF_T writer_typed =
                    ::std::dynamic_pointer_cast<typename WRITER::DELEGATE_T>(writer_base);
            WRITER dw(writer_typed);
            if(dw != dds::core::null)
            {
                *begin = dw;
                return 1;
            }
        }
    }
    return 0;
}


template <typename WRITER, typename BinIterator>
uint32_t
find(const dds::pub::Publisher& pub,
     const std::string& topic_name,
     BinIterator begin)
{
    org::eclipse::cyclonedds::pub::AnyDataWriterDelegate::ref_type writer_base = pub.delegate()->find_datawriter(topic_name);
    if (writer_base) {
        /* Cast base writer to typed delegate: */
        typename WRITER::DELEGATE_REF_T writer_typed =
                ::std::dynamic_pointer_cast<typename WRITER::DELEGATE_T>(writer_base);
        WRITER dw(writer_typed);
        if(dw != dds::core::null)
        {
            *begin = dw;
            return 1;
        }
    }
    return 0;
}


}
}

#endif /* CYCLONEDDS_DDS_PUB_DETAIL_FIND_HPP_ */
