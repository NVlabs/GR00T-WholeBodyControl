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
#ifndef CYCLONEDDS_DDS_CORE_DETAIL_REF_TRAITS_HPP_
#define CYCLONEDDS_DDS_CORE_DETAIL_REF_TRAITS_HPP_

/**
 * @file
 */

// Implementation

#include <memory>
#include <type_traits>

#include <dds/core/types.hpp>
#include <dds/core/Exception.hpp>

template <typename T1, typename T2>
struct dds::core::is_base_of : public ::std::is_base_of<T1, T2> { };

template <typename T1, typename T2>
struct dds::core::is_same : public ::std::is_same<T1, T1> { };

template <typename T>
struct dds::core::smart_ptr_traits
{
    typedef ::std::shared_ptr<T> ref_type;
    typedef ::std::weak_ptr<T>   weak_ref_type;
};

template <typename TO, typename FROM>
TO dds::core::polymorphic_cast(FROM& from)
{
    typename TO::DELEGATE_REF_T dr =
        ::std::dynamic_pointer_cast< typename TO::DELEGATE_T>(from.delegate());
    TO to(dr);

    if(to == dds::core::null)
    {
        throw dds::core::InvalidDowncastError("Attempted invalid downcast.");
    }
    return to;
}

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_DETAIL_REF_TRAITS_HPP_ */
