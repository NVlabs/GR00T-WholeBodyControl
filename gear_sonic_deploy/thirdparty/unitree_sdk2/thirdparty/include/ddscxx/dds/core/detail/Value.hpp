/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_DDS_CORE_DETAIL_VALUE_HPP_
#define CYCLONEDDS_DDS_CORE_DETAIL_VALUE_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/Value.hpp>

// Implementation
namespace dds
{
namespace core
{

/**
 * @internal @todo We can't assume that the compiler supports variadic templates, yet.
 * This code should be refactored to take advantage of compilers that do support variadic
 * templates.
 */

template <typename D>
Value<D>::Value() { }

template <typename D>
template <typename ARG>
Value<D>::Value(const ARG& arg) : d_(arg) { }

template <typename D>
template <typename ARG1, typename ARG2>
Value<D>::Value(const ARG1& arg1, const ARG2& arg2) : d_(arg1, arg2) { }

template <typename D>
template <typename ARG1, typename ARG2, typename ARG3>
Value<D>::Value(const ARG1& arg1, const ARG2& arg2, const ARG3& arg3)
    : d_(arg1, arg2, arg3) { }

template <typename D>
template <typename ARG1, typename ARG2, typename ARG3, typename ARG4>
Value<D>::Value(const ARG1& arg1, const ARG2& arg2, const ARG3& arg3, const ARG4& arg4)
    : d_(arg1, arg2, arg3, arg4) { }

template <typename D>
template <typename ARG1, typename ARG2, typename ARG3, typename ARG4, typename ARG5>
Value<D>::Value(const ARG1& arg1, const ARG2& arg2, const ARG3& arg3, const ARG4& arg4, const ARG5& arg5)
    : d_(arg1, arg2, arg3, arg4, arg5) { }

template <typename D>
template <typename ARG1, typename ARG2, typename ARG3, typename ARG4, typename ARG5, typename ARG6>
Value<D>::Value(const ARG1& arg1, const ARG2& arg2, const ARG3& arg3, const ARG4& arg4, const ARG5& arg5, const ARG6& arg6)
    : d_(arg1, arg2, arg3, arg4, arg5, arg6) { }

template <typename D>
Value<D>& Value<D>::operator=(const Value& other)
{
    if(this != &other)
    {
        d_ = other.d_;
    }
    return *this;
}

template <typename D>
Value<D>& Value<D>::operator=(Value&& other)
{
    if(this != &other)
    {
        d_ = other.d_;
    }
    return *this;
}

template <typename D>
bool Value<D>::operator==(const Value& other) const
{
    return (d_ == other.d_);
}

template <typename D>
bool Value<D>::operator !=(const Value& other) const
{
    return !(d_ == other.d_);
}

template <typename D>
const D* Value<D>::operator->() const
{
    return &d_;
}

template <typename D>
D* Value<D>::operator->()
{
    return &d_;
}

template <typename D>
const D& Value<D>::delegate() const
{
    return d_;
}

template <typename D>
D& Value<D>::delegate()
{
    return d_;
}

template <typename D>
Value<D>::operator D& ()
{
    return d_;
}

template <typename D>
Value<D>::operator const D& () const
{
    return d_;
}

}
}
// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_DETAIL_VALUE_HPP_ */
