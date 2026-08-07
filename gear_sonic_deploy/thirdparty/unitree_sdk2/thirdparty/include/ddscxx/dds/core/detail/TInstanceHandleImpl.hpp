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
#ifndef CYCLONEDDS_DDS_CORE_TINSTANCEHANDLE_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_TINSTANCEHANDLE_IMPL_HPP_

/**
 * @file
 */
#include <dds/core/detail/Value.hpp>
#include <org/eclipse/cyclonedds/core/InstanceHandleDelegate.hpp>

/*
 * OMG PSM class declaration
 */
#include <dds/core/TInstanceHandle.hpp>

// Implementation
namespace dds
{
namespace core
{

template <typename DELEGATE>
TInstanceHandle<DELEGATE>::TInstanceHandle() { }

template <typename DELEGATE>
template <typename ARG0>
TInstanceHandle<DELEGATE>::TInstanceHandle(const ARG0& arg0) : dds::core::Value<DELEGATE>(arg0) { }

template <typename DELEGATE>
TInstanceHandle<DELEGATE>::TInstanceHandle(const dds::core::null_type& nullHandle) : dds::core::Value<DELEGATE>(nullHandle) { }

template <typename DELEGATE>
TInstanceHandle<DELEGATE>::TInstanceHandle(const TInstanceHandle& other): dds::core::Value<DELEGATE>(other.delegate()) { }

template <typename DELEGATE>
TInstanceHandle<DELEGATE>::TInstanceHandle(TInstanceHandle&& other): dds::core::Value<DELEGATE>(other.delegate()) { }

template <typename DELEGATE>
TInstanceHandle<DELEGATE>& TInstanceHandle<DELEGATE>::operator=(const TInstanceHandle& that)
{
    if(this != &that)
    {
        this->delegate() = that.delegate();
    }
    return *this;
}

template <typename DELEGATE>
TInstanceHandle<DELEGATE>& TInstanceHandle<DELEGATE>::operator=(TInstanceHandle&& that)
{
    if(this != &that)
    {
        this->delegate() = that.delegate();
    }
    return *this;
}

template <typename DELEGATE>
bool TInstanceHandle<DELEGATE>::operator==(const TInstanceHandle& that) const
{
    return this->delegate() == that.delegate();
}

template <typename DELEGATE>
bool TInstanceHandle<DELEGATE>::operator<(const TInstanceHandle& that) const
{
    return this->delegate() < that.delegate();
}

template <typename DELEGATE>
bool TInstanceHandle<DELEGATE>::operator>(const TInstanceHandle& that) const
{
    return this->delegate() > that.delegate();
}

template <typename DELEGATE>
const TInstanceHandle<DELEGATE> TInstanceHandle<DELEGATE>::nil()
{
    dds::core::null_type nt;
    static TInstanceHandle nil_handle(nt);
    return nil_handle;
}

template <typename DELEGATE>
bool TInstanceHandle<DELEGATE>::is_nil() const
{
    return this->delegate().is_nil();
}
}
}

inline std::ostream& operator << (std::ostream& os, const dds::core::TInstanceHandle<org::eclipse::cyclonedds::core::InstanceHandleDelegate>& h)
{
    os << h.delegate();
    return os;
}

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_TINSTANCEHANDLE_IMPL_HPP_ */
