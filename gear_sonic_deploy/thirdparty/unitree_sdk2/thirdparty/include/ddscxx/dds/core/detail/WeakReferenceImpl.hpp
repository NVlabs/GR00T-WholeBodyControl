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
#ifndef CYCLONEDDS_DDS_CORE_WEAK_REFERENCE_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_WEAK_REFERENCE_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/detail/ReferenceImpl.hpp>
#include <dds/core/WeakReference.hpp>


// Implementation
namespace dds
{
namespace core
{

template <typename T>
WeakReference<T>::WeakReference() { }

template <typename T>
WeakReference<T>::WeakReference(const T& t)
{
    if (!t.is_nil()) {
        impl_ = t.delegate();
    }
}

template <typename T>
bool WeakReference<T>::expired()
{
    return impl_.expired();
}

template <typename T>
T WeakReference<T>::lock()
{
    return T(impl_.lock());
}
}
}
// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_WEAK_REFERENCE_IMPL_HPP_ */
