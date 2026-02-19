#ifndef OMG_TDDS_CORE_INSTANCE_HANDLE_HPP_
#define OMG_TDDS_CORE_INSTANCE_HANDLE_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Corp.
 * Copyright 2010, Real-Time Innovations, Inc.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dds/core/types.hpp>
#include <dds/core/Value.hpp>


namespace dds
{
namespace core
{
template <typename DELEGATE>
class TInstanceHandle;
}
}

/**
 * @brief
 * Class to hold the handle associated with in sample instance.
 */
template <typename DELEGATE>
class dds::core::TInstanceHandle : public dds::core::Value<DELEGATE>
{
public:
    /**
     * Create an nil instance handle.
     */
    TInstanceHandle();

    /**
     * Create an nil instance handle.
     *
     * @param nullHandle placeholder
     */
    TInstanceHandle(const dds::core::null_type& nullHandle);

    /**
     * Copy an existing InstanceHandle
     *
     * @param other InstanceHandle to copy
     */
    TInstanceHandle(const TInstanceHandle& other);

    /**
     * Move an existing InstanceHandle
     *
     * @param other InstanceHandle to move
     */
    TInstanceHandle(TInstanceHandle&& other);

    /** @cond */
    ~TInstanceHandle() = default;
    /** @endcond */

    /**
     * @cond
     * Parametric constructor for creating an instance-handle
     * from some other type. This function is intended for internal
     * usage.
     */
    template <typename ARG0>
    TInstanceHandle(const ARG0& arg0);
    /** @endcond */

public:
    /**
     * Copy-assign an existing InstanceHandle to this InstanceHandle
     *
     * @param that The TInstanceHandle to assign to this
     */
    TInstanceHandle& operator=(const TInstanceHandle& that);

    /**
     * Move-assign an existing InstanceHandle to this InstanceHandle
     *
     * @param that The TInstanceHandle to assign to this
     */
    TInstanceHandle& operator=(TInstanceHandle&& that);

    /**
     * Compare this InstanceHandle to another InstanceHandle
     *
     * @param that The TInstanceHandle to compare
     * @return true if they match
     */
    bool operator==(const TInstanceHandle& that) const;

    /**
     * Compare this InstanceHandle to another InstanceHandle
     *
     * @param that The TInstanceHandle to compare
     * @return true if this is less than that
     */
    bool operator<(const TInstanceHandle& that) const;

    /**
     * Compare this InstanceHandle to another InstanceHandle
     *
     * @param that The TInstanceHandle to compare
     * @return true if this is greater than that
     */
    bool operator>(const TInstanceHandle& that) const;

public:
    /**
     * Create an nil instance handle.
     *
     * @return a nil InstanceHandle
     */
    static const TInstanceHandle nil();

    /**
     * Check if the InstanceHandle is nil.
     *
     * @return true if the InstanceHandle is nil
     */
    bool is_nil() const;
};


#endif // !defined(OMG_TDDS_CORE_INSTANCE_HANDLE_HPP_)
