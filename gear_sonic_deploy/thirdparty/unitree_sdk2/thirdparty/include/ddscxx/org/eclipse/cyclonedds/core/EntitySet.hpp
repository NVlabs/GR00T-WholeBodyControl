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

#ifndef CYCLONEDDS_CORE_ENTITY_SET_HPP_
#define CYCLONEDDS_CORE_ENTITY_SET_HPP_

#include <dds/core/InstanceHandle.hpp>

#include <org/eclipse/cyclonedds/core/EntityDelegate.hpp>
#include <org/eclipse/cyclonedds/core/Mutex.hpp>
#include <org/eclipse/cyclonedds/core/WeakReferenceSet.hpp>

#include <vector>
#include <memory>
#include <set>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{

class EntitySet
{
public:
    typedef std::vector<org::eclipse::cyclonedds::core::ObjectDelegate::weak_ref_type>::iterator vectorIterator;
    typedef std::vector<org::eclipse::cyclonedds::core::ObjectDelegate::weak_ref_type>           vector;

    /**
     *  @internal Inserts a EntityDelegate into the set.
     * @param entity The org::eclipse::cyclonedds::core::EntityDelegate to store
     */
    void insert(org::eclipse::cyclonedds::core::EntityDelegate& entity);

    /**
     *  @internal Erases a EntityDelegate from the set.
     * @param entity The org::eclipse::cyclonedds::core::EntityDelegate to delete
     */
    void erase(org::eclipse::cyclonedds::core::EntityDelegate& entity);

    /**
     *  @internal Check if Entity with specific handle is part of the set.
     * @param handle The dds::core::InstanceHandle to search for
     */
    bool contains(const dds::core::InstanceHandle& handle);

    /**
     *  @internal Call close() of all the Entities within the set (outside lock).
     */
    void all_close();

    /**
     *  @internal Call retain() of all the Entities within the set (outside lock).
     */
    void all_retain();

    /**
     *  @internal Call enable() of all the Entities within the set (outside lock).
     */
    void all_enable();

    /**
     *  @internal Copy internal set into a vector.
     */
    vector copy();

private:
    WeakReferenceSet<ObjectDelegate::weak_ref_type>::wset entities;
    Mutex mutex;
};

}
}
}
}

#endif /* CYCLONEDDS_CORE_ENTITY_SET_HPP_ */
