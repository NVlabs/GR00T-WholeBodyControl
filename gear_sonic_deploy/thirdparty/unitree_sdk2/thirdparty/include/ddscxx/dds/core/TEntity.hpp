#ifndef OMG_TDDS_CORE_ENTITY_HPP_
#define OMG_TDDS_CORE_ENTITY_HPP_

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

#include <string>

#include <dds/core/Reference.hpp>
#include <dds/core/status/Status.hpp>

namespace dds
{
namespace core
{
template <typename DELEGATE>
class TEntity;
}
}

/**
 * @brief This class is the abstract base class for all the DCPS objects.
 *
 * This class is the abstract base class for all of the DCPS objects
 * that support QoS policies a listener and a status condition:
 *
 * - dds::domain::DomainParticipant
 * - dds::sub::Subscriber
 *   - dds::sub::DataReader
 * - dds::pub::Publisher
 *   - dds::pub::DataWriter
 * - dds::topic::Topic
 *
 * In the ISO C++ PSM each DDS entity behaves like a polymorphic reference
 * in that it automatically manages its resource and it can be
 * safely assigned up and down the DDS Entity type hierarchy.
 *
 */
template <typename DELEGATE>
class dds::core::TEntity : public virtual dds::core::Reference<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TEntity, dds::core::Reference, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TEntity)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_EXPLICIT(TEntity)

public:
    /**
     * Enable entity.
     *
     * This operation enables the Entity. Entity objects can be created
     * either enabled or disabled. This is controlled by the value of
     * the dds::core::policy::EntityFactory QoS policy on the corresponding
     * factory for the Entity.
     *
     * Enabled entities are immediately activated at creation time meaning all their
     * immutable QoS settings can no longer be changed. Disabled Entities are not yet
     * activated, so it is still possible to change there immutable QoS settings. However,
     * once activated the immutable QoS settings can no longer be changed.
     *
     * Creating disabled entities can make sense when the creator of the Entity does not
     * yet know which QoS settings to apply, thus allowing another piece of code to set the
     * QoS later on.
     *
     * The default setting of dds::core::policy::EntityFactory is such that,
     * by default, it is not necessary to explicitly call enable on newly-
     * created entities.
     *
     * The enable operation is idempotent. Calling enable on an already-
     * enabled Entity does not raise exceptions and has no effect.
     *
     * If an Entity has not yet been enabled, the only operations that can be invoked on it
     * are: the ones to set, get or copy the QosPolicy settings, the ones that set (or get) the
     * listener, the ones that get the StatusCondition, the get_status_changes
     * operation (although the status of a disabled entity never changes), and the ‘factory’
     * operations that create, delete or lookup other Entities. Other operations will
     * throw the exception dds::core::NotEnabledError.
     *
     * Entities created from a factory that is disabled are created
     * disabled regardless of the setting of the dds::core::policy::EntityFactory
     * Qos policy. Calling enable on an Entity whose factory is not
     * enabled will fail and throw an dds::core::PreconditionNotMetError
     * exception.
     *
     * If the dds::core::policy::EntityFactory QoS policy has autoenable_created_entities
     * set to TRUE, the enable operation on the factory will automatically
     * enable all entities created from the factory.
     *
     * The Listeners associated with an entity are not called until
     * the entity is enabled. Conditions associated with an entity that
     * is not enabled are inactive; that is, they have a trigger_value==false
     * (dds::core::cond::Condition and dds::core::cond::WaitSet).
     *
     * eg.
     * @code{.cpp}
     * dds::domain::qos::DomainParticipantQos dpq;
     * dpq << dds::core::policy::EntityFactory::ManuallyEnable();
     * ...
     * dds::sub::DataReader<Foo::Bar> dr(dp, topic, drqos);
     * dr.enable();
     * @endcode
     *
     * @throw  dds::core::PreconditionNotMetError
     *              Entities' factory is not enabled.
     */
    void enable();

    /**
     * This operation returns a mask with the communication statuses in the Entity that
     * are “triggered”.
     *
     * This operation retrieves the list of communication statuses in the Entity
     * that are triggered. That is the set of communication statuses whose value have changed
     * since the last time the application called this operation. This operation shows
     * whether a change has occurred even when the status seems unchanged because the
     * status changed back to the original status.
     *
     * When the Entity is first created or if the Entity is not enabled, all
     * communication statuses are in the “un-triggered” state so the mask returned by the
     * operation is empty.
     *
     * The result value is a bit mask in which each bit shows which value has changed. The
     * relevant bits represent one of the following statuses:
     *  - dds::core::status::StatusMask::inconsistent_topic()
     *  - dds::core::status::StatusMask::offered_deadline_missed()
     *  - dds::core::status::StatusMask::requested_deadline_missed()
     *  - dds::core::status::StatusMask::offered_incompatible_qos()
     *  - dds::core::status::StatusMask::requested_incompatible_qos()
     *  - dds::core::status::StatusMask::sample_lost()
     *  - dds::core::status::StatusMask::sample_rejected()
     *  - dds::core::status::StatusMask::data_on_readers()
     *  - dds::core::status::StatusMask::data_available()
     *  - dds::core::status::StatusMask::liveliness_lost()
     *  - dds::core::status::StatusMask::liveliness_changed()
     *  - dds::core::status::StatusMask::publication_matched()
     *  - dds::core::status::StatusMask::subscription_matched()
     *
     * When the entity is first created, or if the entity is not enabled,
     * all communication statuses are in the untriggered state so the list
     * returned by the status_changes operation will be empty.
     *
     * Each status bit is declared as a constant and can be used in an AND operation to
     * check the status bit against the result of type StatusMask. Not all statuses are
     * relevant to all Entity objects. See the respective Listener interfaces for each
     * Entity for more information.
     *
     * The list of statuses returned by the status_changes operation refers
     * to the statuses that are triggered on the Entity itself, and does not
     * include statuses that apply to contained entities.
     *
     * @return dds::core::status::StatusMask
     *              a bit mask in which each bit shows which value has changed.
     */
    const dds::core::status::StatusMask status_changes();

    /**
     * This operation returns the InstanceHandle_t that represents the Entity.
     *
     * The relevant state of some Entity objects are distributed using built-in topics.
     * Each built-in topic sample represents the state of a specific Entity and has a
     * unique instance_handle. This operation returns the instance_handle of the
     * built-in topic sample that represents the specified Entity.<br>
     * Some Entities (dds::pub::Publisher and dds::sub::Subscriber) do not have a corresponding
     * built-in topic sample, but they still have an instance_handle that uniquely
     * identifies the Entity. The instance_handles obtained this way can also be used
     * to check whether a specific Entity is located in a specific DomainParticipant
     * (dds::domain::DomainParticipant::contains_entity()).
     *
     * @return dds::core::InstanceHandle
     *              Result value is the instance_handle of the built-in topic
     *              sample that represents the state of this Entity.
     */
    const dds::core::InstanceHandle instance_handle() const;


    /**
     * This function closes the entity and releases related resources.
     *
     * Resource management for some reference types might involve relatively heavyweight
     * operating- system resources — such as e.g., threads, mutexes, and network sockets —
     * in addition to memory. These objects therefore provide a method close() that shall
     * halt network communication (in the case of entities) and dispose of any appropriate
     * operating-system resources.
     *
     * Users of this PSM are recommended to call close on objects of all reference types once
     * they are finished using them. In addition, implementations may automatically close
     * objects that they deem to be no longer in use, subject to the following restrictions:
     * - Any object to which the application has a direct reference is still in use.
     * - Any object that has been explicitly retained is still in use
     * - The creator of any object that is still in use is itself still in use.
     */
    void close();

    /**
     * Retain the Entity, even when it goes out of scope.
     *
     * This function indicates that references to this object may go out of
     * scope but that the application expects to look it up again later.
     * Therefore the Service must consider this object to be still in use
     * and may not close it automatically.
     */
    void retain();
};



#endif /* OMG_TDDS_CORE_ENTITY_HPP_ */
