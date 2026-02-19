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


/**
 * @file
 */

#ifndef CYCLONEDDS_CORE_POLICY_POLICY_DELEGATE_HPP_
#define CYCLONEDDS_CORE_POLICY_POLICY_DELEGATE_HPP_

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4251 )
#endif

#include <dds/core/types.hpp>
#include <dds/core/LengthUnlimited.hpp>
#include <dds/core/Duration.hpp>
#include <dds/core/policy/PolicyKind.hpp>
#include <org/eclipse/cyclonedds/core/policy/ProprietaryPolicyKind.hpp>

#include <dds/dds.h>

//==============================================================================
/*
 * Unfortunately, there isn't an ddsc or builtin SubscriptionKey policy that
 * is alligned the same as the idl representation. To be able to translate it
 * as well, we have to seperately mention it...
 */
struct _DDS_SubscriptionKeyQosPolicy;


//==============================================================================
// DDS Policy Classes
namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{
namespace policy
{

//==============================================================================

class OMG_DDS_API TimeBasedFilterDelegate;

/**
 *  @internal This policy is useful for cases where a Topic is expected to have each
 * instance updated periodically. On the publishing side this setting
 * establishes a contract that the application must meet. On the subscribing
 * side the setting establishes a minimum requirement for the remote publishers
 * that are expected to supply the data values. When the Service <D4>matches<D5> a
 * DataWriter and a DataReader it checks whether the settings are compatible
 * (i.e., offered deadline period<= requested deadline period) if they are not,
 * the two entities are informed (via the listener or condition mechanism)
 * of the incompatibility of the QoS settings and communication will not occur.
 * Assuming that the reader and writer ends have compatible settings, the
 * fulfillment of this contract is monitored by the Service and the application
 * is informed of any violations by means of the proper listener or condition.
 * The value offered is considered compatible with the value requested if and
 * only if the inequality <D2>offered deadline period <= requested deadline period<D3>
 * evaluates to <D4>TRUE.<D5> The setting of the DEADLINE policy must be set
 * consistently with that of the TIME_BASED_FILTER.
 * For these two policies to be consistent the settings must be such that
 * <D2>deadline period>= minimum_separation.<D3>
 */
class OMG_DDS_API DeadlineDelegate
{
public:
    DeadlineDelegate(const DeadlineDelegate& other);
    explicit DeadlineDelegate(const dds::core::Duration& d);

    DeadlineDelegate& operator=(const DeadlineDelegate& other) = default;

    void period(const dds::core::Duration& d);
    dds::core::Duration period() const;

    bool operator ==(const DeadlineDelegate& other) const;

    void check() const;
    void check_against(const org::eclipse::cyclonedds::core::policy::TimeBasedFilterDelegate& filter) const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::Duration period_;
};

//==============================================================================

class OMG_DDS_API DestinationOrderDelegate
{

public:
    DestinationOrderDelegate(const DestinationOrderDelegate& other);
    explicit DestinationOrderDelegate(dds::core::policy::DestinationOrderKind::Type kind);

    DestinationOrderDelegate& operator=(const DestinationOrderDelegate& other) = default;

    void kind(dds::core::policy::DestinationOrderKind::Type kind);
    dds::core::policy::DestinationOrderKind::Type kind() const;

    bool operator ==(const DestinationOrderDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::policy::DestinationOrderKind::Type kind_;
};

//==============================================================================

class OMG_DDS_API DurabilityDelegate
{
public:
    DurabilityDelegate(const DurabilityDelegate& other);
    explicit DurabilityDelegate(dds::core::policy::DurabilityKind::Type kind);

    DurabilityDelegate& operator=(const DurabilityDelegate& other) = default;

    void kind(dds::core::policy::DurabilityKind::Type kind);
    dds::core::policy::DurabilityKind::Type kind() const;

    bool operator ==(const DurabilityDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

public:
    dds::core::policy::DurabilityKind::Type kind_;
};

//==============================================================================

#ifdef  OMG_DDS_PERSISTENCE_SUPPORT

class OMG_DDS_API DurabilityServiceDelegate
{
public:
    DurabilityServiceDelegate(const DurabilityServiceDelegate& other);
    DurabilityServiceDelegate(const dds::core::Duration& service_cleanup_delay,
                              dds::core::policy::HistoryKind::Type history_kind,
                              int32_t history_depth,
                              int32_t max_samples,
                              int32_t max_instances,
                              int32_t max_samples_per_instance);

    DurabilityServiceDelegate& operator=(const DurabilityServiceDelegate& other) = default;

    void service_cleanup_delay(const dds::core::Duration& d);
    const dds::core::Duration service_cleanup_delay() const;

    void history_kind(dds::core::policy::HistoryKind::Type kind);
    dds::core::policy::HistoryKind::Type history_kind() const;

    void history_depth(int32_t depth);
    int32_t history_depth() const;

    void max_samples(int32_t max_samples);
    int32_t max_samples() const;

    void max_instances(int32_t max_instances);
    int32_t max_instances() const;

    void max_samples_per_instance(int32_t max_samples_per_instance);
    int32_t max_samples_per_instance() const;

    bool operator ==(const DurabilityServiceDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::Duration cleanup_delay_;
    dds::core::policy::HistoryKind::Type history_kind_;
    int32_t history_depth_;
    int32_t max_samples_;
    int32_t max_instances_;
    int32_t max_samples_per_instance_;
};

#endif  // OMG_DDS_PERSISTENCE_SUPPORT

//==============================================================================

/**
 *  @internal This policy controls the behavior of the Entity as a factory for other
 * entities. This policy concerns only DomainParticipant (as factory for
 * Publisher, Subscriber, and Topic), Publisher (as factory for DataWriter),
 * and Subscriber (as factory for DataReader). This policy is mutable.
 * A change in the policy affects only the entities created after the change;
 * not the previously created entities.
 * The setting of autoenable_created_entities to TRUE indicates that the
 * newly created object will be enabled by default.
 * A setting of FALSE indicates that the Entity will not be automatically
 * enabled. The application will need to enable it explicitly by means of the
 * enable operation (see Section 7.1.2.1.1.7, <D2>enable<D3>). The default setting
 * of autoenable_created_entities = TRUE means that, by default, it is not
 * necessary to explicitly call enable on newly created entities.
 */
class OMG_DDS_API EntityFactoryDelegate
{
public:
    EntityFactoryDelegate(const EntityFactoryDelegate& other);
    explicit EntityFactoryDelegate(bool auto_enable);

    void auto_enable(bool on);
    bool auto_enable() const;

    bool operator ==(const EntityFactoryDelegate& other) const;

    EntityFactoryDelegate& operator =(const EntityFactoryDelegate& other) = default;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    bool auto_enable_;
};

//==============================================================================

/**
 *  @internal The purpose of this QoS is to allow the application to attach additional
 * information to the created Publisher or Subscriber.
 * The value of the GROUP_DATA is available to the application on the
 * DataReader and DataWriter entities and is propagated by means of the
 * built-in topics. This QoS can be used by an application combination with
 * the DataReaderListener and DataWriterListener to implement matching policies
 * similar to those of the PARTITION QoS except the decision can be made based
 * on an application-defined policy.
 */
class OMG_DDS_API GroupDataDelegate
{
public:
    /**
     *  @internal Create a <code>GroupData</code> instance.
     */
    GroupDataDelegate();

    /**
     *  @internal Create a <code>GroupData</code> instance.
     *
     * @param other the group data to copy
     */
    GroupDataDelegate(const GroupDataDelegate& other);

    /**
     *  @internal Copies a <code>GroupData</code> instance.
     *
     * @param other the group data to copy
     *
     * @return reference to the instance which was copied to
     */
    GroupDataDelegate& operator=(const GroupDataDelegate& other) = default;

    /**
     *  @internal Create a <code>GroupData</code> instance.
     *
     * @param seq the group data value
     */
    explicit GroupDataDelegate(const dds::core::ByteSeq& seq);

    /**
     *  @internal Set the value for this <code>GroupData</code>
     *
     * @param seq the group data value
     */
    void value(const dds::core::ByteSeq& seq);

    /**
     *  @internal Get the value for this <code>GroupData</code>
     *
     * @return  the group data value
     */
    const dds::core::ByteSeq& value() const;

    bool operator ==(const GroupDataDelegate& other) const;

    /**
     *  @internal Check if policy is consistent.
     */
    void check() const;

    /**
     *  @internal Set internals by the ddsc policy.
     *
     * @param qos the ddsc policy
     */
    void set_iso_policy(const dds_qos_t* qos);
    /**
      *  @internal Set internals by the ddsc policy.
      *
      * @param the ddsc policy
      */
    //@todo void v_policy(const v_builtinGroupDataPolicy& policy);

    /**
     *  @internal Get the ddsc policy representation.
     *
     * @param qos the ddsc policy
     */
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::ByteSeq value_;
};

//==============================================================================

class OMG_DDS_API ResourceLimitsDelegate;

class OMG_DDS_API HistoryDelegate
{
public:
    HistoryDelegate(const HistoryDelegate& other);
    HistoryDelegate(dds::core::policy::HistoryKind::Type kind, int32_t depth);

    HistoryDelegate& operator=(const HistoryDelegate& other) = default;

    dds::core::policy::HistoryKind::Type kind() const;
    void kind(dds::core::policy::HistoryKind::Type kind);

    int32_t depth() const;
    void depth(int32_t depth);

    bool operator ==(const HistoryDelegate& other) const;

    void check() const;
    void check_against(const org::eclipse::cyclonedds::core::policy::ResourceLimitsDelegate& limits) const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::policy::HistoryKind::Type kind_;
    int32_t depth_;
};

//==============================================================================

class OMG_DDS_API LatencyBudgetDelegate
{
public:
    LatencyBudgetDelegate(const LatencyBudgetDelegate& other);
    explicit LatencyBudgetDelegate(const dds::core::Duration& d);

    LatencyBudgetDelegate& operator=(const LatencyBudgetDelegate& other) = default;

    void duration(const dds::core::Duration& d);
    const dds::core::Duration duration() const;

    bool operator ==(const LatencyBudgetDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::Duration duration_;
};

//==============================================================================

/**
 *  @internal The purpose of this QoS is to avoid delivering <D2>stale<D3> data to the
 * application. Each data sample written by the DataWriter has an associated
 * expiration time beyond which the data should not be delivered to any
 * application. Once the sample expires, the data will be removed from the
 * DataReader caches as well as from the transient and persistent
 * information caches. The expiration time of each sample is computed by
 * adding the duration specified by the LIFESPAN QoS to the source timestamp.
 * As described in Section 7.1.2.4.2.11, <D2>write and Section 7.1.2.4.2.12,
 * write_w_timestamp the source timestamp is either automatically computed by
 * the Service each time the DataWriter write operation is called, or else
 * supplied by the application by means of the write_w_timestamp operation.
 *
 * This QoS relies on the sender and receiving applications having their clocks
 * sufficiently synchronized. If this is not the case and the Service can
 * detect it, the DataReader is allowed to use the reception timestamp instead
 * of the source timestamp in its computation of the expiration time.
 */
class OMG_DDS_API LifespanDelegate
{
public:
    LifespanDelegate(const LifespanDelegate& other);
    explicit LifespanDelegate(const dds::core::Duration& d);
    LifespanDelegate& operator=(const LifespanDelegate& other) = default;

    void duration(const dds::core::Duration& d);
    const dds::core::Duration duration() const;

    bool operator ==(const LifespanDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::Duration duration_;
};

//==============================================================================

class OMG_DDS_API LivelinessDelegate
{
public:
public:
    LivelinessDelegate(const LivelinessDelegate& other);
    LivelinessDelegate(dds::core::policy::LivelinessKind::Type kind,
                       dds::core::Duration lease_duration);

    LivelinessDelegate& operator=(const LivelinessDelegate& other) = default;

    void kind(dds::core::policy::LivelinessKind::Type kind);
    dds::core::policy::LivelinessKind::Type kind() const;

    void lease_duration(const dds::core::Duration& lease_duration);
    const dds::core::Duration lease_duration() const;

    bool operator ==(const LivelinessDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::policy::LivelinessKind::Type kind_;
    dds::core::Duration lease_duration_;
};

//==============================================================================

class OMG_DDS_API OwnershipDelegate
{
public:
    OwnershipDelegate(const OwnershipDelegate& other);
    explicit OwnershipDelegate(dds::core::policy::OwnershipKind::Type kind);
    OwnershipDelegate& operator=(const OwnershipDelegate& other) = default;

    void kind(dds::core::policy::OwnershipKind::Type kind);
    dds::core::policy::OwnershipKind::Type kind() const;

    bool operator ==(const OwnershipDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::policy::OwnershipKind::Type kind_;
};

//==============================================================================

#ifdef  OMG_DDS_OWNERSHIP_SUPPORT

class OMG_DDS_API OwnershipStrengthDelegate
{
public:
    OwnershipStrengthDelegate(const OwnershipStrengthDelegate& other);
    explicit OwnershipStrengthDelegate(int32_t s);

    OwnershipStrengthDelegate& operator=(const OwnershipStrengthDelegate& other) = default;

    int32_t strength() const;
    void strength(int32_t s);

    bool operator ==(const OwnershipStrengthDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    int32_t strength_;
};

#endif  // OMG_DDS_OWNERSHIP_SUPPORT

//==============================================================================

class OMG_DDS_API PartitionDelegate
{
public:
    PartitionDelegate(const PartitionDelegate& other);
    explicit PartitionDelegate(const std::string& partition);
    explicit PartitionDelegate(const dds::core::StringSeq& partitions);

    PartitionDelegate& operator=(const PartitionDelegate& other) = default;

    void name(const std::string& partition);
    void name(const dds::core::StringSeq& partitions);
    const dds::core::StringSeq& name() const;

    bool operator ==(const PartitionDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    //@todo void v_policy(const v_builtinPartitionPolicy& policy);

    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::StringSeq name_;
};

//==============================================================================

class OMG_DDS_API PresentationDelegate
{
public:
    PresentationDelegate(const PresentationDelegate& other);
    PresentationDelegate(dds::core::policy::PresentationAccessScopeKind::Type access_scope,
                         bool coherent_access,
                         bool ordered_access);
    PresentationDelegate& operator=(const PresentationDelegate& other) = default;

    void access_scope(dds::core::policy::PresentationAccessScopeKind::Type as);
    dds::core::policy::PresentationAccessScopeKind::Type access_scope() const;

    void coherent_access(bool on);
    bool coherent_access() const;

    void ordered_access(bool on);
    bool ordered_access() const;

    bool operator ==(const PresentationDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::policy::PresentationAccessScopeKind::Type access_scope_;
    bool coherent_access_;
    bool ordered_access_;
};

//==============================================================================

class OMG_DDS_API ReaderDataLifecycleDelegate
{
public:
    ReaderDataLifecycleDelegate(const ReaderDataLifecycleDelegate& other);
    ReaderDataLifecycleDelegate(const dds::core::Duration& nowriter_delay,
                                const dds::core::Duration& disposed_samples_delay);

    const dds::core::Duration autopurge_nowriter_samples_delay() const;
    void autopurge_nowriter_samples_delay(const dds::core::Duration& d);

    const dds::core::Duration autopurge_disposed_samples_delay() const;
    void autopurge_disposed_samples_delay(const dds::core::Duration& d);

    bool operator ==(const ReaderDataLifecycleDelegate& other) const;

    ReaderDataLifecycleDelegate& operator =(const ReaderDataLifecycleDelegate& other) = default;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::Duration autopurge_nowriter_samples_delay_;
    dds::core::Duration autopurge_disposed_samples_delay_;
};

//==============================================================================

class OMG_DDS_API ReaderLifespanDelegate
{
public:
    ReaderLifespanDelegate(const ReaderLifespanDelegate& other);
    explicit ReaderLifespanDelegate(bool used,
                              const dds::core::Duration& d);

    void duration(const dds::core::Duration& d);
    const dds::core::Duration duration() const;

    void used(bool v);
    bool used() const;

    bool operator ==(const ReaderLifespanDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    bool used_;
    dds::core::Duration duration_;
};

//==============================================================================

class OMG_DDS_API ReliabilityDelegate
{
public:
public:
    ReliabilityDelegate(const ReliabilityDelegate& other);
    ReliabilityDelegate(dds::core::policy::ReliabilityKind::Type kind,
                        const dds::core::Duration& max_blocking_time);

    ReliabilityDelegate& operator=(const ReliabilityDelegate& other) = default;

    void kind(dds::core::policy::ReliabilityKind::Type kind);
    dds::core::policy::ReliabilityKind::Type kind() const;

    void max_blocking_time(const dds::core::Duration& d);
    const dds::core::Duration max_blocking_time() const;

    bool operator ==(const ReliabilityDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::policy::ReliabilityKind::Type kind_;
    dds::core::Duration max_blocking_time_;
};

//==============================================================================

class OMG_DDS_API ResourceLimitsDelegate
{
public:
    ResourceLimitsDelegate(const ResourceLimitsDelegate& other);
    ResourceLimitsDelegate(int32_t max_samples,
                           int32_t max_instances,
                           int32_t max_samples_per_instance);

    ResourceLimitsDelegate& operator=(const ResourceLimitsDelegate& other) = default;

    void max_samples(int32_t samples);
    int32_t max_samples() const;

    void max_instances(int32_t max_instances);
    int32_t max_instances() const;

    void max_samples_per_instance(int32_t max_samples_per_instance);
    int32_t max_samples_per_instance() const;

    bool operator ==(const ResourceLimitsDelegate& other) const;

    void check() const;
    void check_against(const org::eclipse::cyclonedds::core::policy::HistoryDelegate& history) const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    int32_t max_samples_;
    int32_t max_instances_;
    int32_t max_samples_per_instance_;
};

//==============================================================================

class OMG_DDS_API TimeBasedFilterDelegate
{
public:
    TimeBasedFilterDelegate(const TimeBasedFilterDelegate& other);
    explicit TimeBasedFilterDelegate(const dds::core::Duration& min_separation);
    TimeBasedFilterDelegate& operator=(const TimeBasedFilterDelegate& other) = default;

    void min_separation(const dds::core::Duration& ms);
    const dds::core::Duration min_separation() const;

    bool operator ==(const TimeBasedFilterDelegate& other) const;

    void check() const;
    void check_against(const org::eclipse::cyclonedds::core::policy::DeadlineDelegate& deadline) const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::Duration min_sep_;
};

//==============================================================================

/**
 *  @internal The purpose of this QoS is to allow the application to attach additional
 * information to the created Topic such that when a remote application
 * discovers their existence it can examine the information and use it in
 * an application-defined way. In combination with the listeners on the
 * DataReader and DataWriter as well as by means of operations such as
 * ignore_topic, these QoS can assist an application to extend the provided QoS.
 */
class OMG_DDS_API TopicDataDelegate
{
public:
    TopicDataDelegate();
    TopicDataDelegate(const TopicDataDelegate& other);
    explicit TopicDataDelegate(const dds::core::ByteSeq& seq);

    TopicDataDelegate& operator=(const TopicDataDelegate& other) = default;

    void value(const dds::core::ByteSeq& seq);
    const dds::core::ByteSeq& value() const;

    bool operator ==(const TopicDataDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    //@todo void v_policy(const v_builtinTopicDataPolicy& policy);

    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::ByteSeq value_;

};

//==============================================================================

/**
 *  @internal The purpose of this QoS is to allow the application to take advantage of
 * transports capable of sending messages with different priorities.
 * This policy is considered a hint. The policy depends on the ability of the
 * underlying transports to set a priority on the messages they send.
 * Any value within the range of a 32-bit signed integer may be chosen;
 * higher values indicate higher priority. However, any further interpretation
 * of this policy is specific to a particular transport and a particular
 * implementation of the Service. For example, a particular transport is
 * permitted to treat a range of priority values as equivalent to one another.
 * It is expected that during transport configuration the application would
 * provide a mapping between the values of the TRANSPORT_PRIORITY set on
 * DataWriter and the values meaningful to each transport. This mapping would
 * then be used by the infrastructure when propagating the data written by
 * the DataWriter.
 */
class OMG_DDS_API TransportPriorityDelegate
{
public:
    TransportPriorityDelegate(const TransportPriorityDelegate& other);
    TransportPriorityDelegate& operator=(const TransportPriorityDelegate& other) = default;
    explicit TransportPriorityDelegate(int32_t prio);

    void value(int32_t prio);
    int32_t value() const;

    bool operator ==(const TransportPriorityDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    int32_t value_;
};

//==============================================================================

/**
 *  @internal The purpose of this QoS is to allow the application to attach additional
 * information to the created Entity objects such that when a remote application
 * discovers their existence it can access that information and use it for its
 * own purposes. One possible use of this QoS is to attach security credentials
 * or some other information that can be used by the remote application to
 * authenticate the source. In combination with operations such as
 * ignore_participant, ignore_publication, ignore_subscription,
 * and ignore_topic these QoS can assist an application to define and enforce
 * its own security policies. The use of this QoS is not limited to security,
 * rather it offers a simple, yet flexible extensibility mechanism.
 */
class OMG_DDS_API UserDataDelegate
{
public:
    /**
     *  @internal Create a <code>UserData</code> instance with an empty user data.
     */
    UserDataDelegate();

    /**
     *  @internal Create a <code>UserData</code> instance.
     *
     * @param other the user data to copy
     */
    UserDataDelegate(const UserDataDelegate& other);

    /**
     *  @internal Copies a <code>UserData</code> instance.
     *
     * @param other the user data to copy
     *
     * @return reference to the instance that was copied to
     */
    UserDataDelegate& operator=(const UserDataDelegate& other) = default;

    /**
     *  @internal Create a <code>UserData</code> instance.
     *
     * @param seq the sequence of octet representing the user data
     */
    explicit UserDataDelegate(const dds::core::ByteSeq& seq);

    /**
     *  @internal Set the value for the user data.
     *
     * @param seq a sequence of octet representing the user data.
     */
    void value(const dds::core::ByteSeq& seq);

    /**
     *  @internal Get the user data.
     *
     * @return the sequence of octet representing the user data
     */
    const dds::core::ByteSeq value() const;

    bool operator ==(const UserDataDelegate& other) const;

    /**
     *  @internal Check if policy is consistent.
     */
    void check() const;

    /**
     *  @internal Set internals by the ddsc policy.
     *
     * @param qos the ddsc policy
     */
    void set_iso_policy(const dds_qos_t* qos);
    /**
     *  @internal Set internals by the ddsc policy.
     *
     * @param the ddsc policy
     */
    //@todo void v_policy(const v_builtinUserDataPolicy& policy);
    /**
     *  @internal Get the ddsc policy representation.
     */
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::ByteSeq value_;
};

//==============================================================================

class OMG_DDS_API WriterDataLifecycleDelegate
{
public:
    WriterDataLifecycleDelegate(const WriterDataLifecycleDelegate& other);
    explicit WriterDataLifecycleDelegate(bool autodispose);

    bool autodispose() const;
    void autodispose(bool b);

    bool operator ==(const WriterDataLifecycleDelegate& other) const;

    WriterDataLifecycleDelegate& operator =(const WriterDataLifecycleDelegate& other) = default;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    bool autodispose_;
};



#ifdef  OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

class OMG_DDS_API DataRepresentationDelegate
{
public:
    DataRepresentationDelegate(const DataRepresentationDelegate& other);
    explicit DataRepresentationDelegate(const dds::core::policy::DataRepresentationIdSeq& value);
    DataRepresentationDelegate& operator=(const DataRepresentationDelegate& other) = default;

    void value(const dds::core::policy::DataRepresentationIdSeq &value);
    const dds::core::policy::DataRepresentationIdSeq& value() const;

    bool operator ==(const DataRepresentationDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::policy::DataRepresentationIdSeq value_;
};

#endif  // OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

#ifdef  OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

class OMG_DDS_API TypeConsistencyEnforcementDelegate {
public:
    TypeConsistencyEnforcementDelegate(const TypeConsistencyEnforcementDelegate& other);
    explicit TypeConsistencyEnforcementDelegate(dds::core::policy::TypeConsistencyKind kind,
                                                bool ignore_sequence_bounds,
                                                bool ignore_string_bounds,
                                                bool ignore_member_names,
                                                bool prevent_type_widening,
                                                bool force_type_validation);
    TypeConsistencyEnforcementDelegate& operator=(const TypeConsistencyEnforcementDelegate& other) = default;

    void kind(dds::core::policy::TypeConsistencyKind kind);
    dds::core::policy::TypeConsistencyKind kind() const;

    void ignore_sequence_bounds(bool ignore_sequence_bounds);
    bool ignore_sequence_bounds() const;

    void ignore_string_bounds(bool ignore_string_bounds);
    bool ignore_string_bounds() const;

    void ignore_member_names(bool ignore_member_names);
    bool ignore_member_names() const;

    void prevent_type_widening(bool prevent_type_widening);
    bool prevent_type_widening() const;

    void force_type_validation(bool force_type_validation);
    bool force_type_validation() const;

    bool operator ==(const TypeConsistencyEnforcementDelegate& other) const;

    void check() const;

    void set_iso_policy(const dds_qos_t* qos);
    void set_c_policy(dds_qos_t* qos) const;

private:
    dds::core::policy::TypeConsistencyKind kind_;
    bool ignore_sequence_bounds_;
    bool ignore_string_bounds_;
    bool ignore_member_names_;
    bool prevent_type_widening_;
    bool force_type_validation_;
};

#endif  // OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

}
}
}
}
}  // namespace org::eclipse::cyclonedds::core::policy

#endif /* CYCLONEDDS_CORE_POLICY_POLICY_DELEGATE_HPP_ */
