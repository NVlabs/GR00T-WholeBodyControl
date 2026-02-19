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
#ifndef CYCLONEDDS_DDS_CORE_POLICY_TCOREPOLICY_IMPL_HPP_
#define CYCLONEDDS_DDS_CORE_POLICY_TCOREPOLICY_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/core/policy/TCorePolicy.hpp>
#include <org/eclipse/cyclonedds/core/ReportUtils.hpp>

// Implementation

namespace dds
{
namespace core
{
namespace policy
{

//TUserData

template <typename D>
TUserData<D>::TUserData() : dds::core::Value<D>() { }

template <typename D>
TUserData<D>::TUserData(const dds::core::ByteSeq& sequence) : dds::core::Value<D>(sequence) { }

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
TUserData<D>::TUserData(const uint8_t* value_begin, const uint8_t* value_end)
{
    (void)value_begin;
    (void)value_end;
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
}

template <typename D>
TUserData<D>::TUserData(const TUserData& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TUserData<D>& TUserData<D>::value(const dds::core::ByteSeq& sequence)
{
    this->delegate().value(sequence);
    return *this;
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
template <typename OCTET_ITER>
TUserData<D>& TUserData<D>::value(OCTET_ITER begin, OCTET_ITER end)
{
    (void)begin;
    (void)end;
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return *this;
}

template <typename D>
const dds::core::ByteSeq TUserData<D>::value() const
{
    return this->delegate().value();
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
const uint8_t* TUserData<D>::begin() const
{
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return NULL;
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
const uint8_t* TUserData<D>::end() const
{
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return NULL;
}

//TGroupData

template <typename D>
TGroupData<D>::TGroupData() : dds::core::Value<D>() { }

template <typename D>
TGroupData<D>::TGroupData(const dds::core::ByteSeq& sequence) : dds::core::Value<D>(sequence) { }

template <typename D>
TGroupData<D>::TGroupData(const TGroupData& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TGroupData<D>::TGroupData(const uint8_t* value_begin, const uint8_t* value_end)
{
    (void)value_begin;
    (void)value_end;
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
}

template <typename D>
TGroupData<D>& TGroupData<D>::value(const dds::core::ByteSeq& sequence)
{
    this->delegate().value(sequence);
    return *this;
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
template <typename OCTET_ITER>
TGroupData<D>& TGroupData<D>::value(OCTET_ITER begin, OCTET_ITER end)
{
    (void)begin;
    (void)end;
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return *this;
}

template <typename D>
const dds::core::ByteSeq TGroupData<D>::value() const
{
    return this->delegate().value();
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
const uint8_t* TGroupData<D>::begin() const
{
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return NULL;
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
const uint8_t* TGroupData<D>::end() const
{
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return NULL;
}

//TTopicData

template <typename D>
TTopicData<D>::TTopicData() : dds::core::Value<D>() { }

template <typename D>
TTopicData<D>::TTopicData(const dds::core::ByteSeq& sequence) : dds::core::Value<D>(sequence) { }

template <typename D>
TTopicData<D>::TTopicData(const TTopicData& other) : dds::core::Value<D>(other.delegate()) { }

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
TTopicData<D>::TTopicData(const uint8_t* value_begin, const uint8_t* value_end)
{
    (void)value_begin;
    (void)value_end;
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
}

template <typename D>
TTopicData<D>& TTopicData<D>::value(const dds::core::ByteSeq& sequence)
{
    this->delegate().value(sequence);
    return *this;
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
template <typename OCTET_ITER>
TTopicData<D>& TTopicData<D>::value(OCTET_ITER begin, OCTET_ITER end)
{
    (void)begin;
    (void)end;
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return *this;
}

template <typename D>
const dds::core::ByteSeq TTopicData<D>::value() const
{
    return this->delegate().value();
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
const uint8_t* TTopicData<D>::begin() const
{
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return NULL;
}

/** @internal @bug OSPL-1746 No implementation
 * @todo Implementation required - see OSPL-1746
 * @see http://jira.prismtech.com:8080/browse/OSPL-1746 */
template <typename D>
const uint8_t* TTopicData<D>::end() const
{
    ISOCPP_THROW_EXCEPTION(ISOCPP_ERROR, "Function not currently supported");
    return NULL;
}

//TEntityFactory

template <typename D>
TEntityFactory<D>::TEntityFactory(bool autoenable_created_entities) : dds::core::Value<D>(autoenable_created_entities) { }

template <typename D>
TEntityFactory<D>::TEntityFactory(const TEntityFactory& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TEntityFactory<D>& TEntityFactory<D>::autoenable_created_entities(bool autoenable_created_entities)
{
    this->delegate().auto_enable(autoenable_created_entities);
    return *this;
}

template <typename D>
bool TEntityFactory<D>::autoenable_created_entities() const
{
    return this->delegate().auto_enable();
}

template <typename D>
TEntityFactory<D> TEntityFactory<D>::AutoEnable()
{
    return TEntityFactory(true);
}

template <typename D>
TEntityFactory<D> TEntityFactory<D>::ManuallyEnable()
{
    return TEntityFactory(false);
}

//TTransportPriority

template <typename D>
TTransportPriority<D>::TTransportPriority(int32_t priority) : dds::core::Value<D>(priority) { }

template <typename D>
TTransportPriority<D>::TTransportPriority(const TTransportPriority& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TTransportPriority<D>& TTransportPriority<D>::value(int32_t priority)
{
    this->delegate().value(priority);
    return *this;
}

template <typename D>
int32_t TTransportPriority<D>::value() const
{
    return this->delegate().value();
}

//TLifeSpan

template <typename D>
TLifespan<D>::TLifespan(const dds::core::Duration& duration) : dds::core::Value<D>(duration) { }

template <typename D>
TLifespan<D>::TLifespan(const TLifespan& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TLifespan<D>& TLifespan<D>::duration(const dds::core::Duration& duration)
{
    this->delegate().duration(duration);
    return *this;
}

template <typename D>
const dds::core::Duration TLifespan<D>::duration() const
{
    return this->delegate().duration();
}

//TDeadline

template <typename D>
TDeadline<D>::TDeadline(const dds::core::Duration& period) : dds::core::Value<D>(period) { }

template <typename D>
TDeadline<D>::TDeadline(const TDeadline& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TDeadline<D>& TDeadline<D>::period(const dds::core::Duration& period)
{
    this->delegate().period(period);
    return *this;
}

template <typename D>
const dds::core::Duration TDeadline<D>::period() const
{
    return this->delegate().period();
}

//TLatencyBudget

template <typename D>
TLatencyBudget<D>::TLatencyBudget(const dds::core::Duration& duration) : dds::core::Value<D>(duration) { }

template <typename D>
TLatencyBudget<D>::TLatencyBudget(const TLatencyBudget& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TLatencyBudget<D>& TLatencyBudget<D>::duration(const dds::core::Duration& duration)
{
    this->delegate().duration(duration);
    return *this;
}

template <typename D>
const dds::core::Duration TLatencyBudget<D>::duration() const
{
    return this->delegate().duration();
}

//TTimeBasedFilter
template <typename D>
TTimeBasedFilter<D>::TTimeBasedFilter(const dds::core::Duration& minimum_separation) : dds::core::Value<D>(minimum_separation) { }

template <typename D>
TTimeBasedFilter<D>::TTimeBasedFilter(const TTimeBasedFilter& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TTimeBasedFilter<D>& TTimeBasedFilter<D>::minimum_separation(const dds::core::Duration& minimum_separation)
{
    this->delegate().min_separation(minimum_separation);
    return *this;
}

template <typename D>
const dds::core::Duration TTimeBasedFilter<D>::minimum_separation() const
{
    return this->delegate().min_separation();
}

//TPartition

template <typename D>
TPartition<D>::TPartition(const std::string& name) : dds::core::Value<D>(name)
{

}

template <typename D>
TPartition<D>::TPartition(const dds::core::StringSeq& names) : dds::core::Value<D>(names)
{

}

template <typename D>
TPartition<D>::TPartition(const TPartition& other) : dds::core::Value<D>(other.delegate())
{

}

template <typename D>
TPartition<D>&
TPartition<D>::name(const std::string& name)
{
    this->delegate().name(name);
    return *this;
}

template <typename D>
TPartition<D>&
TPartition<D>::name(const dds::core::StringSeq& names)
{
    this->delegate().name(names);
    return *this;
}

template <typename D>
const dds::core::StringSeq
TPartition<D>::name() const
{
    return this->delegate().name();
}

#ifdef OMG_DDS_OWNERSHIP_SUPPORT

//TOwnership

template <typename D>
TOwnership<D>::TOwnership(dds::core::policy::OwnershipKind::Type kind) : dds::core::Value<D>(kind) { }

template <typename D>
TOwnership<D>::TOwnership(const TOwnership& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TOwnership<D>& TOwnership<D>::kind(dds::core::policy::OwnershipKind::Type kind)
{
    this->delegate().kind(kind);
    return *this;
}

template <typename D>
dds::core::policy::OwnershipKind::Type TOwnership<D>::kind() const
{
    return this->delegate().kind();
}

template <typename D>
TOwnership<D> TOwnership<D>::Exclusive()
{
    return TOwnership(dds::core::policy::OwnershipKind::EXCLUSIVE);
}

template <typename D>
TOwnership<D> TOwnership<D>::Shared()
{
    return TOwnership(dds::core::policy::OwnershipKind::SHARED);
}

//TOwnershipStrength
template <typename D>
TOwnershipStrength<D>::TOwnershipStrength(int32_t strength) : dds::core::Value<D>(strength) { }

template <typename D>
TOwnershipStrength<D>::TOwnershipStrength(const TOwnershipStrength& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
int32_t TOwnershipStrength<D>::value() const
{
    return this->delegate().strength();
}

template <typename D>
TOwnershipStrength<D>& TOwnershipStrength<D>::value(int32_t strength)
{
    this->delegate().strength(strength);
    return *this;
}

#endif  // OMG_DDS_OWNERSHIP_SUPPORT

//TWriterDataLifeCycle
template <typename D>
TWriterDataLifecycle<D>::TWriterDataLifecycle(bool autodispose_unregistered_instances) : dds::core::Value<D>(autodispose_unregistered_instances) { }

template <typename D>
TWriterDataLifecycle<D>::TWriterDataLifecycle(const TWriterDataLifecycle& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
bool TWriterDataLifecycle<D>::autodispose_unregistered_instances() const
{
    return this->delegate().autodispose();
}

template <typename D>
TWriterDataLifecycle<D>& TWriterDataLifecycle<D>::autodispose_unregistered_instances(bool autodispose_unregistered_instances)
{
    this->delegate().autodispose(autodispose_unregistered_instances);
    return *this;
}

template <typename D>
TWriterDataLifecycle<D> TWriterDataLifecycle<D>::AutoDisposeUnregisteredInstances()
{
    return TWriterDataLifecycle(true);
}

template <typename D>
TWriterDataLifecycle<D> TWriterDataLifecycle<D>::ManuallyDisposeUnregisteredInstances()
{
    return TWriterDataLifecycle(false);
}

//TReaderDataLifecycle
template <typename D>
TReaderDataLifecycle<D>::TReaderDataLifecycle(const dds::core::Duration& autopurge_nowriter_samples_delay, const dds::core::Duration& autopurge_disposed_samples_delay)
    : dds::core::Value<D>(autopurge_nowriter_samples_delay, autopurge_disposed_samples_delay) { }

template <typename D>
TReaderDataLifecycle<D>::TReaderDataLifecycle(const TReaderDataLifecycle& other)
    : dds::core::Value<D>(other.delegate()) { }

template <typename D>
const dds::core::Duration TReaderDataLifecycle<D>::autopurge_nowriter_samples_delay() const
{
    return this->delegate().autopurge_nowriter_samples_delay();
}

template <typename D>
TReaderDataLifecycle<D>& TReaderDataLifecycle<D>::autopurge_nowriter_samples_delay(const dds::core::Duration& autopurge_nowriter_samples_delay)
{
    this->delegate().autopurge_nowriter_samples_delay(autopurge_nowriter_samples_delay);
    return *this;
}

template <typename D>
const dds::core::Duration TReaderDataLifecycle<D>::autopurge_disposed_samples_delay() const
{
    return this->delegate().autopurge_disposed_samples_delay();
}

template <typename D>
TReaderDataLifecycle<D>& TReaderDataLifecycle<D>::autopurge_disposed_samples_delay(const dds::core::Duration& autopurge_disposed_samples_delay)
{
    this->delegate().autopurge_disposed_samples_delay(autopurge_disposed_samples_delay);
    return *this;
}

template <typename D>
TReaderDataLifecycle<D> TReaderDataLifecycle<D>::NoAutoPurgeDisposedSamples()
{
    return TReaderDataLifecycle();
}

template <typename D>
TReaderDataLifecycle<D> TReaderDataLifecycle<D>::AutoPurgeDisposedSamples(const dds::core::Duration& autopurge_disposed_samples_delay)
{
    return TReaderDataLifecycle().autopurge_disposed_samples_delay(autopurge_disposed_samples_delay);
}

//TDurability

template <typename D>
TDurability<D>::TDurability(dds::core::policy::DurabilityKind::Type kind) : dds::core::Value<D>(kind) { }

template <typename D>
TDurability<D>::TDurability(const TDurability& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TDurability<D>& TDurability<D>::kind(dds::core::policy::DurabilityKind::Type kind)
{
    this->delegate().kind(kind);
    return *this;
}

template <typename D>
dds::core::policy::DurabilityKind::Type TDurability<D>::kind() const
{
    return this->delegate().kind();
}

template <typename D>
TDurability<D> TDurability<D>::Volatile()
{
    return TDurability(dds::core::policy::DurabilityKind::VOLATILE);
}

template <typename D>
TDurability<D> TDurability<D>::TransientLocal()
{
    return TDurability(dds::core::policy::DurabilityKind::TRANSIENT_LOCAL);
}

template <typename D>
TDurability<D> TDurability<D>::Transient()
{
    return TDurability(dds::core::policy::DurabilityKind::TRANSIENT);
}

template <typename D>
TDurability<D> TDurability<D>::Persistent()
{
    return TDurability(dds::core::policy::DurabilityKind::PERSISTENT);
}

//TPresentation
template <typename D>
TPresentation<D>::TPresentation(dds::core::policy::PresentationAccessScopeKind::Type access_scope, bool coherent_access, bool ordered_access)
    : dds::core::Value<D>(access_scope, coherent_access, ordered_access) { }

template <typename D>
TPresentation<D>::TPresentation(const TPresentation& other)
    : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TPresentation<D>& TPresentation<D>::access_scope(dds::core::policy::PresentationAccessScopeKind::Type access_scope)
{
    this->delegate().access_scope(access_scope);
    return *this;
}

template <typename D>
dds::core::policy::PresentationAccessScopeKind::Type TPresentation<D>::access_scope() const
{
    return this->delegate().access_scope();
}

template <typename D>
TPresentation<D>& TPresentation<D>::coherent_access(bool coherent_access)
{
    this->delegate().coherent_access(coherent_access);
    return *this;
}

template <typename D>
bool TPresentation<D>::coherent_access() const
{
    return this->delegate().coherent_access();
}

template <typename D>
TPresentation<D>& TPresentation<D>::ordered_access(bool ordered_access)
{
    this->delegate().ordered_access(ordered_access);
    return *this;
}

template <typename D>
bool TPresentation<D>::ordered_access() const
{
    return this->delegate().ordered_access();
}

template <typename D>
TPresentation<D> TPresentation<D>::GroupAccessScope(bool coherent_access, bool ordered_access)
{
    return TPresentation(dds::core::policy::PresentationAccessScopeKind::GROUP, coherent_access, ordered_access);
}

template <typename D>
TPresentation<D> TPresentation<D>::InstanceAccessScope(bool coherent_access, bool ordered_access)
{
    return TPresentation(dds::core::policy::PresentationAccessScopeKind::INSTANCE, coherent_access, ordered_access);
}

template <typename D>
TPresentation<D> TPresentation<D>::TopicAccessScope(bool coherent_access, bool ordered_access)
{
    return TPresentation(dds::core::policy::PresentationAccessScopeKind::TOPIC, coherent_access, ordered_access);
}

//TReliability
template <typename D>
TReliability<D>::TReliability(dds::core::policy::ReliabilityKind::Type kind, const dds::core::Duration& max_blocking_time)
    :  dds::core::Value<D>(kind, max_blocking_time) { }

template <typename D>
TReliability<D>::TReliability(const TReliability& other)
    : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TReliability<D>& TReliability<D>::kind(dds::core::policy::ReliabilityKind::Type kind)
{
    this->delegate().kind(kind);
    return *this;
}

template <typename D>
dds::core::policy::ReliabilityKind::Type TReliability<D>::kind() const
{
    return this->delegate().kind();
}

template <typename D>
TReliability<D>& TReliability<D>::max_blocking_time(const dds::core::Duration& max_blocking_time)
{
    this->delegate().max_blocking_time(max_blocking_time);
    return *this;
}

template <typename D>
const dds::core::Duration TReliability<D>::max_blocking_time() const
{
    return this->delegate().max_blocking_time();
}

template <typename D>
TReliability<D> TReliability<D>::Reliable(const dds::core::Duration& max_blocking_time)
{
    return TReliability(dds::core::policy::ReliabilityKind::RELIABLE, max_blocking_time);
}

template <typename D>
TReliability<D> TReliability<D>::BestEffort(const dds::core::Duration& max_blocking_time)
{
    return TReliability(dds::core::policy::ReliabilityKind::BEST_EFFORT, max_blocking_time);
}

//TDestinationOrder

template <typename D>
TDestinationOrder<D>::TDestinationOrder(dds::core::policy::DestinationOrderKind::Type kind)
    : dds::core::Value<D>(kind) { }

template <typename D>
TDestinationOrder<D>::TDestinationOrder(const TDestinationOrder& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TDestinationOrder<D>& TDestinationOrder<D>::kind(dds::core::policy::DestinationOrderKind::Type kind)
{
    this->delegate().kind(kind);
    return *this;
}

template <typename D>
dds::core::policy::DestinationOrderKind::Type TDestinationOrder<D>::kind() const
{
    return this->delegate().kind();
}

template <typename D>
TDestinationOrder<D> TDestinationOrder<D>::SourceTimestamp()
{
    return TDestinationOrder(dds::core::policy::DestinationOrderKind::BY_SOURCE_TIMESTAMP);
}

template <typename D>
TDestinationOrder<D> TDestinationOrder<D>::ReceptionTimestamp()
{
    return TDestinationOrder(dds::core::policy::DestinationOrderKind::BY_RECEPTION_TIMESTAMP);
}

//THistory
template <typename D>
THistory<D>::THistory(dds::core::policy::HistoryKind::Type kind, int32_t depth)
    : dds::core::Value<D>(kind, depth) { }

template <typename D>
THistory<D>::THistory(const THistory& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
dds::core::policy::HistoryKind::Type THistory<D>::kind() const
{
    return this->delegate().kind();
}

template <typename D>
THistory<D>& THistory<D>::kind(dds::core::policy::HistoryKind::Type kind)
{
    this->delegate().kind(kind);
    return *this;
}

template <typename D>
int32_t THistory<D>::depth() const
{
    return this->delegate().depth();
}

template <typename D>
THistory<D>& THistory<D>::depth(int32_t depth)
{
    this->delegate().depth(depth);
    return *this;
}

template <typename D>
THistory<D> THistory<D>::KeepAll()
{
    return THistory(dds::core::policy::HistoryKind::KEEP_ALL, 1);
}

template <typename D>
THistory<D> THistory<D>::KeepLast(uint32_t depth)
{
    return THistory(dds::core::policy::HistoryKind::KEEP_LAST,
                    static_cast<int32_t>(depth));
}

//TResourceLimits
template <typename D>
TResourceLimits<D>::TResourceLimits(int32_t max_samples, int32_t max_instances, int32_t max_samples_per_instance)
    :  dds::core::Value<D>(max_samples, max_instances, max_samples_per_instance) { }

template <typename D>
TResourceLimits<D>::TResourceLimits(const TResourceLimits& other)
    : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TResourceLimits<D>& TResourceLimits<D>::max_samples(int32_t max_samples)
{
    this->delegate().max_samples(max_samples);
    return *this;
}

template <typename D>
int32_t TResourceLimits<D>::max_samples() const
{
    return this->delegate().max_samples();
}

template <typename D>
TResourceLimits<D>& TResourceLimits<D>::max_instances(int32_t max_instances)
{
    this->delegate().max_instances(max_instances);
    return *this;
}

template <typename D>
int32_t TResourceLimits<D>::max_instances() const
{
    return this->delegate().max_instances();
}

template <typename D>
TResourceLimits<D>& TResourceLimits<D>::max_samples_per_instance(int32_t max_samples_per_instance)
{
    this->delegate().max_samples_per_instance(max_samples_per_instance);
    return *this;
}

template <typename D>
int32_t TResourceLimits<D>::max_samples_per_instance() const
{
    return this->delegate().max_samples_per_instance();
}

//TLiveliness
template <typename D>
TLiveliness<D>::TLiveliness(dds::core::policy::LivelinessKind::Type kind, const dds::core::Duration& lease_duration)
    : dds::core::Value<D>(kind, lease_duration) { }

template <typename D>
TLiveliness<D>::TLiveliness(const TLiveliness& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TLiveliness<D>& TLiveliness<D>::kind(dds::core::policy::LivelinessKind::Type kind)
{
    this->delegate().kind(kind);
    return *this;
}

template <typename D>
dds::core::policy::LivelinessKind::Type TLiveliness<D>::kind() const
{
    return this->delegate().kind();
}

template <typename D>
TLiveliness<D>& TLiveliness<D>::lease_duration(const dds::core::Duration& lease_duration)
{
    this->delegate().lease_duration(lease_duration);
    return *this;
}

template <typename D>
const dds::core::Duration TLiveliness<D>::lease_duration() const
{
    return this->delegate().lease_duration();
}

template <typename D>
TLiveliness<D> TLiveliness<D>::Automatic(const dds::core::Duration& lease_duration)
{
    return TLiveliness(dds::core::policy::LivelinessKind::AUTOMATIC, lease_duration);
}

template <typename D>
TLiveliness<D> TLiveliness<D>::ManualByParticipant(const dds::core::Duration& lease_duration)
{
    return TLiveliness(dds::core::policy::LivelinessKind::MANUAL_BY_PARTICIPANT, lease_duration);
}

template <typename D>
TLiveliness<D> TLiveliness<D>::ManualByTopic(const dds::core::Duration& lease_duration)
{
    return TLiveliness(dds::core::policy::LivelinessKind::MANUAL_BY_TOPIC, lease_duration);
}


#ifdef OMG_DDS_PERSISTENCE_SUPPORT

//TDurabilityService

template <typename D>
TDurabilityService<D>::TDurabilityService(
    const dds::core::Duration& service_cleanup_delay,
    dds::core::policy::HistoryKind::Type history_kind,
    int32_t history_depth,
    int32_t max_samples,
    int32_t max_instances,
    int32_t max_samples_per_instance)
    : dds::core::Value<D>(service_cleanup_delay,
                          history_kind,
                          history_depth,
                          max_samples,
                          max_instances,
                          max_samples_per_instance) { }

template <typename D>
TDurabilityService<D>::TDurabilityService(const TDurabilityService& other)
    : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TDurabilityService<D>& TDurabilityService<D>::service_cleanup_delay(const dds::core::Duration& service_cleanup_delay)
{
    this->delegate().service_cleanup_delay(service_cleanup_delay);
    return *this;
}

template <typename D>
const dds::core::Duration TDurabilityService<D>::service_cleanup_delay() const
{
    return this->delegate().service_cleanup_delay();
}

template <typename D>
TDurabilityService<D>& TDurabilityService<D>::history_kind(dds::core::policy::HistoryKind::Type kind)
{
    this->delegate().history_kind(kind);
    return *this;
}

template <typename D>
dds::core::policy::HistoryKind::Type TDurabilityService<D>::history_kind() const
{
    return this->delegate().history_kind();
}

template <typename D>
TDurabilityService<D>& TDurabilityService<D>::history_depth(int32_t history_depth)
{
    this->delegate().history_depth(history_depth);
    return *this;
}

template <typename D>
int32_t TDurabilityService<D>::history_depth() const
{
    return this->delegate().history_depth();
}

template <typename D>
TDurabilityService<D>& TDurabilityService<D>::max_samples(int32_t max_samples)
{
    this->delegate().max_samples(max_samples);
    return *this;
}

template <typename D>
int32_t TDurabilityService<D>::max_samples() const
{
    return this->delegate().max_samples();
}

template <typename D>
TDurabilityService<D>& TDurabilityService<D>::max_instances(int32_t max_instances)
{
    this->delegate().max_instances(max_instances);
    return *this;
}

template <typename D>
int32_t TDurabilityService<D>::max_instances() const
{
    return this->delegate().max_instances();
}

template <typename D>
TDurabilityService<D>& TDurabilityService<D>::max_samples_per_instance(int32_t max_samples_per_instance)
{
    this->delegate().max_samples_per_instance(max_samples_per_instance);
    return *this;
}

template <typename D>
int32_t TDurabilityService<D>::max_samples_per_instance() const
{
    return this->delegate().max_samples_per_instance();
}

#endif  // OMG_DDS_PERSISTENCE_SUPPORT


#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

//TDataRepresentation

template <typename D>
TDataRepresentation<D>::TDataRepresentation(const dds::core::policy::DataRepresentationIdSeq& value) : dds::core::Value<D>(value) { }

template <typename D>
TDataRepresentation<D>::TDataRepresentation(const TDataRepresentation& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
const dds::core::policy::DataRepresentationIdSeq& TDataRepresentation<D>::value() const
{
    return this->delegate().value();
}

template <typename D>
TDataRepresentation<D>& TDataRepresentation<D>::value(const dds::core::policy::DataRepresentationIdSeq &value)
{
    this->delegate().value(value);
    return *this;
}

#endif  // defined(OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT)

#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

//TTypeConsistencyEnforcement

template <typename D>
TTypeConsistencyEnforcement<D>::TTypeConsistencyEnforcement(
    dds::core::policy::TypeConsistencyKind::Type kind,
    bool ignore_sequence_bounds,
    bool ignore_string_bounds,
    bool ignore_member_names,
    bool prevent_type_widening,
    bool force_type_validation) : dds::core::Value<D>(kind, ignore_sequence_bounds, ignore_string_bounds, ignore_member_names, prevent_type_widening, force_type_validation) { }

template <typename D>
TTypeConsistencyEnforcement<D>::TTypeConsistencyEnforcement(const TTypeConsistencyEnforcement& other) : dds::core::Value<D>(other.delegate()) { }

template <typename D>
TTypeConsistencyEnforcement<D>& TTypeConsistencyEnforcement<D>::kind(dds::core::policy::TypeConsistencyKind::Type kind)
{
    this->delegate().kind(kind);
    return *this;
}

template <typename D>
dds::core::policy::TypeConsistencyKind::Type TTypeConsistencyEnforcement<D>::kind() const
{
    return this->delegate().kind();
}

template <typename D>
TTypeConsistencyEnforcement<D>&  TTypeConsistencyEnforcement<D>::ignore_sequence_bounds(bool ignore_sequence_bounds)
{
  this->delegate().ignore_sequence_bounds(ignore_sequence_bounds);
  return *this;
}

template <typename D>
bool TTypeConsistencyEnforcement<D>::ignore_sequence_bounds() const
{
    return this->delegate().ignore_sequence_bounds();
}

template <typename D>
TTypeConsistencyEnforcement<D>&  TTypeConsistencyEnforcement<D>::ignore_string_bounds(bool ignore_string_bounds)
{
  this->delegate().ignore_string_bounds(ignore_string_bounds);
  return *this;
}

template <typename D>
bool TTypeConsistencyEnforcement<D>::ignore_string_bounds() const
{
    return this->delegate().ignore_string_bounds();
}

template <typename D>
TTypeConsistencyEnforcement<D>&  TTypeConsistencyEnforcement<D>::ignore_member_names(bool ignore_member_names)
{
  this->delegate().ignore_member_names(ignore_member_names);
  return *this;
}

template <typename D>
bool TTypeConsistencyEnforcement<D>::ignore_member_names() const
{
    return this->delegate().ignore_member_names();
}

template <typename D>
TTypeConsistencyEnforcement<D>&  TTypeConsistencyEnforcement<D>::prevent_type_widening(bool prevent_type_widening)
{
  this->delegate().prevent_type_widening(prevent_type_widening);
  return *this;
}

template <typename D>
bool TTypeConsistencyEnforcement<D>::prevent_type_widening() const
{
    return this->delegate().prevent_type_widening();
}

template <typename D>
TTypeConsistencyEnforcement<D>&  TTypeConsistencyEnforcement<D>::force_type_validation(bool force_type_validation)
{
  this->delegate().force_type_validation(force_type_validation);
  return *this;
}

template <typename D>
bool TTypeConsistencyEnforcement<D>::force_type_validation() const
{
    return this->delegate().force_type_validation();
}

#endif  // defined(OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT)

}
}
}

// End of implementation
// OMG PSM class declaration

#endif /* CYCLONEDDS_DDS_CORE_POLICY_TCOREPOLICY_IMPL_HPP_ */
// OMG PSM class declaration
