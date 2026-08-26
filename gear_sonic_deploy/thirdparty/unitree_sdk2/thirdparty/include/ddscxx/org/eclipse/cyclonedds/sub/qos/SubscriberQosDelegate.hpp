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

#ifndef CYCLONEDDS_SUB_QOS_SUBSCRIBER_QOS_DELEGATE_HPP_
#define CYCLONEDDS_SUB_QOS_SUBSCRIBER_QOS_DELEGATE_HPP_

#include <dds/core/policy/CorePolicy.hpp>

struct _DDS_NamedSubscriberQos;

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{
namespace qos
{

class OMG_DDS_API SubscriberQosDelegate
{
public:
    SubscriberQosDelegate();
    SubscriberQosDelegate(const SubscriberQosDelegate& other);

    ~SubscriberQosDelegate();

    void policy(const dds::core::policy::Presentation& presentation);
    void policy(const dds::core::policy::Partition& partition);
    void policy(const dds::core::policy::GroupData& grout_data);
    void policy(const dds::core::policy::EntityFactory& entity_factory);

    template <typename POLICY> const POLICY& policy() const;
    template <typename POLICY> POLICY& policy();

    /* The returned ddsc QoS has to be freed. */
    dds_qos_t* ddsc_qos() const;
    void ddsc_qos(const dds_qos_t* qos);

    void named_qos(const struct _DDS_NamedSubscriberQos &qos);

    void check() const;

    bool operator ==(const SubscriberQosDelegate& other) const;
    SubscriberQosDelegate& operator =(const SubscriberQosDelegate& other);

private:
    dds::core::policy::Presentation presentation_;
    dds::core::policy::Partition partition_;
    dds::core::policy::GroupData group_data_;
    dds::core::policy::EntityFactory entity_factory_;
};



//==============================================================================


template<>
inline const dds::core::policy::Presentation&
SubscriberQosDelegate::policy<dds::core::policy::Presentation>() const
{
    return presentation_;
}
template<>
inline dds::core::policy::Presentation&
SubscriberQosDelegate::policy<dds::core::policy::Presentation>()
{
    return presentation_;
}


template<>
inline const dds::core::policy::Partition&
SubscriberQosDelegate::policy<dds::core::policy::Partition>() const
{
    return partition_;
}
template<>
inline dds::core::policy::Partition&
SubscriberQosDelegate::policy<dds::core::policy::Partition>()
{
    return partition_;
}


template<>
inline const dds::core::policy::GroupData&
SubscriberQosDelegate::policy<dds::core::policy::GroupData>() const
{
    return group_data_;
}
template<>
inline dds::core::policy::GroupData&
SubscriberQosDelegate::policy<dds::core::policy::GroupData>()
{
    return group_data_;
}


template<>
inline const dds::core::policy::EntityFactory&
SubscriberQosDelegate::policy<dds::core::policy::EntityFactory>() const
{
    return entity_factory_;
}
template<>
inline dds::core::policy::EntityFactory&
SubscriberQosDelegate::policy<dds::core::policy::EntityFactory>()
{
    return entity_factory_;
}

}
}
}
}
}

#endif /* CYCLONEDDS_SUB_QOS_SUBSCRIBER_QOS_DELEGATE_HPP_ */
