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

#ifndef CYCLONEDDS_PUB_QOS_PUBLISHER_QOS_DELEGATE_HPP_
#define CYCLONEDDS_PUB_QOS_PUBLISHER_QOS_DELEGATE_HPP_

#include <dds/core/policy/CorePolicy.hpp>

struct _DDS_NamedPublisherQos;

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace pub
{
namespace qos
{

class OMG_DDS_API PublisherQosDelegate
{
public:
    PublisherQosDelegate();
    PublisherQosDelegate(const PublisherQosDelegate& other);

    ~PublisherQosDelegate();

    void policy(const dds::core::policy::Presentation& presentation);
    void policy(const dds::core::policy::Partition& partition);
    void policy(const dds::core::policy::GroupData& gdata);
    void policy(const dds::core::policy::EntityFactory& factory_policy);

    template <typename POLICY> const POLICY& policy() const;
    template <typename POLICY> POLICY& policy();

    /* The returned ddsc QoS has to be freed. */
    dds_qos_t* ddsc_qos() const;
    void ddsc_qos(const dds_qos_t* qos);

    void named_qos(const struct _DDS_NamedPublisherQos &qos);

    void check() const;

    bool operator ==(const PublisherQosDelegate& other) const;
    PublisherQosDelegate& operator =(const PublisherQosDelegate& other);

private:
    dds::core::policy::Presentation    presentation_;
    dds::core::policy::Partition       partition_;
    dds::core::policy::GroupData       gdata_;
    dds::core::policy::EntityFactory   factory_policy_;
};



//==============================================================================


template<>
inline const dds::core::policy::Presentation&
PublisherQosDelegate::policy<dds::core::policy::Presentation>() const
{
    return presentation_;
}
template<>
inline dds::core::policy::Presentation&
PublisherQosDelegate::policy<dds::core::policy::Presentation>()
{
    return presentation_;
}


template<>
inline const dds::core::policy::Partition&
PublisherQosDelegate::policy<dds::core::policy::Partition>() const
{
    return partition_;
}
template<>
inline dds::core::policy::Partition&
PublisherQosDelegate::policy<dds::core::policy::Partition>()
{
    return partition_;
}


template<>
inline const dds::core::policy::GroupData&
PublisherQosDelegate::policy<dds::core::policy::GroupData>() const
{
    return gdata_;
}
template<>
inline dds::core::policy::GroupData&
PublisherQosDelegate::policy<dds::core::policy::GroupData>()
{
    return gdata_;
}


template<>
inline const dds::core::policy::EntityFactory&
PublisherQosDelegate::policy<dds::core::policy::EntityFactory>() const
{
    return factory_policy_;
}
template<>
inline dds::core::policy::EntityFactory&
PublisherQosDelegate::policy<dds::core::policy::EntityFactory>()
{
    return factory_policy_;
}

}
}
}
}
}

#endif /* CYCLONEDDS_PUB_QOS_PUBLISHER_QOS_DELEGATE_HPP_ */
