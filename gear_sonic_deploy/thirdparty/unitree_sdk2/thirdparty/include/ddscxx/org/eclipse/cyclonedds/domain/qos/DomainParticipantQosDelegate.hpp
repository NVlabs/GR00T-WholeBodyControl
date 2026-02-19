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

#ifndef CYCLONEDDS_DOMAIN_QOS_DOMAIN_PARTICIPANT_QOS_DELEGATE_HPP_
#define CYCLONEDDS_DOMAIN_QOS_DOMAIN_PARTICIPANT_QOS_DELEGATE_HPP_

#include <dds/core/policy/CorePolicy.hpp>

struct _DDS_NamedDomainParticipantQos;

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace domain
{
namespace qos
{

class OMG_DDS_API DomainParticipantQosDelegate
{
public:
    DomainParticipantQosDelegate();
    DomainParticipantQosDelegate(const DomainParticipantQosDelegate& other);

    ~DomainParticipantQosDelegate();

    void policy(const dds::core::policy::UserData& ud);
    void policy(const dds::core::policy::EntityFactory& efp);

    template <typename POLICY> const POLICY& policy() const;
    template <typename POLICY> POLICY& policy();

    /* The returned ddsc QoS has to be freed. */
    dds_qos_t* ddsc_qos() const;
    void ddsc_qos(const dds_qos_t* qos);

    void named_qos(const struct _DDS_NamedDomainParticipantQos &qos);

    void check() const;

    bool operator ==(const DomainParticipantQosDelegate& other) const;
    DomainParticipantQosDelegate& operator =(const DomainParticipantQosDelegate& other);

private:
    dds::core::policy::UserData user_data_;
    dds::core::policy::EntityFactory entity_factory_;
};



//==============================================================================


template<>
inline const dds::core::policy::UserData&
DomainParticipantQosDelegate::policy<dds::core::policy::UserData> () const
{
    return user_data_;
}
template<>
inline dds::core::policy::UserData&
DomainParticipantQosDelegate::policy<dds::core::policy::UserData> ()
{
    return user_data_;
}


template<>
inline const dds::core::policy::EntityFactory&
DomainParticipantQosDelegate::policy<dds::core::policy::EntityFactory> () const
{
    return entity_factory_;
}
template<>
inline dds::core::policy::EntityFactory&
DomainParticipantQosDelegate::policy<dds::core::policy::EntityFactory> ()
{
    return entity_factory_;
}

}
}
}
}
}

#endif /* CYCLONEDDS_DOMAIN_QOS_DOMAIN_PARTICIPANT_QOS_DELEGATE_HPP_ */
