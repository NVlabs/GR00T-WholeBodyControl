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

#ifndef CYCLONEDDS_DOMAIN_DOMAINPARTICIPANT_LISTENER_HPP_
#define CYCLONEDDS_DOMAIN_DOMAINPARTICIPANT_LISTENER_HPP_

#include <dds/domain/DomainParticipantListener.hpp>
#include <org/eclipse/cyclonedds/topic/AnyTopicListener.hpp>


namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace domain
{

DDSCXX_WARNING_MSVC_OFF(4250)

class OMG_DDS_API DomainParticipantListener :
    public virtual dds::domain::DomainParticipantListener,
    public virtual org::eclipse::cyclonedds::topic::AnyTopicListener
{
public:
    virtual ~DomainParticipantListener() { }
};


class OMG_DDS_API NoOpDomainParticipantListener :
    public virtual dds::domain::NoOpDomainParticipantListener,
    public virtual org::eclipse::cyclonedds::topic::NoOpAnyTopicListener
{
public:
    virtual ~NoOpDomainParticipantListener()  { }
};

DDSCXX_WARNING_MSVC_ON(4250)

}
}
}
}

#endif /* CYCLONEDDS_DOMAIN_DOMAINPARTICIPANT_LISTENER_HPP_ */
