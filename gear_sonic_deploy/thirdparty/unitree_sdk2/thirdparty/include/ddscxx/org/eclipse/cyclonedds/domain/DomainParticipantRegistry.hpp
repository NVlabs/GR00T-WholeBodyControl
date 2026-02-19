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

#ifndef CYCLONEDDS_DOMAIN_PARTICIPANT_REGISTRY_HPP_
#define CYCLONEDDS_DOMAIN_PARTICIPANT_REGISTRY_HPP_

#include <org/eclipse/cyclonedds/core/EntityRegistry.hpp>
#include <org/eclipse/cyclonedds/domain/DomainParticipantDelegate.hpp>
#include <dds/domain/TDomainParticipant.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace domain
{

class OMG_DDS_API DomainParticipantRegistry {
public:

    static void insert(dds::domain::TDomainParticipant<org::eclipse::cyclonedds::domain::DomainParticipantDelegate>& participant);

    static void remove(org::eclipse::cyclonedds::domain::DomainParticipantDelegate *delegate);

private:
    static org::eclipse::cyclonedds::core::EntityRegistry
               <org::eclipse::cyclonedds::domain::DomainParticipantDelegate *,
                dds::domain::TDomainParticipant<org::eclipse::cyclonedds::domain::DomainParticipantDelegate> > registry;
};

}
}
}
}

#endif /* CYCLONEDDS_DOMAIN_PARTICIPANT_REGISTRY_HPP_ */
