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

#ifndef CYCLONEDDS_DOMAIN_WRAP_HPP_
#define CYCLONEDDS_DOMAIN_WRAP_HPP_

#include <map>
#include <dds/dds.h>
#include <dds/core/ref_traits.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace domain
{

/*
 * In cyclonedds, you can now create a domain explicitly. A participant will
 * attach to that domain automatically if the ids match. If the participant
 * can not find a domain, then it'll create one implicitly.
 *
 * The explicit creation is needed for the 'config by string' feature.
 *
 * To be able to use that feature in C++, this small 'domain' wrapper class
 * is introduced.
 *
 * It should be created for every new domain (detected during participant
 * creation) and should be deleted when the last participant is closed that
 * uses the related domain.
 *
 * Either, this wrapper contains an explicitly created domain or not.
 * When not, it's basically just a placeholder.
 * When yes, it'll delete the explicitly created domain in its destructor.
 */
class DomainWrap
{
public:
    typedef ::dds::core::smart_ptr_traits< DomainWrap >::ref_type ref_type;
    typedef ::std::map<dds_domainid_t, ref_type> map_ref_type;
    typedef map_ref_type::iterator map_ref_iter;

public:
    DomainWrap(dds_domainid_t id, const std::string& config);
    DomainWrap(dds_domainid_t id, const ddsi_config& config);
    ~DomainWrap();

private:
    dds_entity_t ddsc_domain;
};

}}}} /* namespaced */

#endif /* CYCLONEDDS_DOMAIN_WRAP_HPP_ */
