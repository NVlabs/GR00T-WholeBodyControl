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

#ifndef CYCLONEDDS_CORE_POLICY_POLICY_HPP_
#define CYCLONEDDS_CORE_POLICY_POLICY_HPP_


/******************************************************************************
 *
 * PROPRIETARY POLICIES
 *
 ******************************************************************************/

#include <org/eclipse/cyclonedds/core/policy/PolicyDelegate.hpp>
#include <org/eclipse/cyclonedds/core/policy/TPolicy.hpp>


/* Use same macro as in dds/core/policy/CorePolicy.hpp, called OMG_DDS_POLICY_TRAITS */
#define LITE_POLICY_TRAITS(POLICY, ID) \
    template <> \
    class policy_id<POLICY> { \
    public: \
        static const dds::core::policy::QosPolicyId value = ID; \
    };\
    template <> \
    class policy_name<POLICY> { \
    public:\
        static const std::string& name(); \
    };

namespace dds
{
namespace core
{
namespace policy
{
template <typename Policy>
class policy_id;
template <typename Policy>
class policy_name;
}
}
}


#undef LITE_POLICY_TRAITS

#endif /* CYCLONEDDS_CORE_POLICY_POLICY_HPP_ */
