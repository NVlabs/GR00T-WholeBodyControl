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

#ifndef CYCLONEDDS_CORE_COND_WAITSET_DELEGATE_HPP_
#define CYCLONEDDS_CORE_COND_WAITSET_DELEGATE_HPP_

#include <vector>
#include <map>

#include <dds/core/Duration.hpp>
#include <dds/core/cond/Condition.hpp>
#include <org/eclipse/cyclonedds/core/config.hpp>
#include <org/eclipse/cyclonedds/core/DDScObjectDelegate.hpp>

namespace dds
{
  namespace core
  {
    namespace cond
    {
      template <typename DELEGATE> class TWaitSet;
    }
  }
}

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{
namespace cond
{

DDSCXX_WARNING_MSVC_OFF(4251)

    class OMG_DDS_API WaitSetDelegate :
        public org::eclipse::cyclonedds::core::DDScObjectDelegate
    {
    public:

        typedef ::dds::core::smart_ptr_traits< WaitSetDelegate >::ref_type ref_type;
        typedef ::dds::core::smart_ptr_traits< WaitSetDelegate >::weak_ref_type weak_ref_type;
        typedef std::vector<dds::core::cond::Condition> ConditionSeq;
        typedef std::map<org::eclipse::cyclonedds::core::cond::ConditionDelegate *,
                dds::core::cond::Condition> ConditionMap;

        typedef std::map<org::eclipse::cyclonedds::core::cond::ConditionDelegate *,
                dds::core::cond::Condition>::iterator ConditionIterator;

        typedef std::map<org::eclipse::cyclonedds::core::cond::ConditionDelegate *,
                dds::core::cond::Condition>::const_iterator ConstConditionIterator;

        typedef std::pair<org::eclipse::cyclonedds::core::cond::ConditionDelegate *,
                dds::core::cond::Condition> ConditionEntry;

        WaitSetDelegate ();
        virtual ~WaitSetDelegate ();

        void init (ObjectDelegate::weak_ref_type weak_ref);
        void close ();

        ConditionSeq& wait (ConditionSeq& triggered, const dds::core::Duration& timeout);

        void dispatch (const dds::core::Duration & timeout);

        void attach_condition (const dds::core::cond::Condition & cond);
        bool detach_condition (org::eclipse::cyclonedds::core::cond::ConditionDelegate * cond);
        void add_condition_locked(const dds::core::cond::Condition& cond);
        void remove_condition_locked(org::eclipse::cyclonedds::core::cond::ConditionDelegate *cond,
                                     const dds_entity_t entity_handle = DDS_HANDLE_NIL);

        ConditionSeq & conditions (ConditionSeq & conds) const;

    private:
        ConditionMap conditions_;
    };

DDSCXX_WARNING_MSVC_ON(4251)

}
}
}
}
}

#endif /* CYCLONEDDS_CORE_COND_WAITSET_DELEGATE_HPP_ */
