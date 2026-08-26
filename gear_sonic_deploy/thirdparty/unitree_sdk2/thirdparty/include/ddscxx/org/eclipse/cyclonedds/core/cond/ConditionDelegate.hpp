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

#ifndef CYCLONEDDS_CORE_COND_CONDITION_DELEGATE_HPP_
#define CYCLONEDDS_CORE_COND_CONDITION_DELEGATE_HPP_

#include <org/eclipse/cyclonedds/core/DDScObjectDelegate.hpp>
#include <org/eclipse/cyclonedds/core/cond/FunctorHolder.hpp>
#include <org/eclipse/cyclonedds/core/Mutex.hpp>
#include <org/eclipse/cyclonedds/core/ScopedLock.hpp>
#include <org/eclipse/cyclonedds/core/ReportUtils.hpp>

#include <set>

namespace dds
{
namespace core
{
namespace cond
{
template <typename DELEGATE> class TCondition;
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

class WaitSetDelegate;

class OMG_DDS_API ConditionDelegate :
                      public virtual org::eclipse::cyclonedds::core::DDScObjectDelegate
{
public:
    typedef ::dds::core::smart_ptr_traits< ConditionDelegate >::ref_type
                                                                      ref_type;
    typedef ::dds::core::smart_ptr_traits< ConditionDelegate >::weak_ref_type
                                                                 weak_ref_type;

    ConditionDelegate();

    ~ConditionDelegate();

    void init(ObjectDelegate::weak_ref_type weak_ref);

    void close();

    virtual bool trigger_value() const = 0;

    template <typename FUN>
    void set_handler(FUN functor)
    {
        org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);

        if (this->myFunctor)
        {
            delete this->myFunctor;
        }
        myFunctor =
           new org::eclipse::cyclonedds::core::cond::FunctorHolder<FUN>(functor);
    }

    void reset_handler();

    virtual void dispatch();

    virtual void add_waitset(
        const dds::core::cond::TCondition<ConditionDelegate> & cond,
        org::eclipse::cyclonedds::core::cond::WaitSetDelegate *waitset);

    virtual bool remove_waitset(
        org::eclipse::cyclonedds::core::cond::WaitSetDelegate *waitset);

    virtual void detach_from_waitset(const dds_entity_t entity_handle);
    virtual void detach_and_close(const dds_entity_t entity_handle);

    dds::core::cond::TCondition<ConditionDelegate> wrapper();

private:
    std::set<WaitSetDelegate *> waitSetList;
    org::eclipse::cyclonedds::core::Mutex waitSetListUpdateMutex;
    org::eclipse::cyclonedds::core::cond::FunctorHolderBase *myFunctor;
};

DDSCXX_WARNING_MSVC_ON(4251)

}
}
}
}
}

#endif /* CYCLONEDDS_CORE_COND_CONDITION_DELEGATE_HPP_ */
