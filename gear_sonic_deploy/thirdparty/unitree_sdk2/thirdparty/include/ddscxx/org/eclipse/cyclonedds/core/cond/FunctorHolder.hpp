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

#ifndef CYCLONEDDS_CORE_COND_FUNCTOR_HOLDER_HPP_
#define CYCLONEDDS_CORE_COND_FUNCTOR_HOLDER_HPP_

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

class ConditionDelegate;

class FunctorHolderBase
{
public:
    FunctorHolderBase() { };

    virtual ~FunctorHolderBase() { };

    virtual void dispatch(dds::core::cond::TCondition<org::eclipse::cyclonedds::core::cond::ConditionDelegate> &condition) = 0;
};

template <typename FUN>
class FunctorHolder : public FunctorHolderBase
{
public:
    FunctorHolder(FUN functor) : myFunctor(functor)
    {
    }

    virtual ~FunctorHolder() { };

    void dispatch(dds::core::cond::TCondition<org::eclipse::cyclonedds::core::cond::ConditionDelegate> &condition)
    {
        myFunctor(condition);
    }

private:
    FUN myFunctor;
};

}
}
}
}
}

#endif  /* CYCLONEDDS_CORE_COND_FUNCTOR_HOLDER_HPP_ */
