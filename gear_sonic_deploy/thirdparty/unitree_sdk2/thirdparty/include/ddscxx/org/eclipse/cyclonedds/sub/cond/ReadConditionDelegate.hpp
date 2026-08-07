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

#ifndef CYCLONEDDS_SUB_COND_READCONDITION_DELEGATE_HPP_
#define CYCLONEDDS_SUB_COND_READCONDITION_DELEGATE_HPP_

#include <dds/sub/AnyDataReader.hpp>

#include <org/eclipse/cyclonedds/core/cond/ConditionDelegate.hpp>
#include <org/eclipse/cyclonedds/sub/QueryDelegate.hpp>



namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{
namespace cond
{


class OMG_DDS_API ReadConditionDelegate :
    public virtual org::eclipse::cyclonedds::core::cond::ConditionDelegate,
    public virtual org::eclipse::cyclonedds::sub::QueryDelegate
{
public:
    ReadConditionDelegate(
            const dds::sub::AnyDataReader& dr,
            const dds::sub::status::DataState& state_filter);

    ~ReadConditionDelegate();

    void init(ObjectDelegate::weak_ref_type weak_ref);

    void close();

    virtual bool trigger_value() const;
};

}
}
}
}
}

#endif /* CYCLONEDDS_SUB_COND_READCONDITION_DELEGATE_HPP_ */
