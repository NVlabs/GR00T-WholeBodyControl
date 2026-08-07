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

#ifndef CYCLONEDDS_SUB_COND_QUERYCONDITION_DELEGATE_HPP_
#define CYCLONEDDS_SUB_COND_QUERYCONDITION_DELEGATE_HPP_

#include <org/eclipse/cyclonedds/sub/cond/ReadConditionDelegate.hpp>

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

class OMG_DDS_API QueryConditionDelegate :
    public org::eclipse::cyclonedds::sub::cond::ReadConditionDelegate
{
public:

    typedef QueryDelegate::iterator iterator;
    typedef QueryDelegate::const_iterator const_iterator;

public:
    QueryConditionDelegate(
            const dds::sub::AnyDataReader& dr,
            const std::string& query_expression,
            const dds::sub::status::DataState& state_filter);

    QueryConditionDelegate(
            const dds::sub::AnyDataReader& dr,
            const std::string& expression,
            const std::vector<std::string>& params,
            const dds::sub::status::DataState& data_state);

    QueryConditionDelegate(
            const dds::sub::AnyDataReader& dr,
            const dds::sub::status::DataState& data_state);

    ~QueryConditionDelegate();

    typedef bool (*Filter_fn) (const void * sample);

    void set_filter(Filter_fn filter);

    Filter_fn get_filter();

    void init(ObjectDelegate::weak_ref_type weak_ref);
protected:
    Filter_fn cpp_filter;

    static bool trans_filter(const void *csample, const void *ctx);
};

}
}
}
}
}

#endif /* CYCLONEDDS_SUB_COND_QUERYCONDITION_DELEGATE_HPP_ */
