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
#ifndef CYCLONEDDS_DDS_SUB_DETAIL_MANIPULATOR_HPP_
#define CYCLONEDDS_DDS_SUB_DETAIL_MANIPULATOR_HPP_

/**
 * @file
 */

#include <dds/sub/Query.hpp>

namespace dds
{
namespace sub
{
namespace functors
{
namespace detail
{

class MaxSamplesManipulatorFunctor
{
public:
    MaxSamplesManipulatorFunctor(uint32_t n) :
        n_(n)
    {
    }

    template<typename S>
    void operator()(S& s)
    {
        s.max_samples(n_);
    }
private:
    uint32_t n_;
};

class ContentFilterManipulatorFunctor
{
public:
    ContentFilterManipulatorFunctor(const dds::sub::Query& q) :
        query_(q)
    {
    }

    template<typename S>
    void operator()(S& s)
    {
        s.content(query_);
    }
private:
    const dds::sub::Query query_;
};

class StateFilterManipulatorFunctor
{
public:
    StateFilterManipulatorFunctor(
        const dds::sub::status::DataState& s) :
        state_(s)
    {
    }

    template<typename S>
    void operator()(S& s)
    {
        s.state(state_);
    }
private:
    dds::sub::status::DataState state_;
};

class InstanceManipulatorFunctor
{
public:
    InstanceManipulatorFunctor(const dds::core::InstanceHandle& h) :
        handle_(h)
    {
    }

    template<typename S>
    void operator()(S& s)
    {
        s.instance(handle_);
    }
private:
    dds::core::InstanceHandle handle_;
};

class NextInstanceManipulatorFunctor
{
public:
    NextInstanceManipulatorFunctor(
        const dds::core::InstanceHandle& h) :
        handle_(h)
    {
    }

    template<typename S>
    void operator()(S& s)
    {
        s.next_instance(handle_);
    }
private:
    dds::core::InstanceHandle handle_;
};


}
}
}
}



#endif /* CYCLONEDDS_DDS_SUB_DETAIL_MANIPULATOR_HPP_ */
