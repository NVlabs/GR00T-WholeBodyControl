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

#ifndef CYCLONEDDS_SUB_RANK_IMPL_HPP_
#define CYCLONEDDS_SUB_RANK_IMPL_HPP_

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{
class RankImpl;
}
}
}
}

class org::eclipse::cyclonedds::sub::RankImpl
{
public:
    RankImpl() : s_(0), g_(0), ag_(0) { }
    RankImpl(int32_t s, int32_t g, int32_t ag) : s_(s), g_(g), ag_(ag) { }

public:
    inline int32_t absolute_generation() const
    {
        return ag_;
    }

    inline int32_t generation() const
    {
        return g_;
    }

    inline int32_t sample() const
    {
        return s_;
    }

    bool operator ==(const RankImpl& other) const
    {
        return other.s_ == s_ && other.g_ == g_ && other.ag_;
    }
private:
    int32_t s_;
    int32_t g_;
    int32_t ag_;
};


#endif /* CYCLONEDDS_SUB_RANK_IMPL_HPP_ */
