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

#ifndef CYCLONEDDS_SUB_GENERATION_COUNT_IMPL_HPP_
#define CYCLONEDDS_SUB_GENERATION_COUNT_IMPL_HPP_

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{
class GenerationCountImpl;
}
}
}
}

class org::eclipse::cyclonedds::sub::GenerationCountImpl
{
public:
    GenerationCountImpl() : d_(0), nw_(0) { }
    GenerationCountImpl(int32_t d, int32_t nw) : d_(d), nw_(nw) { }

public:
    inline int32_t disposed() const
    {
        return d_;
    }

    inline void disposed(int32_t d)
    {
        this->d_ = d;
    }

    inline int32_t no_writers() const
    {
        return nw_;
    }

    inline void no_writers(int32_t nw)
    {
        this->nw_ = nw;
    }


    bool operator ==(const GenerationCountImpl& other) const
    {
        return other.d_ == d_ && other.nw_ == nw_;
    }

private:
    int32_t d_, nw_;
};

#endif /* CYCLONEDDS_SUB_GENERATION_COUNT_IMPL_HPP_ */
