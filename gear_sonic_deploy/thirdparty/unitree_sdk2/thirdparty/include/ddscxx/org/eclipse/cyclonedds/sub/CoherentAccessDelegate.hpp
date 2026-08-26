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

#ifndef CYCLONEDDS_SUB_COHERENT_ACCESS_DELEGATE_HPP_
#define CYCLONEDDS_SUB_COHERENT_ACCESS_DELEGATE_HPP_

#include <dds/sub/Subscriber.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{

class OMG_DDS_API CoherentAccessDelegate
{
public:
    CoherentAccessDelegate(const dds::sub::Subscriber sub);
    ~CoherentAccessDelegate();

    void end();

    bool operator==(const CoherentAccessDelegate& other) const;

private:
    dds::sub::Subscriber sub;
    bool ended;
};

}
}
}
}

#endif /* CYCLONEDDS_SUB_COHERENT_ACCESS_DELEGATE_HPP_ */
