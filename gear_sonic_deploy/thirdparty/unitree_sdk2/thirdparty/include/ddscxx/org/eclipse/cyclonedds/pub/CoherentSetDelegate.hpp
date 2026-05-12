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

#ifndef CYCLONEDDS_PUB_COHERENT_SET_DELEGATE_HPP_
#define CYCLONEDDS_PUB_COHERENT_SET_DELEGATE_HPP_

#include <dds/pub/Publisher.hpp>


namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace pub
{


class OMG_DDS_API CoherentSetDelegate
{
public:
    CoherentSetDelegate(const dds::pub::Publisher& pub);
    ~CoherentSetDelegate();

    void end();

    bool operator ==(const CoherentSetDelegate& other) const;

private:
    dds::pub::Publisher pub;
    bool ended;
};

}
}
}
}

#endif /* CYCLONEDDS_PUB_COHERENT_SET_DELEGATE_HPP_ */
