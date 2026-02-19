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
#ifndef CYCLONEDDS_DDS_PUB_TCOHERENTSET_IMPL_HPP_
#define CYCLONEDDS_DDS_PUB_TCOHERENTSET_IMPL_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/pub/TCoherentSet.hpp>
#include <org/eclipse/cyclonedds/core/ReportUtils.hpp>

// Implementation

namespace dds
{
namespace pub
{

template <typename DELEGATE>
TCoherentSet<DELEGATE>::TCoherentSet(const dds::pub::Publisher& pub) : dds::core::Value<DELEGATE>(pub)  {  }

template <typename DELEGATE>
void TCoherentSet<DELEGATE>::end()
{
    this->delegate().end();
}

template <typename DELEGATE>
TCoherentSet<DELEGATE>::~TCoherentSet()
{
    this->delegate().end();
}

}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_PUB_TCOHERENTSET_HPP_ */
