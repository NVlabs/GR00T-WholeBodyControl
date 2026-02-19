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

#ifndef IDLCXX_HASH_HPP_
#define IDLCXX_HASH_HPP_

#include "dds/core/macros.hpp"
#include "dds/ddsi/ddsi_keyhash.h"
#include <vector>

namespace org
{

  namespace eclipse
  {
    namespace cyclonedds
    {
      namespace topic
      {
        bool OMG_DDS_API simple_key(const std::vector<unsigned char>& in, ddsi_keyhash_t& out);

        bool OMG_DDS_API complex_key(const std::vector<unsigned char>& in, ddsi_keyhash_t& out);
      }
    }
  }
}

#endif /* IDLCXX_HASH_HPP_ */
