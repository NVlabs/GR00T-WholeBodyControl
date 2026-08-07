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
/* Feature macros:

   - SSM: support for source-specific multicast
     requires: NETWORK_PARTIITONS
     also requires platform support; SSM is silently disabled if the
     platform doesn't support it

   - BANDWIDTH_LIMITING: transmit-side bandwidth limiting
     requires: NETWORK_CHANNELS (for now, anyway)

   - IPV6: support for IPV6
     requires: platform support (which itself is not part of DDSI)

   - NETWORK_PARTITIONS: support for multiple network partitions

   - NETWORK_CHANNELS: support for multiple network channels

*/
#include "dds/features.h"

#ifdef DDS_HAS_SSM
  #ifndef DDS_HAS_NETWORK_PARTITIONS
    #error "SSM requires NETWORK_PARTITIONS"
  #endif

  #include "dds/ddsrt/sockets.h"
  #ifndef DDSRT_HAVE_SSM
    #error "DDSRT_HAVE_SSM should be defined"
  #elif ! DDSRT_HAVE_SSM
    #undef DDS_HAS_SSM
  #endif
#endif

#ifdef DDS_HAS_BANDWIDTH_LIMITING
  #ifndef DDS_HAS_NETWORK_CHANNELS
    #error "BANDWIDTH_LIMITING requires NETWORK_CHANNELS"
  #endif
#endif
