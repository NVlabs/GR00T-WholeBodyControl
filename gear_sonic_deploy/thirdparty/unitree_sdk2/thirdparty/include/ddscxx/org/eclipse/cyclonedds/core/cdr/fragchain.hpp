/*
 * Copyright(c) 2022 ZettaScale Technology
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CDR_SERIALIZATION_FRAGCHAIN_HPP_
#define CDR_SERIALIZATION_FRAGCHAIN_HPP_

#include <cstddef>  //size_t
#include <dds/core/macros.hpp>

// unfortunate namespace pollution from C
struct nn_rdata;

namespace org { namespace eclipse { namespace cyclone { namespace core { namespace cdr {

OMG_DDS_API void serdata_from_ser_copyin_fragchain (unsigned char * __restrict cursor, const struct nn_rdata* fragchain, size_t size);

} } } } }

#endif
