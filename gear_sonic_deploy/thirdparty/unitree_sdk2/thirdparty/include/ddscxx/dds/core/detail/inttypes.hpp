/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_DDS_CORE_DETAIL_INTTYPES_HPP_
#define CYCLONEDDS_DDS_CORE_DETAIL_INTTYPES_HPP_

/**
 * @file
 */

// Implementation

/* (from spec:) This implementation-defined header stands in for the C99 header files
 * inttypes.h. Under toolchains that support inttypes.h, this header can
 * simply include that one. Under toolchains that do not, this header must
 * provide equivalent definitions.
 */
#if defined(__GNUC__) && __GNUC__ < 5
#define __STDC_FORMAT_MACROS
#endif

#include <stdint.h>
#include <inttypes.h>

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_DETAIL_INTTYPES_HPP_ */
