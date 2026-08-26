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
#ifndef DDSRT_BSWAP_H
#define DDSRT_BSWAP_H

#include <stdint.h>
#include <stdlib.h>

#include "dds/export.h"
#include "dds/ddsrt/endian.h"

#if defined (__cplusplus)
extern "C" {
#endif

enum ddsrt_byte_order_selector {
  DDSRT_BOSEL_NATIVE,
  DDSRT_BOSEL_BE,
  DDSRT_BOSEL_LE,
};

DDS_INLINE_EXPORT inline uint16_t ddsrt_bswap2u (uint16_t x)
{
  return (uint16_t) ((x >> 8) | (x << 8));
}

DDS_INLINE_EXPORT inline int16_t ddsrt_bswap2 (int16_t x)
{
  return (int16_t) ddsrt_bswap2u ((uint16_t) x);
}

DDS_INLINE_EXPORT inline uint32_t ddsrt_bswap4u (uint32_t x)
{
  return (x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | (x << 24);
}

DDS_INLINE_EXPORT inline int32_t ddsrt_bswap4 (int32_t x)
{
  return (int32_t) ddsrt_bswap4u ((uint32_t) x);
}

DDS_INLINE_EXPORT inline uint64_t ddsrt_bswap8u (uint64_t x)
{
  const uint32_t newhi = ddsrt_bswap4u ((uint32_t) x);
  const uint32_t newlo = ddsrt_bswap4u ((uint32_t) (x >> 32));
  return ((uint64_t) newhi << 32) | (uint64_t) newlo;
}

DDS_INLINE_EXPORT inline int64_t ddsrt_bswap8 (int64_t x)
{
  return (int64_t) ddsrt_bswap8u ((uint64_t) x);
}

#if DDSRT_ENDIAN == DDSRT_LITTLE_ENDIAN
#define ddsrt_toBE2(x) ddsrt_bswap2 (x)
#define ddsrt_toBE2u(x) ddsrt_bswap2u (x)
#define ddsrt_toBE4(x) ddsrt_bswap4 (x)
#define ddsrt_toBE4u(x) ddsrt_bswap4u (x)
#define ddsrt_toBE8(x) ddsrt_bswap8 (x)
#define ddsrt_toBE8u(x) ddsrt_bswap8u (x)
#define ddsrt_toLE2(x) (x)
#define ddsrt_toLE2u(x) (x)
#define ddsrt_toLE4(x) (x)
#define ddsrt_toLE4u(x) (x)
#define ddsrt_toLE8(x) (x)
#define ddsrt_toLE8u(x) (x)
#define ddsrt_toBO2(bo, x) ((bo) == DDSRT_BOSEL_BE ? ddsrt_bswap2 (x) : (x))
#define ddsrt_toBO2u(bo, x) ((bo) == DDSRT_BOSEL_BE ? ddsrt_bswap2u (x) : (x))
#define ddsrt_toBO4(bo, x) ((bo) == DDSRT_BOSEL_BE ? ddsrt_bswap4 (x) : (x))
#define ddsrt_toBO4u(bo, x) ((bo) == DDSRT_BOSEL_BE ? ddsrt_bswap4u (x) : (x))
#define ddsrt_toBO8(bo, x) ((bo) == DDSRT_BOSEL_BE ? ddsrt_bswap8 (x) : (x))
#define ddsrt_toBO8u(bo, x) ((bo) == DDSRT_BOSEL_BE ? ddsrt_bswap8u (x) : (x))
#define ddsrt_fromBE2(x) ddsrt_bswap2 (x)
#define ddsrt_fromBE2u(x) ddsrt_bswap2u (x)
#define ddsrt_fromBE4(x) ddsrt_bswap4 (x)
#define ddsrt_fromBE4u(x) ddsrt_bswap4u (x)
#define ddsrt_fromBE8(x) ddsrt_bswap8 (x)
#define ddsrt_fromBE8u(x) ddsrt_bswap8u (x)
#else
#define ddsrt_toBE2u(x) (x)
#define ddsrt_toBE4(x) (x)
#define ddsrt_toBE4u(x) (x)
#define ddsrt_toBE8(x) (x)
#define ddsrt_toBE8u(x) (x)
#define ddsrt_toLE2(x) ddsrt_bswap2 (x)
#define ddsrt_toLE2u(x) ddsrt_bswap2u (x)
#define ddsrt_toLE4(x) ddsrt_bswap4 (x)
#define ddsrt_toLE4u(x) ddsrt_bswap4u (x)
#define ddsrt_toLE8(x) ddsrt_bswap8 (x)
#define ddsrt_toLE8u(x) ddsrt_bswap8u (x)
#define ddsrt_toBO2(bo, x) ((bo) == DDSRT_BOSEL_LE ? ddsrt_bswap2 (x) : (x))
#define ddsrt_toBO2u(bo, x) ((bo) == DDSRT_BOSEL_LE ? ddsrt_bswap2u (x) : (x))
#define ddsrt_toBO4(bo, x) ((bo) == DDSRT_BOSEL_LE ? ddsrt_bswap4 (x) : (x))
#define ddsrt_toBO4u(bo, x) ((bo) == DDSRT_BOSEL_LE ? ddsrt_bswap4u (x) : (x))
#define ddsrt_toBO8(bo, x) ((bo) == DDSRT_BOSEL_LE ? ddsrt_bswap8 (x) : (x))
#define ddsrt_toBO8u(bo, x) ((bo) == DDSRT_BOSEL_LE ? ddsrt_bswap8u (x) : (x))
#define ddsrt_fromBE2(x) (x)
#define ddsrt_fromBE2u(x) (x)
#define ddsrt_fromBE4(x) (x)
#define ddsrt_fromBE4u(x) (x)
#define ddsrt_fromBE8(x) (x)
#define ddsrt_fromBE8u(x) (x)
#endif

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_BSWAP_H */

