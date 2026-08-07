/*
 * Copyright(c) 2006 to 2019 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSRT_TYPES_VXWORKS_H
#define DDSRT_TYPES_VXWORKS_H

#if defined(_WRS_KERNEL)
/* inttypes.h does not exist in VxWorks DKM. */
#include <st_inttypes.h>
#include <cafe/inttypes.h>
/* The above inttypes includes don't seem to define uintmax_t &c. */
#ifdef _WRS_CONFIG_LP64 /* Used in cafe/inttypes.h too. */
#define _PFX_64 "l"
typedef unsigned long int       uintmax_t;
#else
#define _PFX_64 "ll"
typedef unsigned long long int  uintmax_t;
#endif

/* Not a complete replacement for inttypes.h (yet); No SCN/PRI?LEAST/FAST/etc. */
#define PRId8      "d"
#define PRId16     "d"
#define PRId32     "d"
#define PRId64     _PFX_64 "d"

#define PRIi8      "i"
#define PRIi16     "i"
#define PRIi32     "i"
#define PRIi64     _PFX_64 "i"

#define PRIo8      "o"
#define PRIo16     "o"
#define PRIo32     "o"
#define PRIo64     _PFX_64 "o"

#define PRIu8      "u"
#define PRIu16     "u"
#define PRIu32     "u"
#define PRIu64     _PFX_64 "u"

#define PRIx8      "x"
#define PRIx16     "x"
#define PRIx32     "x"

#define PRIX8      "X"
#define PRIX16     "X"
#define PRIX32     "X"
#define PRIX64     _PFX_64 "X"

#define PRIdMAX    _PFX_64 "d"
#define PRIiMAX    _PFX_64 "i"
#define PRIoMAX    _PFX_64 "o"
#define PRIuMAX    _PFX_64 "u"
#define PRIxMAX    _PFX_64 "x"
#define PRIXMAX    _PFX_64 "X"

#define PRIdPTR    _PFX_64 "d"
#define PRIiPTR    _PFX_64 "i"
#define PRIoPTR    _PFX_64 "o"
#define PRIXPTR    _PFX_64 "X"

#else /* _WRS_KERNEL */
#include <inttypes.h>
#endif /* _WRS_KERNEL */

#endif /* DDSRT_TYPES_VXWORKS_H */
