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
#ifndef DDSRT_ATOMICS_H
#define DDSRT_ATOMICS_H

#include <stddef.h>

#include "dds/export.h"
#include "dds/ddsrt/arch.h"
#include "dds/ddsrt/endian.h"
#include "dds/ddsrt/types.h"

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * @brief Types on which atomic operations are defined.
 *
 * @note 64-bit types are defined even if atomic operations on them are not
 *       really supported. atomic
 */
typedef struct { uint32_t v; } ddsrt_atomic_uint32_t;
typedef struct { uint64_t v; } ddsrt_atomic_uint64_t;
typedef struct { uintptr_t v; } ddsrt_atomic_uintptr_t;
typedef ddsrt_atomic_uintptr_t ddsrt_atomic_voidp_t;

#if DDSRT_64BIT
# define DDSRT_HAVE_ATOMIC64 1
#endif

/**
 * @brief Initializers for the types on which atomic operations are defined.
 */
#define DDSRT_ATOMIC_UINT32_INIT(v) { (v) }
#define DDSRT_ATOMIC_UINT64_INIT(v) { (v) }
#define DDSRT_ATOMIC_UINTPTR_INIT(v) { (v) }
#define DDSRT_ATOMIC_VOIDP_INIT(v) { (uintptr_t) (v) }


#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40100
#include "dds/ddsrt/atomics/gcc.h"
#elif defined(_WIN32)
#include "dds/ddsrt/atomics/msvc.h"
#elif defined(__sun)
#include "dds/ddsrt/atomics/sun.h"
#elif defined(__IAR_SYSTEMS_ICC__) && defined(__ICCARM__)
#include "dds/ddsrt/atomics/arm.h"
#else
#error "Atomic operations are not supported"
#endif

#if ! DDSRT_HAVE_ATOMIC64
/* 64-bit atomics are not supported by all hardware, but it would be a shame not to use them when
   they are available.  That necessitates an alternative implementation when they are not, either in
   the form of a different implementation where it is used, or as an emulation using a mutex in
   ddsrt.  It seems that the places where they'd be used end up adding a mutex, so an emulation in
   ddsrt while being able to check whether it is supported by hardware is a sensible approach.  */
DDS_EXPORT uint64_t ddsrt_atomic_ld64 (const volatile ddsrt_atomic_uint64_t *x);
DDS_EXPORT void ddsrt_atomic_st64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT void ddsrt_atomic_inc64 (volatile ddsrt_atomic_uint64_t *x);
DDS_EXPORT uint64_t ddsrt_atomic_inc64_nv (volatile ddsrt_atomic_uint64_t *x);
DDS_EXPORT void ddsrt_atomic_dec64 (volatile ddsrt_atomic_uint64_t *x);
DDS_EXPORT uint64_t ddsrt_atomic_dec64_nv (volatile ddsrt_atomic_uint64_t *x);
DDS_EXPORT void ddsrt_atomic_add64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT uint64_t ddsrt_atomic_add64_nv (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT void ddsrt_atomic_sub64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT uint64_t ddsrt_atomic_sub64_nv (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT void ddsrt_atomic_and64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT uint64_t ddsrt_atomic_and64_ov (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT uint64_t ddsrt_atomic_and64_nv (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT void ddsrt_atomic_or64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT uint64_t ddsrt_atomic_or64_ov (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT uint64_t ddsrt_atomic_or64_nv (volatile ddsrt_atomic_uint64_t *x, uint64_t v);
DDS_EXPORT int ddsrt_atomic_cas64 (volatile ddsrt_atomic_uint64_t *x, uint64_t exp, uint64_t des);
#endif

void ddsrt_atomics_init (void);
void ddsrt_atomics_fini (void);

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_ATOMICS_H */
