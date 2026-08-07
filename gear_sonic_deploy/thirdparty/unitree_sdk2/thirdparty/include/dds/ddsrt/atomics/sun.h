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
#include <atomic.h>

#if defined (__cplusplus)
extern "C" {
#endif

#define DDSRT_ATOMIC64_SUPPORT 1

/* LD, ST */

inline uint32_t ddsrt_atomic_ld32 (const volatile ddsrt_atomic_uint32_t *x) { return x->v; }
inline uint64_t ddsrt_atomic_ld64 (const volatile ddsrt_atomic_uint64_t *x) { return x->v; }
inline uintptr_t ddsrt_atomic_ldptr (const volatile ddsrt_atomic_uintptr_t *x) { return x->v; }
inline void *ddsrt_atomic_ldvoidp (const volatile ddsrt_atomic_voidp_t *x) { return (void *) ddsrt_atomic_ldptr (x); }

inline void ddsrt_atomic_st32 (volatile ddsrt_atomic_uint32_t *x, uint32_t v) { x->v = v; }
inline void ddsrt_atomic_st64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v) { x->v = v; }
inline void ddsrt_atomic_stptr (volatile ddsrt_atomic_uintptr_t *x, uintptr_t v) { x->v = v; }
inline void ddsrt_atomic_stvoidp (volatile ddsrt_atomic_voidp_t *x, void *v) { ddsrt_atomic_stptr (x, (uintptr_t) v); }

/* INC */

inline void ddsrt_atomic_inc32 (volatile ddsrt_atomic_uint32_t *x) {
  atomic_inc_32 (&x->v);
}
inline void ddsrt_atomic_inc64 (volatile ddsrt_atomic_uint64_t *x) {
  atomic_inc_64 (&x->v);
}
inline uint32_t ddsrt_atomic_inc32_ov (volatile ddsrt_atomic_uint32_t *x) {
  uint32_t oldval, newval;
  do { oldval = x->v; newval = oldval + 1; } while (atomic_cas_32 (&x->v, oldval, newval) != oldval);
  return oldval;
}
inline uint32_t ddsrt_atomic_inc32_nv (volatile ddsrt_atomic_uint32_t *x) {
  return atomic_inc_32_nv (&x->v);
}
inline uint64_t ddsrt_atomic_inc64_nv (volatile ddsrt_atomic_uint64_t *x) {
  return atomic_inc_64_nv (&x->v);
}

/* DEC */

inline void ddsrt_atomic_dec32 (volatile ddsrt_atomic_uint32_t *x) {
  atomic_dec_32 (&x->v);
}
inline void ddsrt_atomic_dec64 (volatile ddsrt_atomic_uint64_t *x) {
  atomic_dec_64 (&x->v);
}
inline uint32_t ddsrt_atomic_dec32_nv (volatile ddsrt_atomic_uint32_t *x) {
  return atomic_dec_32_nv (&x->v);
}
inline uint64_t ddsrt_atomic_dec64_nv (volatile ddsrt_atomic_uint64_t *x) {
  return atomic_dec_64_nv (&x->v);
}
inline uint32_t ddsrt_atomic_dec32_ov (volatile ddsrt_atomic_uint32_t *x) {
  uint32_t oldval, newval;
  do { oldval = x->v; newval = oldval - 1; } while (atomic_cas_32 (&x->v, oldval, newval) != oldval);
  return oldval;
}
inline uint64_t ddsrt_atomic_dec64_ov (volatile ddsrt_atomic_uint64_t *x) {
  uint64_t oldval, newval;
  do { oldval = x->v; newval = oldval - 1; } while (atomic_cas_64 (&x->v, oldval, newval) != oldval);
  return oldval;
}

/* ADD */

inline void ddsrt_atomic_add32 (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  atomic_add_32 (&x->v, v);
}
inline void ddsrt_atomic_add64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  atomic_add_64 (&x->v, v);
}
inline uint32_t ddsrt_atomic_add32_ov (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  return atomic_add_32_nv (&x->v, v) - v;
}
inline uint32_t ddsrt_atomic_add32_nv (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  return atomic_add_32_nv (&x->v, v);
}
inline uint64_t ddsrt_atomic_add64_nv (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  return atomic_add_64_nv (&x->v, v);
}

/* SUB */

inline void ddsrt_atomic_sub32 (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  atomic_add_32 (&x->v, -v);
}
inline void ddsrt_atomic_sub64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  atomic_add_64 (&x->v, -v);
}
inline uint32_t ddsrt_atomic_sub32_ov (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  return atomic_add_32_nv (&x->v, -v) + v;
}
inline uint32_t ddsrt_atomic_sub32_nv (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  return atomic_add_32_nv (&x->v, -v);
}
inline uint64_t ddsrt_atomic_sub64_nv (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  return atomic_add_64_nv (&x->v, -v);
}

/* AND */

inline void ddsrt_atomic_and32 (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  atomic_and_32 (&x->v, v);
}
inline void ddsrt_atomic_and64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  atomic_and_64 (&x->v, v);
}
inline uint32_t ddsrt_atomic_and32_ov (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  uint32_t oldval, newval;
  do { oldval = x->v; newval = oldval & v; } while (atomic_cas_32 (&x->v, oldval, newval) != oldval);
  return oldval;
}
inline uint64_t ddsrt_atomic_and64_ov (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  uint64_t oldval, newval;
  do { oldval = x->v; newval = oldval & v; } while (atomic_cas_64 (&x->v, oldval, newval) != oldval);
  return oldval;
}
inline uint32_t ddsrt_atomic_and32_nv (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  return atomic_and_32_nv (&x->v, v);
}
inline uint64_t ddsrt_atomic_and64_nv (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  return atomic_and_64_nv (&x->v, v);
}

/* OR */

inline void ddsrt_atomic_or32 (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  atomic_or_32 (&x->v, v);
}
inline void ddsrt_atomic_or64 (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  atomic_or_64 (&x->v, v);
}
inline uint32_t ddsrt_atomic_or32_ov (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  uint32_t oldval, newval;
  do { oldval = x->v; newval = oldval | v; } while (atomic_cas_32 (&x->v, oldval, newval) != oldval);
  return oldval;
}
inline uint64_t ddsrt_atomic_or64_ov (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  uint64_t oldval, newval;
  do { oldval = x->v; newval = oldval | v; } while (atomic_cas_64 (&x->v, oldval, newval) != oldval);
  return oldval;
}
inline uint32_t ddsrt_atomic_or32_nv (volatile ddsrt_atomic_uint32_t *x, uint32_t v) {
  return atomic_or_32_nv (&x->v, v);
}
inline uint64_t ddsrt_atomic_or64_nv (volatile ddsrt_atomic_uint64_t *x, uint64_t v) {
  return atomic_or_64_nv (&x->v, v);
}

/* CAS */

inline int ddsrt_atomic_cas32 (volatile ddsrt_atomic_uint32_t *x, uint32_t exp, uint32_t des) {
  return atomic_cas_32 (&x->v, exp, des) == exp;
}
inline int ddsrt_atomic_cas64 (volatile ddsrt_atomic_uint64_t *x, uint64_t exp, uint64_t des) {
  return atomic_cas_64 (&x->v, exp, des) == exp;
}
inline int ddsrt_atomic_casptr (volatile ddsrt_atomic_uintptr_t *x, uintptr_t exp, uintptr_t des) {
  return atomic_cas_ulong (&x->v, exp, des) == exp;
}
inline int ddsrt_atomic_casvoidp (volatile ddsrt_atomic_voidp_t *x, void *exp, void *des) {
  return atomic_cas_ptr (&x->v, exp, des) == exp;
}

/* FENCES */

inline void ddsrt_atomic_fence (void) {
  membar_exit ();
  membar_enter ();
}
inline void ddsrt_atomic_fence_ldld (void) {
  membar_consumer ();
}
inline void ddsrt_atomic_fence_stst (void) {
  membar_producer ();
}
inline void ddsrt_atomic_fence_acq (void) {
  membar_enter ();
}
inline void ddsrt_atomic_fence_rel (void) {
  membar_exit ();
}

#if defined (__cplusplus)
}
#endif
