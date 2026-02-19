/*
 * Copyright(c) 2019 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSI_VENDOR_H
#define DDSI_VENDOR_H

#include <stdint.h>
#include <stdbool.h>

#include "dds/export.h"

typedef struct {
  uint8_t id[2];
} nn_vendorid_t;


/* All existing vendor codes have the major part equal to 1 (and this will probably be true for a long, long time) */
#define NN_VENDORID_MINOR_RTI                0x01
#define NN_VENDORID_MINOR_ADLINK_OSPL        0x02
#define NN_VENDORID_MINOR_OCI                0x03
#define NN_VENDORID_MINOR_MILSOFT            0x04
#define NN_VENDORID_MINOR_KONGSBERG          0x05
#define NN_VENDORID_MINOR_TWINOAKS           0x06
#define NN_VENDORID_MINOR_LAKOTA             0x07
#define NN_VENDORID_MINOR_ICOUP              0x08
#define NN_VENDORID_MINOR_ETRI               0x09
#define NN_VENDORID_MINOR_RTI_MICRO          0x0a
#define NN_VENDORID_MINOR_ADLINK_JAVA        0x0b
#define NN_VENDORID_MINOR_ADLINK_GATEWAY     0x0c
#define NN_VENDORID_MINOR_ADLINK_LITE        0x0d
#define NN_VENDORID_MINOR_TECHNICOLOR        0x0e
#define NN_VENDORID_MINOR_EPROSIMA           0x0f
#define NN_VENDORID_MINOR_ECLIPSE            0x10
#define NN_VENDORID_MINOR_ADLINK_CLOUD       0x20

#define NN_VENDORID_INIT(vendor) {{ 0x01, NN_VENDORID_MINOR_##vendor }}
#define NN_VENDORID_INIT_UNKNOWN {{ 0x00, 0x00 }}

#define NN_VENDORID(vendor) ((nn_vendorid_t) NN_VENDORID_INIT(vendor))
#define NN_VENDORID_UNKNOWN ((nn_vendorid_t) NN_VENDORID_INIT_UNKNOWN)

#define NN_VENDORID_ECLIPSE NN_VENDORID(ECLIPSE)

#if defined (__cplusplus)
extern "C" {
#endif

DDS_INLINE_EXPORT inline bool vendor_equals (nn_vendorid_t a, nn_vendorid_t b) {
  return ((a.id[0] << 8) | a.id[1]) == ((b.id[0] << 8) | b.id[1]);
}
DDS_INLINE_EXPORT inline bool vendor_is_eclipse (nn_vendorid_t vendor) {
  const nn_vendorid_t x = NN_VENDORID_INIT (ECLIPSE);
  return vendor_equals (vendor, x);
}
DDS_INLINE_EXPORT inline bool vendor_is_rti (nn_vendorid_t vendor) {
  const nn_vendorid_t x = NN_VENDORID_INIT (RTI);
  return vendor_equals (vendor, x);
}
DDS_INLINE_EXPORT inline bool vendor_is_opensplice (nn_vendorid_t vendor) {
  const nn_vendorid_t x = NN_VENDORID_INIT (ADLINK_OSPL);
  return vendor_equals (vendor, x);
}
DDS_INLINE_EXPORT inline bool vendor_is_twinoaks (nn_vendorid_t vendor) {
  const nn_vendorid_t x = NN_VENDORID_INIT (TWINOAKS);
  return vendor_equals (vendor, x);
}
DDS_INLINE_EXPORT inline bool vendor_is_eprosima (nn_vendorid_t vendor) {
  const nn_vendorid_t x = NN_VENDORID_INIT (EPROSIMA);
  return vendor_equals (vendor, x);
}
DDS_INLINE_EXPORT inline bool vendor_is_cloud (nn_vendorid_t vendor) {
  const nn_vendorid_t x = NN_VENDORID_INIT (ADLINK_CLOUD);
  return vendor_equals (vendor, x);
}
DDS_INLINE_EXPORT inline bool vendor_is_eclipse_or_opensplice (nn_vendorid_t vendor) {
  return vendor_is_eclipse (vendor) || vendor_is_opensplice (vendor);
}
DDS_INLINE_EXPORT inline bool vendor_is_adlink (nn_vendorid_t vendor) {
  const nn_vendorid_t a = NN_VENDORID_INIT (ADLINK_OSPL);
  const nn_vendorid_t b = NN_VENDORID_INIT (ADLINK_LITE);
  const nn_vendorid_t c = NN_VENDORID_INIT (ADLINK_GATEWAY);
  const nn_vendorid_t d = NN_VENDORID_INIT (ADLINK_JAVA);
  const nn_vendorid_t e = NN_VENDORID_INIT (ADLINK_CLOUD);
  return (vendor_equals (vendor, a) ||
          vendor_equals (vendor, b) ||
          vendor_equals (vendor, c) ||
          vendor_equals (vendor, d) ||
          vendor_equals (vendor, e));
}
DDS_INLINE_EXPORT inline bool vendor_is_eclipse_or_adlink (nn_vendorid_t vendor) {
  return vendor_is_eclipse (vendor) || vendor_is_adlink (vendor);
}

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_VENDOR_H */
