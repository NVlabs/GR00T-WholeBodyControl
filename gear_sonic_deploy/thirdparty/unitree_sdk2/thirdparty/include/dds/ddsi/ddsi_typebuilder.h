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
#ifndef DDSI_TYPEBUILDER_H
#define DDSI_TYPEBUILDER_H

#include "dds/features.h"

#include <stdbool.h>
#include <stdint.h>
#include "dds/dds.h"
#include "dds/ddsi/ddsi_typelib.h"

#if defined (__cplusplus)
extern "C" {
#endif

#ifdef DDS_HAS_TYPE_DISCOVERY

DDS_EXPORT dds_return_t ddsi_topic_descriptor_from_type (struct ddsi_domaingv *gv, dds_topic_descriptor_t *desc, const struct ddsi_type *type);
DDS_EXPORT void ddsi_topic_descriptor_fini (dds_topic_descriptor_t *desc);

#endif /* DDS_HAS_TYPE_DISCOVERY */

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_TYPEBUILDER_H */
