/*
 * Copyright(c) 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSI_DELIVER_LOCALLY_H
#define DDSI_DELIVER_LOCALLY_H

#include <stdint.h>
#include <stdbool.h>

#include "dds/export.h"
#include "dds/ddsrt/retcode.h"
#include "dds/ddsrt/avl.h"
#include "dds/ddsi/ddsi_guid.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_domaingv;
struct ddsi_tkmap_instance;
struct ddsi_sertype;
struct ddsi_serdata;
struct entity_index;
struct ddsi_reader;
struct ddsi_entity_common;
struct ddsi_writer_info;
struct ddsi_local_reader_ary;

typedef struct ddsi_serdata * (*deliver_locally_makesample_t) (struct ddsi_tkmap_instance **tk, struct ddsi_domaingv *gv, struct ddsi_sertype const * const type, void *vsourceinfo);
typedef struct ddsi_reader * (*deliver_locally_first_reader_t) (struct entity_index *entity_index, struct ddsi_entity_common *source_entity, ddsrt_avl_iter_t *it);
typedef struct ddsi_reader * (*deliver_locally_next_reader_t) (struct entity_index *entity_index, ddsrt_avl_iter_t *it);

/** return:
    - DDS_RETCODE_OK to try again immediately
    - DDS_RETCODE_TRY_AGAIN to complete restart the operation later
    - anything else: error to be returned from deliver_locally_xxx */
typedef dds_return_t (*deliver_locally_on_failure_fastpath_t) (struct ddsi_entity_common *source_entity, bool source_entity_locked, struct ddsi_local_reader_ary *fastpath_rdary, void *vsourceinfo);

struct deliver_locally_ops {
  deliver_locally_makesample_t makesample;
  deliver_locally_first_reader_t first_reader;
  deliver_locally_next_reader_t next_reader;
  deliver_locally_on_failure_fastpath_t on_failure_fastpath;
};

dds_return_t deliver_locally_one (struct ddsi_domaingv *gv, struct ddsi_entity_common *source_entity, bool source_entity_locked, const ddsi_guid_t *rdguid, const struct ddsi_writer_info *wrinfo, const struct deliver_locally_ops * __restrict ops, void *vsourceinfo);

dds_return_t deliver_locally_allinsync (struct ddsi_domaingv *gv, struct ddsi_entity_common *source_entity, bool source_entity_locked, struct ddsi_local_reader_ary *fastpath_rdary, const struct ddsi_writer_info *wrinfo, const struct deliver_locally_ops * __restrict ops, void *vsourceinfo);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_DELIVER_LOCALLY_H */
