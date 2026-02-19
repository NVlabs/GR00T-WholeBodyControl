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

/** @file
 *
 * @brief DDS C Error API
 *
 * This header file defines the public API of error values and convenience
 * functions in the CycloneDDS C language binding.
 */
#ifndef DDS_ERROR_H
#define DDS_ERROR_H

#include "dds/export.h"
#include "dds/ddsrt/log.h"
#include "dds/ddsrt/retcode.h"

#if defined (__cplusplus)
extern "C" {
#endif

/* ** DEPRECATED ** */
#ifndef DOXYGEN_SHOULD_SKIP_THIS

/* Error masks for returned status values */

#define DDS_ERR_NR_MASK       0x000000ff
#define DDS_ERR_LINE_MASK     0x003fff00
#define DDS_ERR_FILE_ID_MASK  0x7fc00000

/* Error code handling functions */

/** Macro to extract error number */
#define dds_err_nr(e) (e)

/** Macro to extract line number */
#define dds_err_line(e) (0)

/** Macro to extract file identifier */
#define dds_err_file_id(e) (0)

#endif // DOXYGEN_SHOULD_SKIP_THIS

#if defined (__cplusplus)
}
#endif
#endif
