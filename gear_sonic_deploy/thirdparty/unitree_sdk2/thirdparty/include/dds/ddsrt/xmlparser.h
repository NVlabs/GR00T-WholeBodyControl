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
#ifndef DDSRT_XMLPARSER_H
#define DDSRT_XMLPARSER_H

#include <stdio.h>
#include <stdint.h>

#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

    typedef int (*ddsrt_xmlp_proc_elem_open_t) (void *varg, uintptr_t parentinfo, uintptr_t *eleminfo, const char *name, int line);
    typedef int (*ddsrt_xmlp_proc_attr_t) (void *varg, uintptr_t eleminfo, const char *name, const char *value, int line);
    typedef int (*ddsrt_xmlp_proc_elem_data_t) (void *varg, uintptr_t eleminfo, const char *data, int line);
    typedef int (*ddsrt_xmlp_proc_elem_close_t) (void *varg, uintptr_t eleminfo, int line);
    typedef void (*ddsrt_xmlp_error) (void *varg, const char *msg, int line);

    struct ddsrt_xmlp_callbacks {
        ddsrt_xmlp_proc_elem_open_t elem_open;
        ddsrt_xmlp_proc_attr_t attr;
        ddsrt_xmlp_proc_elem_data_t elem_data;
        ddsrt_xmlp_proc_elem_close_t elem_close;
        ddsrt_xmlp_error error;
    };

    struct ddsrt_xmlp_state;

#define DDSRT_XMLP_REQUIRE_EOF          1u /* set by default; if not set, junk may follow top-level closing tag */
#define DDSRT_XMLP_ANONYMOUS_CLOSE_TAG  2u /* clear by default; if set allow closing an element with </> instead of </name> */
#define DDSRT_XMLP_MISSING_CLOSE_AS_EOF 4u /* clear by default; if set, treat missing close tag as EOF */
    DDS_EXPORT struct ddsrt_xmlp_state *ddsrt_xmlp_new_file (FILE *fp, void *varg, const struct ddsrt_xmlp_callbacks *cb);
    DDS_EXPORT struct ddsrt_xmlp_state *ddsrt_xmlp_new_string (const char *string, void *varg, const struct ddsrt_xmlp_callbacks *cb);
    DDS_EXPORT void ddsrt_xmlp_set_options (struct ddsrt_xmlp_state *st, unsigned options);
    DDS_EXPORT size_t ddsrt_xmlp_get_bufpos (const struct ddsrt_xmlp_state *st);
    DDS_EXPORT void ddsrt_xmlp_free (struct ddsrt_xmlp_state *st);
    DDS_EXPORT int ddsrt_xmlp_parse (struct ddsrt_xmlp_state *st);

#if defined (__cplusplus)
}
#endif

#endif
