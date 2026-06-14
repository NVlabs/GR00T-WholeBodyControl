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
#ifndef DDSI_THREADMON_H
#define DDSI_THREADMON_H

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_threadmon;
struct ddsi_domaingv;

struct ddsi_threadmon *ddsi_threadmon_new (int64_t liveliness_monitoring_interval, bool noprogress_log_stacktraces);
dds_return_t ddsi_threadmon_start (struct ddsi_threadmon *sl, const char *name);
void ddsi_threadmon_register_domain (struct ddsi_threadmon *sl, const struct ddsi_domaingv *gv);
void ddsi_threadmon_unregister_domain (struct ddsi_threadmon *sl, const struct ddsi_domaingv *gv);
void ddsi_threadmon_stop (struct ddsi_threadmon *sl);
void ddsi_threadmon_free (struct ddsi_threadmon *sl);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_THREADMON_H */
