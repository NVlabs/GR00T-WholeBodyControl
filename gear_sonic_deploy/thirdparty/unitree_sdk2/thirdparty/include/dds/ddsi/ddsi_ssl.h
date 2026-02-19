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
#ifndef _DDSI_SSL_H_
#define _DDSI_SSL_H_

#include "dds/features.h"

#ifdef DDS_HAS_SSL

#ifdef _WIN32
/* supposedly WinSock2 must be included before openssl headers otherwise winsock will be used */
#include <WinSock2.h>
#endif
#include <openssl/ssl.h>

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_ssl_plugins
{
  bool (*init) (struct ddsi_domaingv *gv);
  void (*fini) (void);
  void (*ssl_free) (SSL *ssl);
  void (*bio_vfree) (BIO *bio);
  ssize_t (*read) (SSL *ssl, void *buf, size_t len, dds_return_t *err);
  ssize_t (*write) (SSL *ssl, const void *msg, size_t len, dds_return_t *err);
  SSL * (*connect) (const struct ddsi_domaingv *gv, ddsrt_socket_t sock);
  BIO * (*listen) (ddsrt_socket_t sock);
  SSL * (*accept) (const struct ddsi_domaingv *gv, BIO *bio, ddsrt_socket_t *sock);
};

void ddsi_ssl_config_plugin (struct ddsi_ssl_plugins *plugin);

#if defined (__cplusplus)
}
#endif

#endif
#endif
