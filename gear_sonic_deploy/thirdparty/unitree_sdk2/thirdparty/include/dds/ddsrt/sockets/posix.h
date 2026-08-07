/*
 * Copyright(c) 2006 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSRT_SOCKETS_POSIX_H
#define DDSRT_SOCKETS_POSIX_H

#if DDSRT_WITH_LWIP
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#else
#include <sys/socket.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stddef.h>
#include <sys/types.h>
#include <sys/select.h>
#endif

#include "dds/ddsrt/iovec.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef int ddsrt_socket_t;
#define DDSRT_INVALID_SOCKET (-1)
#define PRIdSOCK "d"

#if LWIP_SOCKET
# define DDSRT_HAVE_SSM         0
# define IFF_UP               0x1
# define IFF_BROADCAST        0x2
# define IFF_LOOPBACK         0x8
# define IFF_POINTOPOINT     0x10
# define IFF_MULTICAST     0x1000
#elif __SunOS_5_6
# define DDSRT_HAVE_SSM         0
#else /* LWIP_SOCKET */
# define DDSRT_HAVE_SSM         1
#endif /* LWIP_SOCKET */

typedef struct msghdr ddsrt_msghdr_t;

#if (defined(__sun) && !defined(_XPG4_2)) || \
    (defined(LWIP_SOCKET))
# define DDSRT_MSGHDR_FLAGS 0
#else
# define DDSRT_MSGHDR_FLAGS 1
#endif

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_SOCKETS_POSIX_H */
