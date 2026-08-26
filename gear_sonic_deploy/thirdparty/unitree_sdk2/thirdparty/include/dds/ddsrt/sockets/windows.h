#ifndef DDSRT_WINDOWS_SOCKET_H
#define DDSRT_WINDOWS_SOCKET_H

#include <winsock2.h>
#include <mswsock.h> // Required for MSG_TRUNC when compiling with mingw-w64.
#include <ws2tcpip.h>
#include "dds/ddsrt/iovec.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef SOCKET ddsrt_socket_t;
#define DDSRT_INVALID_SOCKET (INVALID_SOCKET)
#define PRIdSOCK PRIuPTR

#if defined(NTDDI_VERSION) && \
    defined(_WIN32_WINNT_WS03) && \
    (NTDDI_VERSION >= _WIN32_WINNT_WS03)
#define DDSRT_HAVE_SSM 1
#else
#define DDSRT_HAVE_SSM 0
#endif

#define IFF_POINTOPOINT IFF_POINTTOPOINT

// Required when compiling with mingw-w64.
#ifndef MCAST_JOIN_SOURCE_GROUP
#define MCAST_JOIN_SOURCE_GROUP 45
#endif
#ifndef MCAST_LEAVE_SOURCE_GROUP
#define MCAST_LEAVE_SOURCE_GROUP 46
#endif

typedef struct ddsrt_msghdr {
  void *msg_name;
  socklen_t msg_namelen;
  ddsrt_iovec_t *msg_iov;
  ddsrt_msg_iovlen_t msg_iovlen;
  void *msg_control;
  size_t msg_controllen;
  int msg_flags;
} ddsrt_msghdr_t;

#define DDSRT_MSGHDR_FLAGS 1

#if defined(__cplusplus)
}
#endif

#endif /* DDSRT_WINDOWS_SOCKET_H */
