#ifndef DDSRT_SOCKETS_H
#define DDSRT_SOCKETS_H

#include <stdbool.h>

#include "dds/export.h"
#include "dds/config.h"
#include "dds/ddsrt/types.h"
#include "dds/ddsrt/attributes.h"
#include "dds/ddsrt/retcode.h"
#include "dds/ddsrt/time.h"
#include "dds/ddsrt/misc.h"
#if _WIN32
#include "dds/ddsrt/sockets/windows.h"
#else
#include "dds/ddsrt/sockets/posix.h"
#endif

#if defined (__cplusplus)
extern "C" {
#endif

#define INET_ADDRSTRLEN_EXTENDED (INET_ADDRSTRLEN + 6) /* ":12345" */

#if DDSRT_HAVE_IPV6
#define INET6_ADDRSTRLEN_EXTENDED (INET6_ADDRSTRLEN + 8) /* "[]:12345" */
extern const struct in6_addr ddsrt_in6addr_any;
extern const struct in6_addr ddsrt_in6addr_loopback;
#endif /* DDSRT_HAVE_IPV6 */

#define DDSRT_AF_TERM (-1)

#if DDSRT_HAVE_GETHOSTNAME
DDS_EXPORT dds_return_t
ddsrt_gethostname(
  char *hostname,
  size_t buffersize);
#endif

DDS_EXPORT dds_return_t
ddsrt_socket(
  ddsrt_socket_t *sockptr,
  int domain,
  int type,
  int protocol);

DDS_EXPORT dds_return_t
ddsrt_close(
  ddsrt_socket_t sock);

DDS_EXPORT dds_return_t
ddsrt_connect(
  ddsrt_socket_t sock,
  const struct sockaddr *addr,
  socklen_t addrlen);

DDS_EXPORT dds_return_t
ddsrt_accept(
  ddsrt_socket_t sock,
  struct sockaddr *addr,
  socklen_t *addrlen,
  ddsrt_socket_t *connptr);

DDS_EXPORT dds_return_t
ddsrt_listen(
  ddsrt_socket_t sock,
  int backlog);

DDS_EXPORT dds_return_t
ddsrt_bind(
  ddsrt_socket_t sock,
  const struct sockaddr *addr,
  socklen_t addrlen);

DDS_EXPORT dds_return_t
ddsrt_getsockname(
  ddsrt_socket_t sock,
  struct sockaddr *addr,
  socklen_t *addrlen);

DDS_EXPORT dds_return_t
ddsrt_send(
  ddsrt_socket_t sock,
  const void *buf,
  size_t len,
  int flags,
  ssize_t *sent);

DDS_EXPORT dds_return_t
ddsrt_sendmsg(
  ddsrt_socket_t sock,
  const ddsrt_msghdr_t *msg,
  int flags,
  ssize_t *sent);

DDS_EXPORT dds_return_t
ddsrt_recv(
  ddsrt_socket_t sock,
  void *buf,
  size_t len,
  int flags,
  ssize_t *rcvd);

DDS_EXPORT dds_return_t
ddsrt_recvmsg(
  ddsrt_socket_t sock,
  ddsrt_msghdr_t *msg,
  int flags,
  ssize_t *rcvd);

DDS_EXPORT dds_return_t
ddsrt_getsockopt(
  ddsrt_socket_t sock,
  int32_t level, /* SOL_SOCKET */
  int32_t optname, /* SO_REUSEADDR, SO_DONTROUTE, SO_BROADCAST, SO_SNDBUF, SO_RCVBUF, ... */
  void *optval,
  socklen_t *optlen);

DDS_EXPORT dds_return_t
ddsrt_setsockopt(
  ddsrt_socket_t sock,
  int32_t level, /* SOL_SOCKET */
  int32_t optname, /* SO_REUSEADDR, SO_DONTROUTE, SO_BROADCAST, SO_SNDBUF, SO_RCVBUF, ... */
  const void *optval,
  socklen_t optlen);

/**
 * @brief Set the I/O on the socket to blocking or non-nonblocking.
 *
 * @param[in]  sock      Socket to set I/O mode for.
 * @param[in]  nonblock  true for nonblocking, or false for blocking I/O.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             I/O mode successfully set to (non)blocking.
 * @retval DDS_RETCODE_TRY_AGAIN
 *             A blocking call is in progress.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             @sock is not a valid socket.
 * @retval DDS_RETCODE_ERROR
 *             An unknown error error occurred.
 */
DDS_EXPORT dds_return_t
ddsrt_setsocknonblocking(
  ddsrt_socket_t sock,
  bool nonblock);

/**
 * @brief Set whether a port may be shared with other sockets
 *
 * Maps to SO_REUSEPORT (if defined) followed by SO_REUSEADDR
 *
 * @param[in]   sock  Socket to set reusability for.
 * @param[in]   reuse Whether to allow sharing the address/port.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             SO_REUSEPORT successfully set, or not defined,
 *             or returned ENOPROTOOPT
 *             SO_REUSEADDR successfully set
 * @retval DDS_RETCODE_UNSUPPORTED
 *             Network stack doesn't support SO_REUSEADDR and
 *             returned ENOPROTOOPT
 * @retval DDS_RETCODE_BAD_PARAMETER
 *             Bad things happened (e.g., not a socket)
 * @retval DDS_RETCODE_ERROR
 *             An unknown error occurred.
 */
DDS_EXPORT dds_return_t
ddsrt_setsockreuse(
  ddsrt_socket_t sock,
  bool reuse);

/**
 * @brief Monitor multiple sockets, waiting until one or more become ready.
 *
 * @param[in]  nfds      Highest-numbered file descriptor in any of the sets.
 * @param[in]  readfds   Set of sockets to monitor for read ready status.
 * @param[in]  writefds  Set of sockets to monitor for write ready status.
 * @param[in]  errorfds  Set of sockets to monitor for exceptional conditions.
 * @param[in]  reltime   Interval to block for sockets to become ready.
 *
 * @returns The number of sockets ready in the sets or a return code.
 */
DDS_EXPORT dds_return_t
ddsrt_select(
  int32_t nfds,
  fd_set *readfds,
  fd_set *writefds,
  fd_set *errorfds,
  dds_duration_t reltime);

#if _WIN32
/* SOCKETs on Windows are NOT integers. The nfds parameter is only there for
   compatibility, the implementation ignores it. Implicit casts will generate
   warnings though, therefore ddsrt_select is redefined to discard the
   parameter on Windows. */
#define ddsrt_select(nfds, readfds, writefds, errorfds, timeout) \
    ddsrt_select(-1, readfds, writefds, errorfds, timeout)
#endif /* _WIN32 */

/**
 * @brief Get the size of a socket address.
 *
 * @param[in]  sa  Socket address to return the size for.
 *
 * @returns Size of the socket address based on the address family, or 0 if
 *          the address family is unknown.
 */
DDS_EXPORT socklen_t
ddsrt_sockaddr_get_size(
  const struct sockaddr *const sa) ddsrt_nonnull_all;

/**
 * @brief Get the port number from a socket address.
 *
 * @param[in]  sa  Socket address to retrieve the port from.
 *
 * @return Port number in host order.
 */
DDS_EXPORT uint16_t
ddsrt_sockaddr_get_port(
  const struct sockaddr *const sa) ddsrt_nonnull_all;

/**
 * @brief Check if the given address is unspecified.
 *
 * @param[in]  sa  Socket address to check.
 *
 * @return true if the address is unspecified, false otherwise.
 */
DDS_EXPORT bool
ddsrt_sockaddr_isunspecified(
  const struct sockaddr *__restrict sa) ddsrt_nonnull_all;

/**
 * @brief Check if the given address is a loopback address.
 *
 * @param[in]  sa  Socket address to check.
 *
 * @return true if the address is a loopback address, false otherwise.
 */
DDS_EXPORT bool
ddsrt_sockaddr_isloopback(
  const struct sockaddr *__restrict sa) ddsrt_nonnull_all;

/**
 * @brief Check if given socket IP addresses reside in the same subnet.
 *
 * Checks if two socket IP addresses reside in the same subnet, considering the
 * given subnetmask. IPv6 mapped IPv4 addresses are not taken in account.
 *
 * @param[in]  sa1   First address
 * @param[in]  sa2   Second address.
 * @param[in]  mask  Subnetmask.
 *
 * @returns true if both addresses reside in the same subnet, false otherwise.
 */
DDS_EXPORT bool
ddsrt_sockaddr_insamesubnet(
  const struct sockaddr *sa1,
  const struct sockaddr *sa2,
  const struct sockaddr *mask)
ddsrt_nonnull_all;

DDS_EXPORT dds_return_t
ddsrt_sockaddrfromstr(
  int af, const char *str, void *sa);

DDS_EXPORT dds_return_t
ddsrt_sockaddrtostr(
  const void *sa, char *buf, size_t size);

#if DDSRT_HAVE_DNS
DDSRT_WARNING_MSVC_OFF(4200)
typedef struct {
  size_t naddrs;
  struct sockaddr_storage addrs[];
} ddsrt_hostent_t;
DDSRT_WARNING_MSVC_ON(4200)

/**
 * @brief Lookup addresses for given host name.
 *
 * @param[in]   name  Host name to resolve.
 * @param[in]   af    Address family, either AF_INET, AF_INET6 or AF_UNSPEC.
 * @param[out]  hentp Structure of type ddsrt_hostent_t.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *             Host name successfully resolved to address(es).
 * @retval DDS_RETCODE_HOST_NOT_FOUND
 *             Host not found.
 * @retval DDS_RETCODE_NO_DATA
 *             Valid name, no data record of requested type.
 * @retval DDS_RETCODE_ERROR
 *             Nonrecoverable error.
 * @retval DDS_RETCODE_TRY_AGAIN
 *             Nonauthoratitative host not found.
 */
DDS_EXPORT dds_return_t
ddsrt_gethostbyname(
  const char *name,
  int af,
  ddsrt_hostent_t **hentp);
#endif

#if defined (__cplusplus)
}
#endif

#endif /* DDSRT_SOCKETS_H */
