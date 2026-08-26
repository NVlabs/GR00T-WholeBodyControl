/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef _DDSI_TRAN_H_
#define _DDSI_TRAN_H_

/* DDSI Transport module */

#include "dds/ddsrt/ifaddrs.h"
#include "dds/ddsrt/atomics.h"
#include "dds/ddsi/ddsi_locator.h"
#include "dds/ddsi/ddsi_config.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct nn_interface;
struct ddsi_domaingv;

/* Types supporting handles */

#define DDSI_TRAN_CONN 1
#define DDSI_TRAN_LISTENER 2

/* Flags */

#define DDSI_TRAN_ON_CONNECT 0x0001

/* Core types */

typedef struct ddsi_tran_base * ddsi_tran_base_t;
typedef struct ddsi_tran_conn * ddsi_tran_conn_t;
typedef struct ddsi_tran_listener * ddsi_tran_listener_t;
typedef struct ddsi_tran_factory * ddsi_tran_factory_t;
typedef struct ddsi_tran_qos ddsi_tran_qos_t;

/* Function pointer types */

typedef ssize_t (*ddsi_tran_read_fn_t) (ddsi_tran_conn_t, unsigned char *, size_t, bool, ddsi_locator_t *);
typedef ssize_t (*ddsi_tran_write_fn_t) (ddsi_tran_conn_t, const ddsi_locator_t *, size_t, const ddsrt_iovec_t *, uint32_t);
typedef int (*ddsi_tran_locator_fn_t) (ddsi_tran_factory_t, ddsi_tran_base_t, ddsi_locator_t *);
typedef bool (*ddsi_tran_supports_fn_t) (const struct ddsi_tran_factory *, int32_t);
typedef ddsrt_socket_t (*ddsi_tran_handle_fn_t) (ddsi_tran_base_t);
typedef int (*ddsi_tran_listen_fn_t) (ddsi_tran_listener_t);
typedef void (*ddsi_tran_free_fn_t) (ddsi_tran_factory_t);
typedef void (*ddsi_tran_peer_locator_fn_t) (ddsi_tran_conn_t, ddsi_locator_t *);
typedef void (*ddsi_tran_disable_multiplexing_fn_t) (ddsi_tran_conn_t);
typedef ddsi_tran_conn_t (*ddsi_tran_accept_fn_t) (ddsi_tran_listener_t);
typedef dds_return_t (*ddsi_tran_create_conn_fn_t) (ddsi_tran_conn_t *conn, ddsi_tran_factory_t fact, uint32_t, const struct ddsi_tran_qos *);
typedef dds_return_t (*ddsi_tran_create_listener_fn_t) (ddsi_tran_listener_t *listener, ddsi_tran_factory_t fact, uint32_t port, const struct ddsi_tran_qos *);
typedef void (*ddsi_tran_release_conn_fn_t) (ddsi_tran_conn_t);
typedef void (*ddsi_tran_close_conn_fn_t) (ddsi_tran_conn_t);
typedef void (*ddsi_tran_unblock_listener_fn_t) (ddsi_tran_listener_t);
typedef void (*ddsi_tran_release_listener_fn_t) (ddsi_tran_listener_t);
typedef int (*ddsi_tran_join_mc_fn_t) (ddsi_tran_conn_t, const ddsi_locator_t *srcip, const ddsi_locator_t *mcip, const struct nn_interface *interf);
typedef int (*ddsi_tran_leave_mc_fn_t) (ddsi_tran_conn_t, const ddsi_locator_t *srcip, const ddsi_locator_t *mcip, const struct nn_interface *interf);
typedef int (*ddsi_is_loopbackaddr_fn_t) (const struct ddsi_tran_factory *tran, const ddsi_locator_t *loc);
typedef int (*ddsi_is_mcaddr_fn_t) (const struct ddsi_tran_factory *tran, const ddsi_locator_t *loc);
typedef int (*ddsi_is_ssm_mcaddr_fn_t) (const struct ddsi_tran_factory *tran, const ddsi_locator_t *loc);
typedef int (*ddsi_is_valid_port_fn_t) (const struct ddsi_tran_factory *tran, uint32_t port);
typedef uint32_t (*ddsi_receive_buffer_size_fn_t) (const struct ddsi_tran_factory *fact);

enum ddsi_nearby_address_result {
  DNAR_DISTANT,
  DNAR_LOCAL
};

typedef enum ddsi_nearby_address_result (*ddsi_is_nearby_address_fn_t) (const ddsi_locator_t *loc, size_t ninterf, const struct nn_interface *interf, size_t *interf_idx);

enum ddsi_locator_from_string_result {
  AFSR_OK,      /* conversion succeeded */
  AFSR_INVALID, /* bogus input */
  AFSR_UNKNOWN, /* transport or hostname lookup failure */
  AFSR_MISMATCH /* recognised format, but mismatch with expected (e.g., IPv4/IPv6) */
};

typedef enum ddsi_locator_from_string_result (*ddsi_locator_from_string_fn_t) (const struct ddsi_tran_factory *tran, ddsi_locator_t *loc, const char *str);

typedef int (*ddsi_locator_from_sockaddr_fn_t) (const struct ddsi_tran_factory *tran, ddsi_locator_t *loc, const struct sockaddr *sockaddr);

typedef char * (*ddsi_locator_to_string_fn_t) (char *dst, size_t sizeof_dst, const ddsi_locator_t *loc, ddsi_tran_conn_t conn, int with_port);

typedef int (*ddsi_enumerate_interfaces_fn_t) (ddsi_tran_factory_t tran, enum ddsi_transport_selector transport_selector, ddsrt_ifaddrs_t **interfs);

/* Data types */
struct ddsi_tran_base
{
  /* Data */

  uint32_t m_port;
  uint32_t m_trantype;
  bool m_multicast;
  struct ddsi_domaingv *gv;

  /* Functions */

  ddsi_tran_handle_fn_t m_handle_fn;
};

struct ddsi_tran_conn
{
  struct ddsi_tran_base m_base;

  /* Functions */

  ddsi_tran_read_fn_t m_read_fn;
  ddsi_tran_write_fn_t m_write_fn;
  ddsi_tran_peer_locator_fn_t m_peer_locator_fn;
  ddsi_tran_disable_multiplexing_fn_t m_disable_multiplexing_fn;
  ddsi_tran_locator_fn_t m_locator_fn;

  /* Data */

  bool m_server;
  bool m_connless;
  bool m_stream;
  bool m_closed;
  ddsrt_atomic_uint32_t m_count;

  /* Relationships */

  const struct nn_interface *m_interf;
  ddsi_tran_factory_t m_factory;
  ddsi_tran_listener_t m_listener;
  ddsi_tran_conn_t m_conn;
};

struct ddsi_tran_listener
{
  struct ddsi_tran_base m_base;

  /* Functions */

  ddsi_tran_listen_fn_t m_listen_fn;
  ddsi_tran_accept_fn_t m_accept_fn;
  ddsi_tran_locator_fn_t m_locator_fn;

  /* Relationships */

  ddsi_tran_conn_t m_connections;
  ddsi_tran_factory_t m_factory;
  ddsi_tran_listener_t m_listener;
};

struct ddsi_tran_factory
{
  /* Functions */

  ddsi_tran_create_conn_fn_t m_create_conn_fn;
  ddsi_tran_create_listener_fn_t m_create_listener_fn;
  ddsi_tran_release_conn_fn_t m_release_conn_fn;
  ddsi_tran_close_conn_fn_t m_close_conn_fn;
  ddsi_tran_unblock_listener_fn_t m_unblock_listener_fn;
  ddsi_tran_release_listener_fn_t m_release_listener_fn;
  ddsi_tran_supports_fn_t m_supports_fn;
  ddsi_tran_free_fn_t m_free_fn;
  ddsi_tran_join_mc_fn_t m_join_mc_fn;
  ddsi_tran_leave_mc_fn_t m_leave_mc_fn;
  ddsi_is_loopbackaddr_fn_t m_is_loopbackaddr_fn;
  ddsi_is_mcaddr_fn_t m_is_mcaddr_fn;
  ddsi_is_ssm_mcaddr_fn_t m_is_ssm_mcaddr_fn;
  ddsi_is_nearby_address_fn_t m_is_nearby_address_fn;
  ddsi_locator_from_string_fn_t m_locator_from_string_fn;
  ddsi_locator_to_string_fn_t m_locator_to_string_fn;
  ddsi_enumerate_interfaces_fn_t m_enumerate_interfaces_fn;
  ddsi_is_valid_port_fn_t m_is_valid_port_fn;
  ddsi_receive_buffer_size_fn_t m_receive_buffer_size_fn;
  ddsi_locator_from_sockaddr_fn_t m_locator_from_sockaddr_fn;

  /* Data */

  /// Transport name, also used as prefix in string representation of locator (e.g., udp/1.2.3.4)
  const char *m_typename;

  /// Whether this is a connection-oriented transport like TCP (false), where a socket communicates
  /// with one other socket after connecting; or whether it can send to any address at any time like
  /// UDP (true).
  bool m_connless;

  /// Whether this transport deals with byte streams (TCP, true) or with datagrams (UDP, false). A
  /// byte stream forces the upper layer to do add some framing.
  bool m_stream;

  /// Whether this transport is enabled for DDS communications. Only locators mapping handled
  /// by enabled transports are taken into account when parsing discovery data.
  ///
  /// The usefulness of disabled transports is (currently) limited to running in UDP mode while using
  /// the TCP transport code as a portable means for providing a debug interface.
  bool m_enable;

  /// Whether this transport should be included in SPDP advertisements. Not all transports are
  /// created equally: those that only provide a representation for an integrated pub-sub messaging
  /// system that can be used to by-pass RTPS should not by included in anything related to SPDP.
  bool m_enable_spdp;

  /// Default SPDP address for this transport (the spec only gives an UDPv4 default one), NULL if
  /// no default address exists.
  const char *m_default_spdp_address;

  struct ddsi_domaingv *gv;

  /* Relationships */

  ddsi_tran_factory_t m_factory;
};

enum ddsi_tran_qos_purpose {
  DDSI_TRAN_QOS_XMIT_UC, // will send unicast only
  DDSI_TRAN_QOS_XMIT_MC, // may send unicast or multicast
  DDSI_TRAN_QOS_RECV_UC, // will be used for receiving unicast
  DDSI_TRAN_QOS_RECV_MC  // will be used for receiving multicast
};

struct ddsi_tran_qos
{
  enum ddsi_tran_qos_purpose m_purpose;
  int m_diffserv;
  struct nn_interface *m_interface; // only for purpose = XMIT
};

void ddsi_tran_factories_fini (struct ddsi_domaingv *gv);
void ddsi_factory_add (struct ddsi_domaingv *gv, ddsi_tran_factory_t factory);
DDS_EXPORT void ddsi_factory_free (ddsi_tran_factory_t factory);
DDS_EXPORT ddsi_tran_factory_t ddsi_factory_find (const struct ddsi_domaingv *gv, const char * type);
ddsi_tran_factory_t ddsi_factory_find_supported_kind (const struct ddsi_domaingv *gv, int32_t kind);
void ddsi_factory_conn_init (const struct ddsi_tran_factory *factory, const struct nn_interface *interf, ddsi_tran_conn_t conn);

DDS_INLINE_EXPORT inline bool ddsi_factory_supports (const struct ddsi_tran_factory *factory, int32_t kind) {
  return factory->m_supports_fn (factory, kind);
}
DDS_INLINE_EXPORT inline int ddsi_is_valid_port (const struct ddsi_tran_factory *factory, uint32_t port) {
  return factory->m_is_valid_port_fn (factory, port);
}
DDS_INLINE_EXPORT inline uint32_t ddsi_receive_buffer_size (const struct ddsi_tran_factory *factory) {
  return factory->m_receive_buffer_size_fn (factory);
}
DDS_INLINE_EXPORT inline dds_return_t ddsi_factory_create_conn (ddsi_tran_conn_t *conn, ddsi_tran_factory_t factory, uint32_t port, const struct ddsi_tran_qos *qos) {
  *conn = NULL;
  if ((qos->m_interface != NULL) != (qos->m_purpose == DDSI_TRAN_QOS_XMIT_UC || qos->m_purpose == DDSI_TRAN_QOS_XMIT_MC))
    return DDS_RETCODE_BAD_PARAMETER;
  if (!ddsi_is_valid_port (factory, port))
    return DDS_RETCODE_BAD_PARAMETER;
  return factory->m_create_conn_fn (conn, factory, port, qos);
}
DDS_INLINE_EXPORT inline dds_return_t ddsi_factory_create_listener (ddsi_tran_listener_t *listener, ddsi_tran_factory_t factory, uint32_t port, const struct ddsi_tran_qos *qos) {
  *listener = NULL;
  if (!ddsi_is_valid_port (factory, port))
    return DDS_RETCODE_BAD_PARAMETER;
  return factory->m_create_listener_fn (listener, factory, port, qos);
}

void ddsi_tran_free (ddsi_tran_base_t base);
DDS_INLINE_EXPORT inline ddsrt_socket_t ddsi_tran_handle (ddsi_tran_base_t base) {
  return base->m_handle_fn (base);
}
DDS_INLINE_EXPORT inline ddsrt_socket_t ddsi_conn_handle (ddsi_tran_conn_t conn) {
  return conn->m_base.m_handle_fn (&conn->m_base);
}
DDS_INLINE_EXPORT inline uint32_t ddsi_conn_type (const struct ddsi_tran_conn *conn) {
  return conn->m_base.m_trantype;
}
DDS_INLINE_EXPORT inline uint32_t ddsi_conn_port (const struct ddsi_tran_conn *conn) {
  return conn->m_base.m_port;
}
DDS_INLINE_EXPORT inline int ddsi_conn_locator (ddsi_tran_conn_t conn, ddsi_locator_t * loc) {
  return conn->m_locator_fn (conn->m_factory, &conn->m_base, loc);
}
DDS_INLINE_EXPORT inline ssize_t ddsi_conn_write (ddsi_tran_conn_t conn, const ddsi_locator_t *dst, size_t niov, const ddsrt_iovec_t *iov, uint32_t flags) {
  return conn->m_closed ? -1 : (conn->m_write_fn) (conn, dst, niov, iov, flags);
}
DDS_INLINE_EXPORT inline ssize_t ddsi_conn_read (ddsi_tran_conn_t conn, unsigned char * buf, size_t len, bool allow_spurious, ddsi_locator_t *srcloc) {
  return conn->m_closed ? -1 : conn->m_read_fn (conn, buf, len, allow_spurious, srcloc);
}
bool ddsi_conn_peer_locator (ddsi_tran_conn_t conn, ddsi_locator_t * loc);
void ddsi_conn_disable_multiplexing (ddsi_tran_conn_t conn);
void ddsi_conn_add_ref (ddsi_tran_conn_t conn);
void ddsi_conn_free (ddsi_tran_conn_t conn);
int ddsi_conn_join_mc (ddsi_tran_conn_t conn, const ddsi_locator_t *srcip, const ddsi_locator_t *mcip, const struct nn_interface *interf);
int ddsi_conn_leave_mc (ddsi_tran_conn_t conn, const ddsi_locator_t *srcip, const ddsi_locator_t *mcip, const struct nn_interface *interf);
void ddsi_conn_transfer_group_membership (ddsi_tran_conn_t conn, ddsi_tran_conn_t newconn);
int ddsi_conn_rejoin_transferred_mcgroups (ddsi_tran_conn_t conn);
int ddsi_is_loopbackaddr (const struct ddsi_domaingv *gv, const ddsi_locator_t *loc);
int ddsi_is_mcaddr (const struct ddsi_domaingv *gv, const ddsi_locator_t *loc);
int ddsi_is_ssm_mcaddr (const struct ddsi_domaingv *gv, const ddsi_locator_t *loc);
enum ddsi_nearby_address_result ddsi_is_nearby_address (const struct ddsi_domaingv *gv, const ddsi_locator_t *loc, size_t ninterf, const struct nn_interface *interf, size_t *interf_idx);

DDS_EXPORT enum ddsi_locator_from_string_result ddsi_locator_from_string (const struct ddsi_domaingv *gv, ddsi_locator_t *loc, const char *str, ddsi_tran_factory_t default_factory);

DDS_EXPORT int ddsi_locator_from_sockaddr (const struct ddsi_tran_factory *tran, ddsi_locator_t *loc, const struct sockaddr *sockaddr);

/*  8 for transport/
    1 for [
   48 for IPv6 hex digits (3*16) + separators
    2 for ]:
   10 for port (DDSI loc has signed 32-bit)
   11 for @ifindex
    1 for terminator
   --
   81
*/
#define DDSI_LOCSTRLEN 81

char *ddsi_xlocator_to_string (char *dst, size_t sizeof_dst, const ddsi_xlocator_t *loc);
char *ddsi_locator_to_string (char *dst, size_t sizeof_dst, const ddsi_locator_t *loc);

char *ddsi_xlocator_to_string_no_port (char *dst, size_t sizeof_dst, const ddsi_xlocator_t *loc);
char *ddsi_locator_to_string_no_port (char *dst, size_t sizeof_dst, const ddsi_locator_t *loc);

int ddsi_enumerate_interfaces (ddsi_tran_factory_t factory, enum ddsi_transport_selector transport_selector, ddsrt_ifaddrs_t **interfs);

DDS_INLINE_EXPORT inline int ddsi_listener_locator (ddsi_tran_listener_t listener, ddsi_locator_t *loc) {
  return listener->m_locator_fn (listener->m_factory, &listener->m_base, loc);
}
DDS_INLINE_EXPORT inline int ddsi_listener_listen (ddsi_tran_listener_t listener) {
  return listener->m_listen_fn (listener);
}
DDS_INLINE_EXPORT inline ddsi_tran_conn_t ddsi_listener_accept (ddsi_tran_listener_t listener) {
  return listener->m_accept_fn (listener);
}
void ddsi_listener_unblock (ddsi_tran_listener_t listener);
void ddsi_listener_free (ddsi_tran_listener_t listener);

#if defined (__cplusplus)
}
#endif

#endif
