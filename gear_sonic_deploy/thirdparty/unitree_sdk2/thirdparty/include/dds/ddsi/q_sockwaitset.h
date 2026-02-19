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
#ifndef Q_SOCKWAITSET_H
#define Q_SOCKWAITSET_H

#if defined (__cplusplus)
extern "C" {
#endif

typedef struct os_sockWaitset * os_sockWaitset;
typedef struct os_sockWaitsetCtx * os_sockWaitsetCtx;
struct ddsi_tran_conn;

/*
  Allocates a new connection waitset. The waitset is thread-safe in
  that multiple threads may add and remove connections from the wait set
  or trigger it. However only a single thread may process events from
  the wait set using the Wait and NextEvent functions in a single handling
  loop.
*/
os_sockWaitset os_sockWaitsetNew (void);

/*
  Frees the waitset WS. Any connections associated with it will
  be closed.
*/
void os_sockWaitsetFree (os_sockWaitset ws);

/*
  Triggers the waitset, from any thread.  It is level
  triggered, when called while no thread is waiting in
  os_sockWaitsetWait the trigger will cause an (early) wakeup on the
  next call to os_sockWaitsetWait.  Returns DDS_RETCODE_OK if
  successfully triggered, DDS_RETCODE_BAD_PARAMETER if an error occurs.

  Triggering a waitset may require resources and they may be counted.
  Do not trigger a waitset arbitrarily often without ensuring
  os_sockWaitsetWait is called often enough to let it release any
  resources used.

  Shared state updates preceding os_sockWaitsetTrigger are visible
  following os_sockWaitsetWait.
*/
void os_sockWaitsetTrigger (os_sockWaitset ws);

/*
  A connection may be associated with only one waitset at any time, and
  may be added to the waitset only once.  Failure to comply with this
  restriction results in undefined behaviour.

  Closing a connection associated with a waitset is handled gracefully: no
  operations will signal errors because of it.

  Returns < 0 on error, 0 if already present, 1 if added
*/
int os_sockWaitsetAdd (os_sockWaitset ws, struct ddsi_tran_conn * conn);

/*
  Drops all connections from the waitset from index onwards. Index
  0 corresponds to the first connection added to the waitset, index 1 to
  the second, etc. Behaviour is undefined when called after a successful wait
  but before all events had been enumerated.
*/
void os_sockWaitsetPurge (os_sockWaitset ws, unsigned index);

/*
  Waits until some of the connections in WS have data to be read.

  Returns a new wait set context if one or more connections have data to read.
  However, the return may be spurious (NULL) (i.e., no events)

  If a context is returned it must be enumerated before os_sockWaitsetWait
  may be called again.

  Shared state updates preceding os_sockWaitsetTrigger are visible
  following os_sockWaitsetWait.
*/
os_sockWaitsetCtx os_sockWaitsetWait (os_sockWaitset ws);

/*
  Returns the index of the next triggered connection in the
  waitset contect ctx, or -1 if the set of available events has been
  exhausted. Index 0 is the first connection added to the waitset, index
  1 the second, &c.

  Following a call to os_sockWaitsetWait on waitset that returned
  a context, one MUST enumerate all available events before
  os_sockWaitsetWait may be called again.

  If the return value is >= 0, *conn contains the connection on which
  data is available.
*/
int os_sockWaitsetNextEvent (os_sockWaitsetCtx ctx, struct ddsi_tran_conn ** conn);

/* Remove connection */
void os_sockWaitsetRemove (os_sockWaitset ws, struct ddsi_tran_conn * conn);

#if defined (__cplusplus)
}
#endif
#endif /* Q_SOCKWAITSET_H */
