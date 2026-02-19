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
#ifndef DDSI_HANDSHAKE_H
#define DDSI_HANDSHAKE_H

#include "dds/ddsi/ddsi_entity.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_participant;
struct ddsi_proxy_participant;
struct ddsi_handshake;
struct dssi_hsadmin;

enum ddsi_handshake_state {
    STATE_HANDSHAKE_IN_PROGRESS,
    STATE_HANDSHAKE_TIMED_OUT,
    STATE_HANDSHAKE_FAILED,
    STATE_HANDSHAKE_PROCESSED,
    STATE_HANDSHAKE_SEND_TOKENS,
    STATE_HANDSHAKE_OK
};

/* The handshake will not use the related handshake object after this callback
 * was executed. This means that it can be deleted in this callback. */
typedef void (*ddsi_handshake_end_cb_t)(
    struct ddsi_handshake *handshake,
    struct ddsi_participant *pp,
    struct ddsi_proxy_participant *proxypp,
    enum ddsi_handshake_state result);

#ifdef DDS_HAS_SECURITY

#include "dds/ddsi/ddsi_security_msg.h"

/**
 * @brief Release the handshake.
 *
 * This function will decrement the refcount associated with the handshake
 * and delete the handshake when the refcount becomes 0.
 *
 * @param[in] handshake    The handshake.
 */
void ddsi_handshake_release(struct ddsi_handshake *handshake);

/**
 * @brief Handle an authentication handshake message received from the remote participant.
 *
 * During the authentication phase handshake messages are being exchanged between the local and
 * the remote participant. THis function will handle a handshake message received from a remote
 * participant.
 *
 * @param[in] handshake  The handshake.
 * @param[in] pp         The local participant.
 * @param[in] proxypp    The remote participant.
 * @param[in] msg        The handshake message received.
 */
void ddsi_handshake_handle_message(struct ddsi_handshake *handshake, const struct ddsi_participant *pp, const struct ddsi_proxy_participant *proxypp, const struct nn_participant_generic_message *msg);

/**
 * @brief Notify the handshake that crypto tokens have been received.
 *
 * The handshake could be finished at one end while the other side has not yet processed the
 * final handshake messages. The arrival of crypto tokens signals that the other side has also finished
 * processing the handshake. This function is used to signal the handshake that crypto tokens have been
 * received.
 *
 * @param[in] handshake     The handshake.
 */
void ddsi_handshake_crypto_tokens_received(struct ddsi_handshake *handshake);

/**
 * @brief Get the shared secret handle.
 *
 * During the handshake a shared secret is established which is used to encrypt
 * and decrypt the crypto token exchange messages. This function will return a
 * handle to the shared secret which will be passed to the crypto plugin to
 * determine the session keys used for the echange of the the crypto tokens.
 *
 * @param[in] handshake  The handshake.
 *
 * @returns handle to the shared sercet.
 */
int64_t ddsi_handshake_get_shared_secret(const struct ddsi_handshake *handshake);

/**
 * @brief Get the handshake handle
 *
 * This function returns the handshake handle that was returned by the authentication plugin
 * when starting the handshake.
 *
 * @param[in]  handshake  The handshake.
 *
 * @returns The handshake handle.
 */
int64_t ddsi_handshake_get_handle(const struct ddsi_handshake *handshake);

/**
 * @brief Create and start the handshake for the participants
 *
 * This function will create a handshake for the specified local
 * and remote participants when it does not yet exists. It will start the
 * handshake procedure by calling the corresponding functions of the authentication plugin.
 * The callback function is called by the handshake when to report events,
 * for example to indicate that the handshake has finished or has failed.
 *
 * @param[in] pp         The local participant.
 * @param[in] proxypp    The remote participant.
 * @param[in] callback   The callback function.
 *
 */
void ddsi_handshake_register(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp, ddsi_handshake_end_cb_t callback);

/**
 * @brief Remove the handshake associated with the specified participants.
 *
 * This function will remove the handshake from the handshake administation and release
 * the handshake. When the handshake argument is not specified the handshake is searched
 * in the handshake administation.
 *
 * @param[in] pp         The local participant.
 * @param[in] proxypp    The remote participant.
 *
 */
void ddsi_handshake_remove(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp);

/**
 * @brief Searches for the handshake associated with the specified participants
 *
 * This function will search through the handshake administration to find the handshake
 * corresponding the to specified local and remote participant.
 *
 * @param[in] pp         The local participant.
 * @param[in] proxypp    The remote participant.
 *
 * @returns The handshake
 */
struct ddsi_handshake * ddsi_handshake_find(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp);

/**
 * @brief Searches for the handshake associated with the specified participants
 *
 * This function will search through the handshake administration to find the handshake
 * corresponding the to specified local and remote participant.
 *
 * @param[in] pp         The local participant.
 * @param[in] proxypp    The remote participant.
 *
 * @returns The handshake
 */
struct ddsi_handshake * ddsi_handshake_find(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp);

/**
 * @brief Initialize the handshake administration
 *
 * @param[in] gv         The global parameters
 */
void ddsi_handshake_admin_init(struct ddsi_domaingv *gv);

/**
* @brief Stop handshake background processing.
*
* @param[in] gv         The global parameters
*/
void ddsi_handshake_admin_stop(struct ddsi_domaingv *gv);

/**
 * @brief Deinitialze the handshake administration.
 *
 * @param[in] gv         The global parameters
 */
void ddsi_handshake_admin_deinit(struct ddsi_domaingv *gv);

#else /* DDS_HAS_SECURITY */

#include "dds/ddsi/q_unused.h"

DDS_INLINE_EXPORT inline void ddsi_handshake_register(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(struct ddsi_proxy_participant *proxypp), UNUSED_ARG(ddsi_handshake_end_cb_t callback))
{
}

DDS_INLINE_EXPORT inline void ddsi_handshake_release(UNUSED_ARG(struct ddsi_handshake *handshake))
{
}

DDS_INLINE_EXPORT inline void ddsi_handshake_crypto_tokens_received(UNUSED_ARG(struct ddsi_handshake *handshake))
{
}

DDS_INLINE_EXPORT inline int64_t ddsi_handshake_get_shared_secret(UNUSED_ARG(const struct ddsi_handshake *handshake))
{
  return 0;
}

DDS_INLINE_EXPORT inline int64_t ddsi_handshake_get_handle(UNUSED_ARG(const struct ddsi_handshake *handshake))
{
  return 0;
}

DDS_INLINE_EXPORT inline void ddsi_handshake_remove(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(struct ddsi_proxy_participant *proxypp), UNUSED_ARG(struct ddsi_handshake *handshake))
{
}

DDS_INLINE_EXPORT inline struct ddsi_handshake * ddsi_handshake_find(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(struct ddsi_proxy_participant *proxypp))
{
  return NULL;
}

#endif /* DDS_HAS_SECURITY */

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_HANDSHAKE_H */
