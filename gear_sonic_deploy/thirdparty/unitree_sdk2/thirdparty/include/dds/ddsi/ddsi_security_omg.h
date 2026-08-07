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
#ifndef DDSI_OMG_SECURITY_H
#define DDSI_OMG_SECURITY_H

#include "dds/features.h"

#include "dds/ddsrt/sync.h"
#include "dds/ddsrt/avl.h"

#include "dds/ddsi/ddsi_plist.h"
#include "dds/ddsi/ddsi_entity.h"
#include "dds/ddsi/ddsi_entity_match.h"
#include "dds/ddsi/ddsi_domaingv.h"
#include "dds/ddsi/q_radmin.h"
#include "dds/ddsi/q_xmsg.h"
#include "dds/ddsrt/retcode.h"
#include "dds/ddsrt/types.h"
#include "dds/ddsrt/sync.h"
#include "dds/ddsi/ddsi_xqos.h"

#ifdef DDS_HAS_SECURITY
#include "dds/security/dds_security_api.h"
#endif

#if defined (__cplusplus)
extern "C" {
#endif

typedef enum {
  NN_RTPS_MSG_STATE_ERROR,
  NN_RTPS_MSG_STATE_PLAIN,
  NN_RTPS_MSG_STATE_ENCODED
} nn_rtps_msg_state_t;

struct ddsi_hsadmin;
struct participant_sec_attributes;
struct ddsi_proxy_participant_sec_attributes;
struct ddsi_writer_sec_attributes;
struct ddsi_reader_sec_attributes;
struct dds_security_context;
struct ddsi_topic;
struct ddsi_proxy_endpoint_common;

#ifdef DDS_HAS_SECURITY

typedef struct nn_msg_sec_info {
  unsigned encoded:1;
  unsigned use_rtps_encoding:1;
  int64_t src_pp_handle;
  int64_t dst_pp_handle;
} nn_msg_sec_info_t;

struct pp_proxypp_match {
  ddsrt_avl_node_t avlnode;
  ddsi_guid_t proxypp_guid;
  DDS_Security_ParticipantCryptoHandle proxypp_crypto_handle;
};

struct proxypp_pp_match {
  ddsrt_avl_node_t avlnode;
  ddsi_guid_t pp_guid;
  DDS_Security_ParticipantCryptoHandle pp_crypto_handle;
  DDS_Security_PermissionsHandle permissions_handle;
  DDS_Security_SharedSecretHandle shared_secret;
  bool authenticated;
};

struct participant_sec_attributes {
  ddsrt_avl_node_t avlnode;
  ddsi_guid_t pp_guid;
  DDS_Security_ParticipantSecurityAttributes attr;
  DDS_Security_IdentityHandle local_identity_handle;
  DDS_Security_PermissionsHandle permissions_handle;
  DDS_Security_ParticipantCryptoHandle crypto_handle;
  bool plugin_attr;
  ddsrt_mutex_t lock;
  ddsrt_avl_ctree_t proxy_participants;
  bool initialized;
};

struct ddsi_proxy_participant_sec_attributes {
  struct dds_security_context *sc;
  DDS_Security_IdentityHandle remote_identity_handle;
  DDS_Security_ParticipantCryptoHandle crypto_handle;
  ddsrt_mutex_t lock;
  ddsrt_avl_tree_t participants;
  bool initialized;
};

struct ddsi_writer_sec_attributes {
  DDS_Security_EndpointSecurityAttributes attr;
  DDS_Security_DatawriterCryptoHandle crypto_handle;
  bool plugin_attr;
};

struct ddsi_reader_sec_attributes {
  DDS_Security_EndpointSecurityAttributes attr;
  DDS_Security_DatareaderCryptoHandle crypto_handle;
  bool plugin_attr;
};

DDS_EXPORT struct dds_security_access_control *q_omg_participant_get_access_control(const struct ddsi_participant *pp);
DDS_EXPORT struct dds_security_authentication *q_omg_participant_get_authentication(const struct ddsi_participant *pp);
DDS_EXPORT struct dds_security_cryptography *q_omg_participant_get_cryptography(const struct ddsi_participant *pp);

void q_omg_vlog_exception(const struct ddsrt_log_cfg *lc, uint32_t cat, DDS_Security_SecurityException *exception, const char *file, uint32_t line, const char *func, const char *fmt, va_list ap);
void q_omg_log_exception(const struct ddsrt_log_cfg *lc, uint32_t cat, DDS_Security_SecurityException *exception, const char *file, uint32_t line, const char *func, const char *fmt, ...);

/**
 * @brief Check if access control is enabled for the participant.
 *
 * @param[in] pp  Participant to check.
 *
 * @returns bool  True if access control is enabled for participant
 */
bool q_omg_participant_is_access_protected(const struct ddsi_participant *pp);

/**
 * @brief Check if protection at RTPS level is enabled for the participant.
 *
 * @param[in] pp  Participant to check.
 *
 * @returns bool  True if RTPS protection enabled for participant
 */
bool q_omg_participant_is_rtps_protected(const struct ddsi_participant *pp);

/**
 * @brief Check if liveliness is protected for the participant.
 *
 * @param[in] pp  Participant to check.
 *
 * @returns bool  True  if liveliness data for participant is protected
 */
bool q_omg_participant_is_liveliness_protected(const struct ddsi_participant *pp);

/**
 * @brief Check if discovery is protected for the participant.
 *
 * @param[in] pp  Participant to check.
 *
 * @returns bool  True  if discovery data for participant is protected
 */
bool q_omg_participant_is_discovery_protected(const struct ddsi_participant *pp);

/**
 * @brief Check if security is enabled for the participant.
 *
 * @param[in] pp  Participant to check if it is secure.
 *
 * @returns bool  True if participant is secure
 */
bool q_omg_participant_is_secure(const struct ddsi_participant *pp);

/**
 * @brief Check if security is enabled for the proxy participant.
 *
 * @param[in] proxypp  Proxy participant to check if it is secure.
 *
 * @returns bool  True if proxy participant is secure
 */
bool q_omg_proxy_participant_is_secure(const struct ddsi_proxy_participant *proxypp);

/**
 * @brief Check security if it is allowed to create the participant.
 *
 * When security is enabled for this participant it is checked if the
 * participant is authenticated by checking the provided security
 * certificates. When that is ok the participant is registered with
 * access control and with cryptography. When that is all successful
 * this function return true;
 *
 * @param[in] pp         The participant to check if alloweed by security.
 * #param[in] domain_id  The domain_id
 *
 * @returns dds_return_t
 * @retval DDS_RETCODE_OK   Participant is allowed
 * @retval DDS_RETCODE_NOT_ALLOWED_BY_SECURITY
 *                          Participant is not allowed
 */
dds_return_t q_omg_security_check_create_participant(struct ddsi_participant *pp, uint32_t domain_id);

void q_omg_security_participant_set_initialized(struct ddsi_participant *pp);

bool q_omg_security_participant_is_initialized(struct ddsi_participant *pp);

/**
 * @brief Remove the participant from the security plugins.
 *
 * When the participant was registered with the security
 * plugins then this function will release the allocated
 * security resources.
 *
 * @param[in] pp  Participant to remove.
 */
void q_omg_security_deregister_participant(struct ddsi_participant *pp);

/**
 * @brief Get the identity handle associate with this participant.
 *
 * This function returns the identity handle that was created
 * when the participant was authenticated. This handle corresponds
 * with the handle returned by calling validate_local_identity on
 * the authentication plugin.
 *
 * @param[in] pp  Participant to check if it is secure.
 *
 * @returns int64_t
 * @retval !0 Identity handle associated with the participant.
 * @retval 0  Invalid handle the participant was not registered
 */
int64_t q_omg_security_get_local_participant_handle(const struct ddsi_participant *pp);

/**
 * @brief Get security info flags of the given participant.
 *
 * @param[in]  pp    Participant to get the security info from.
 * @param[out] info  The security info.
 *
 * @returns bool
 * @retval true   Security info set.
 * @retval false  Security info not set.
 */
bool q_omg_get_participant_security_info(const struct ddsi_participant *pp, nn_security_info_t *info);

/**
 * @brief Get the is_rtps_protected flag of the given remote participant.
 *
 * @param[in] pp        The participant.
 * @param[in] entityid  ID of the entity to check.
 *
 * @returns bool
 * @retval true   RTPS protected is set.
 * @retval false  RTPS protected is not set.
 */
bool q_omg_security_is_local_rtps_protected(const struct ddsi_participant *pp, ddsi_entityid_t entityid);

/**
 * @brief Check if the participant and the proxy participant
 *        have compatible security info settings.
 *
 * Associated with a secure participant is the ParticipantSecurityInfo parameter.
 * This parameter contains the setting of the security attributes and the associated
 * plugin security attributes of the secure participant.
 * This function will check if the received ParticipantSecurityInfo parameter is
 * compatible with the local ParticipantSecurityInfo parameter.
 *
 * @param[in] pp      The participant.
 * @param[in] proxypp The proxy participant.
 *
 * @returns bool
 * @retval true   The participant and the proxy participant have compatible
 *                security info settings.
 * @retval false  Otherwise.
 */
bool q_omg_is_similar_participant_security_info(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp);

/**
 * @brief Check if the parameter list key hash is protected
 *
 * @param[in] plist        The parameter list
 *
 * @returns bool  True if the parameter list key hash is protected
 */
bool q_omg_plist_keyhash_is_protected(const ddsi_plist_t *plist);

/**
 * @brief Check if the endpoint is protected
 *
 * Checks whether the provided parameter list has the flag
 * ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_VALID set. When this flag
 * is set, this implies that the remote endpoint has protection
 * enabled.
 *
 * @param[in] plist        The parameter list
 *
 * @returns bool  True if the endpoint is protected
 */
bool q_omg_is_endpoint_protected(const ddsi_plist_t *plist);

/**
 * @brief Writes the security attributes and security plugin attributes to log (category discovery)
 *
 * @param[in] gv        Global variable
 * @param[in] plist     The parameter list
 */
void q_omg_log_endpoint_protection(struct ddsi_domaingv * const gv, const ddsi_plist_t *plist);

/**
 * @brief Check if security allows to create the topic.
 *
 * This function checks with access control if is allowed to create
 * this topic for the specified domain.
 *
 * @param[in] gv          The domain global information.
 * @param[in] pp_guid     The participant guid.
 * @param[in] topic_name  The name of the  topic.
 * @param[in] qos         The topic QoS used.
 *
 * @returns bool
 * @retval true   Creation of the topic is allowed
 * @retval false  Otherwise.
 */
bool q_omg_security_check_create_topic(const struct ddsi_domaingv *gv, const ddsi_guid_t *pp_guid, const char *topic_name, const struct dds_qos *qos);

/**
 * @brief Get security info flags of the given writer.
 *
 * @param[in]  wr    Writer to get the security info from.
 * @param[out] info  The security info.
 *
 * @returns bool
 * @retval true   Security info set.
 * @retval false  Security info not set (probably unsecure writer).
 */
bool q_omg_get_writer_security_info(const struct ddsi_writer *wr, nn_security_info_t *info);

/**
 * @brief Return the builtin writer id for this writers' discovery.
 *
 * Return builtin entity id of the writer to use for the publication
 * discovery information.
 * Depending on whether the discovery is protected or not (for the
 * given writer), either the default writer or protected writer needs
 * to be used.
 *
 * @param[in] wr Writer to determine the publication writer from.
 *
 * @returns unsigned
 * @retval NN_ENTITYID_SEDP_BUILTIN_PUBLICATIONS_SECURE_WRITER
 * @retval NN_ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER
 */
unsigned determine_publication_writer(const struct ddsi_writer *wr);

/**
 * @brief Check if security allows to create the writer.
 *
 * This function checks with access control if is allowed to create
 * this writer for the specified domain.
 *
 * @param[in] pp          Participant on which the topic is being created.
 * @param[in] domain_id   The corresponding domain_id.
 * @param[in] topic_name  The name of the topic.
 * @param[in] writer_qos  The writer QoS used.
 *
 * @returns bool
 * @retval true   Creation of the writer is allowed
 * @retval false  Otherwise.
 */
bool q_omg_security_check_create_writer(struct ddsi_participant *pp, uint32_t domain_id, const char *topic_name, const struct dds_qos *writer_qos);

/**
 * @brief Register the writer with security.
 *
 * This function registers the writer with security
 * when the associated participant has security enabled.
 * The security settings associated with this writer are determined
 * and the writer is registered with cryptography when needed by
 * the security settings which indicate if payload protection and or
 * submessage protection is enabled for this writer.
 *
 * @param[in] wr  The writer to register.
 */
void q_omg_security_register_writer(struct ddsi_writer *wr);

/**
 * @brief Remove the writer from security.
 *
 * When the writer was registered with security then this function
 * will remove the writer from security which will free the allocated
 * security resource created for this writer.
 *
 * @param[in] wr  The writer to remove.
 */
void q_omg_security_deregister_writer(struct ddsi_writer *wr);

/**
 * @brief Get security info flags of the given reader.
 *
 * @param[in]  rd    Reader to get the security info from.
 * @param[out] info  The security info.
 *
 * @returns bool
 * @retval true   Security info set.
 * @retval false  Security info not set (probably unsecure reader).
 */
bool q_omg_get_reader_security_info(const struct ddsi_reader *rd, nn_security_info_t *info);

/**
 * @brief Return the builtin writer id for this readers' discovery.
 *
 * Return builtin entity id of the writer to use for the subscription
 * discovery information.
 * Depending on whether the discovery is protected or not (for the
 * given reader), either the default writer or protected writer needs
 * to be used.
 *
 * @param[in] rd Reader to determine the subscription writer from.
 *
 * @returns unsigned
 * @retval NN_ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_SECURE_WRITER
 * @retval NN_ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER
 */
unsigned determine_subscription_writer(const struct ddsi_reader *rd);

#ifdef DDS_HAS_TOPIC_DISCOVERY
/**
 * @brief Return the builtin writer id for topic discovery.
 *
 * Return builtin entity id of the writer to use for the topic
 * discovery information.
 *
 * @param[in] tp Topic to determine the writer from.
 *
 * @returns unsigned
 * @retval NN_ENTITYID_SEDP_BUILTIN_TOPIC_WRITER
 */
unsigned determine_topic_writer(const struct ddsi_topic *tp);
#endif /* DDS_HAS_TOPIC_DISCOVERY */

/**
 * @brief Check if security allows to create the reader.
 *
 * This function checks with access control if is allowed to create
 * this reader for the specified domain.
 *
 * @param[in] pp          Participant on which the topic is being created.
 * @param[in] domain_id   The corresponding domain_id.
 * @param[in] topic_name  The name of the topic.
 * @param[in] reader_qos  The reader QoS used.
 *
 * @returns bool
 * @retval true   Creation of the writer is allowed
 * @retval false  Otherwise.
 */
bool q_omg_security_check_create_reader(struct ddsi_participant *pp, uint32_t domain_id, const char *topic_name, const struct dds_qos *reader_qos);

/**
 * @brief Register the reader with security.
 *
 * This function registers the reader with security
 * when the associated participant has security enabled.
 * The security settings associated with this reader are determined
 * and the reader is registered with cryptography when needed by
 * the security settings which indicate if submessage protection is
 *  enabled for this reader.
 *
 * @param[in] rd  The reader to register.
 */
void q_omg_security_register_reader(struct ddsi_reader *rd);

/**
 * @brief Remove the reader from security.
 *
 * When the reader was registered with security then this function
 * will remove the reader from security which will free the allocated
 * security resource created for this reader.
 *
 * @param[in] rd  The reader to remove.
 */
void q_omg_security_deregister_reader(struct ddsi_reader *rd);

/**
 * @brief Determine if the proxy participant is allowed to be deleted
 *        by the given writer.
 *
 * If an proxy participant is authenticated, it is only allowed to
 * to deleted when a dispose is received from the proper protected
 * discovery writer.
 *
 * @param[in] gv           Used for tracing.
 * @param[in] guid         Guid of the proxy participant to be deleted.
 * @param[in] pwr_entityid Writer that send the dispose.
 *
 * @returns bool
 * @retval true   The proxy participant may be deleted.
 * @retval false  The proxy participant may not be deleted by this writer.
 */
bool is_proxy_participant_deletion_allowed(struct ddsi_domaingv * const gv, const struct ddsi_guid *guid, const ddsi_entityid_t pwr_entityid);

/**
 * @brief Determine if the messages, related to the given remote
 * entity, are RTPS protected or not.
 *
 * @param[in] proxy_pp  Related proxy participant.
 * @param[in] entityid  ID of the entity to check.
 *
 * @returns bool
 * @retval true   The entity messages are RTPS protected.
 * @retval false  The entity messages are not RTPS protected.
 */
bool q_omg_security_is_remote_rtps_protected(const struct ddsi_proxy_participant *proxy_pp, ddsi_entityid_t entityid);

/**
 * @brief Set security information, depending on plist, into the given
 * proxy participant.
 *
 * @param[in] proxypp  Proxy participant to set security info on.
 * @param[in] plist    Paramater list, possibly contains security info.
 */
void set_proxy_participant_security_info(struct ddsi_proxy_participant *proxypp, const ddsi_plist_t *plist);

/**
 * @brief Determine if the messages, related to the given remote
 * entity, are RTPS protected or not.
 *
 * @param[in] pp       The participant.
 * @param[in] entityid ID of the entity to check.
 *
 * @returns bool
 * @retval true   The entity messages are RTPS protected.
 * @retval false  The entity messages are not RTPS protected.
 */
bool q_omg_security_is_local_rtps_protected(const struct ddsi_participant *pp, ddsi_entityid_t entityid);

/**
 * @brief Check if the participant allows communication with unauthenticated
 *        participants
 *
 * @param[in] pp  The participant.
 *
 * @returns bool
 * @retval true   The participant allows unauthenticated communication
 * @retval false  Otherwise.
 */
bool q_omg_participant_allow_unauthenticated(struct ddsi_participant *pp);

/**
 * @brief Initialize the proxy participant security attributes
 *
 * @param[in] proxypp  The proxy participant.
 *
 */
void q_omg_security_init_remote_participant(struct ddsi_proxy_participant *proxypp);

void q_omg_security_remote_participant_set_initialized(struct ddsi_proxy_participant *proxypp);

bool q_omg_security_remote_participant_is_initialized(struct ddsi_proxy_participant *proxypp);

/**
 * @brief Registers the matched proxy participant with the crypto plugin
 *
 * When the proxy participant is authenticated and allowed by access control then the match between the local and
 * the remote participant must be registered with the cypto factory provided by the crypto plugin. The
 * shared secret handle obtained from the authentication phase and the permission handle returned when validating
 * the proxy participant with access control plugin have to be provided.
 *
 *
 * @param[in] pp                 The participant.
 * @param[in] proxypp            The proxy participant.
 * @param[in] shared_secret      The shared_secret handle.
 *
 * @returns bool
 * @retval true    The proxy participant is allowed.
 * @retval false   The proxy participant is not allowed.
 */
bool q_omg_security_register_remote_participant(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp, int64_t shared_secret);

/**
 * @brief Sets the matching participant and proxy participant as authorized.
 *
 * When the authentication handshake has finished successfully and the
 * volatile secure readers and writers are matched then with this function
 * the matching local and remote participant are set to authenticated which
 * allows the crypto tokens to be exchanged and the corresponding entities
 * be matched.
 *
 * @param[in] pp                 The participant.
 * @param[in] proxypp            The proxy participant.
 */
void q_omg_security_set_remote_participant_authenticated(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp);

/**
 * @brief Removes a registered proxy participant from administation of the authentication,
 *        access control and crypto plugins.
 *
 * @param[in] proxypp            The proxy participant.
 */
void q_omg_security_deregister_remote_participant(struct ddsi_proxy_participant *proxypp);

/**
 * @brief Generate and send the crypto tokens needed for encoding RTPS messages.
 *
 * When the security settings indicate that RTPS message encoding or signing is
 * configured for the participant then this function will ask the cypto echange for
 * the corresponding cypto tokens and send these to the proxy participant.
 *
 * @param[in] pp                 The participant.
 * @param[in] proxypp            The proxy participant.
 */
void q_omg_security_participant_send_tokens(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp);

/**
 * @brief Get the cypto handle associated with the proxy participant.
 *
 * This function returns the handle which is the association between
 * the proxy participant and the crypto plugin. This handle is created
 * when the proxy participant is registered with the crypto plugin.
 *
 * @param[in] proxypp            The proxy participant.
 *
 * @returns handle
 * @retval !0  Valid crypto handle associated with the proxy participant.
 * @retval 0   Otherwise.
 */
int64_t q_omg_security_get_remote_participant_handle(struct ddsi_proxy_participant *proxypp);

/**
 * @brief Set the crypto tokens used for the encryption and decryption of RTPS messages.
 *
 * The remote participant  will send the crypto tokens when the security settings determine that the
 * communication between the participants must be secure. These tokens are used for the necryption and
 * decryption of complete RTPS messages. When these tokens are received this function will register these tokens
 * with the crypto plugin. The crypto plugin will return a crypto handle that will be used to associate the
 * stored tokens with the remote participant.
 *
 * @param[in] pp        The local participant.
 * @param[in] proxypp   The remote participant.
 * @param[in] tokens    The crypto token received from the remote participant for the local participant.
 */
void q_omg_security_set_participant_crypto_tokens(struct ddsi_participant *pp, struct ddsi_proxy_participant *proxypp, const nn_dataholderseq_t *tokens);

/**
 * @brief Check if the writer has the is_discovery_protected flag set
 *
 * @param[in] wr        The local writer.
 *
 * @returns bool  True if the writer has the is_discovery_protected flag set
 */
bool q_omg_writer_is_discovery_protected(const struct ddsi_writer *wr);

/**
 * @brief Check if the writer has the is_submessage_protected flag set
 *
 * @param[in] wr        The local writer.
 *
 * @returns bool  True if the writer has the is_submessage_protected flag set
 */
bool q_omg_writer_is_submessage_protected(const struct ddsi_writer *wr);

/**
 * @brief Check if the writer has the is_payload_protected flag set
 *
 * @param[in] wr        The local writer.
 *
 * @returns bool  True if the writer has the is_payload_protected flag set
 */
bool q_omg_writer_is_payload_protected(const struct ddsi_writer *wr);

/**
 * @brief Check if the remote writer is allowed to communicate with endpoints of the
 *        local participant.
 *
 * This function will check with the access control plugin if the remote writer
 * is allowed to communicate with this participant.
 *
 * @param[in] pwr       The remote writer.
 * @param[in] domain_id The domain id.
 * @param[in] pp        The local participant.
 *
 * @returns bool
 * @retval true   The remote writer is allowed to communicate.
 * @retval false  Otherwise.
 */
bool q_omg_security_check_remote_writer_permissions(const struct ddsi_proxy_writer *pwr, uint32_t domain_id, struct ddsi_participant *pp);

/**
 * @brief Check it the remote writer is allowed to communicate with the local reader.
 *
 * When a remote writer is allowed by access control it has to be checked if the remote
 * writer is allowed to communicate with a particular local reader. This function will
 * check if the provided security end-point attributes are compatible, When the security
 * attributes are compatible then the function will register the reader and remote writer
 * match with the crypto factory and will also ask the crypto exchange to generate the
 * crypto tokens associate with the local reader which will be sent to the remote entity.
 * Note that the reader crypto tokens are used to encrypt the reader specific submessages
 * when submessage encoding or signing is configured.
 *
 * @param[in] rd   The local reader.
 * @param[in] pwr  The remote writer.
 * @param[out] crypto_handle  The crypto handle associated with the match.
 *
 * @returns bool
 * @retval true   The local reader and remote writer are allowed to communicate.
 * @retval false  Otherwise.
 */
bool q_omg_security_match_remote_writer_enabled(struct ddsi_reader *rd, struct ddsi_proxy_writer *pwr, int64_t *crypto_handle);

/**
 * @brief Release the security information associated with the match between a reader and
 * a remote writer.
 *
 * This function releases the security resources that were allocated for this reader and remote
 * writer match. For example it will release the security tokens that where associated with this
 * reader and the remote writer.
 *
 * @param[in] gv       The global parameters.
 * @param[in] rd_guid  The guid of the reader.
 * @param[in] match    The reader-proxy_writer match.
 */
void q_omg_security_deregister_remote_writer_match(const struct ddsi_domaingv *gv, const ddsi_guid_t *rd_guid, struct ddsi_rd_pwr_match *match);

/**
 * @brief Set the crypto tokens used for the secure communication from the remote writer to the reader.
 *
 * The remote writer instance will send the crypto tokens when the security settings determine that the
 * communication between the remote writer and the reader must be secure. When these tokens are received
 * this function will register these tokens with the crypto plugin and set the corresponding crypto handle returned
 * by the crypto plugin which is then used for decrypting messages received from that remote writer to the reader.
 *
 * @param[in] rd        The local reader.
 * @param[in] pwr_guid  The guid of the remote writer.
 * @param[in] tokens    The crypto token received from the remote writer for the reader.
 */
void q_omg_security_set_remote_writer_crypto_tokens(struct ddsi_reader *rd, const ddsi_guid_t *pwr_guid, const nn_dataholderseq_t *tokens);

/**
 * @brief Release all the security resources associated with the remote writer.
 *
 * Cleanup security resource associated with the remote writer.
 *
 * @param[in] pwr       The remote writer.
 */
void q_omg_security_deregister_remote_writer(const struct ddsi_proxy_writer *pwr);

/**
 * @brief Set security information, depending on plist and proxy participant,
 * into the given proxy reader.
 *
 * @param[in] prd      Proxy reader to set security info on.
 * @param[in] plist    Paramater list, possibly contains security info.
 */
void set_proxy_reader_security_info(struct ddsi_proxy_reader *prd, const ddsi_plist_t *plist);

/**
 * @brief Determine the security settings associated with the remote reader.
 *
 * From the security information contained in the parameter list from the remote reader
 * the corresponding security settings are determined and returned in the info parameter.
 *
 * @param[in] prd       The remote reader.
 * @param[in] plist     The parameter list from the remote reader.
 * @param[out] info     The security settings associated with the remote reader.
 */
void q_omg_get_proxy_reader_security_info(struct ddsi_proxy_reader *prd, const ddsi_plist_t *plist, nn_security_info_t *info);

/**
 * @brief Check if the reader has the is_discovery_protected flag set
 *
 * @param[in] rd        The local reader.
 *
 * @returns bool  True if the reader has the is_discovery_protected flag set
 */
bool q_omg_reader_is_discovery_protected(const struct ddsi_reader *rd);

/**
 * @brief Check if the reader has the is_submessage_protected flag set
 *
 * @param[in] rd        The local reader.
 *
 * @returns bool  True if the reader has the is_submessage_protected flag set
 */
bool q_omg_reader_is_submessage_protected(const struct ddsi_reader *rd);

/**
 * @brief Check if the remote reader is allowed to communicate with endpoints of the
 *        local participant.
 *
 * This function will check with the access control plugin if the remote reader
 * is allowed to communicate with this participant.
 *
 * @param[in] prd         The remote reader.
 * @param[in] domain_id   The domain id.
 * @param[in] pp          The local participant.
 * @param[out] relay_only The "relay_only" value returned by the access control
 *                        operation check_remote_datareader()
 *
 * @returns bool
 * @retval true   The remote reader is allowed to communicate.
 * @retval false  Otherwise; relay_only is unspecified.
 */
bool q_omg_security_check_remote_reader_permissions(const struct ddsi_proxy_reader *prd, uint32_t domain_id, struct ddsi_participant *pp, bool *relay_only);


/**
 * @brief Set security information, depending on plist and proxy participant,
 * into the given proxy endpoint.
 *
 * @param[in] entity            The endpoint common attributes.
 * @param[in] proxypp_sec_info  The security info of the proxy participant
 * @param[in] plist             Paramater list which may contain security info.
 * @param[in] info              The proxy endpoint security info to be set.
 */
void q_omg_get_proxy_endpoint_security_info(const struct ddsi_entity_common *entity, nn_security_info_t *proxypp_sec_info, const ddsi_plist_t *plist, nn_security_info_t *info);

/**
 * @brief Check it the local writer is allowed to communicate with the remote reader.
 *
 * When a remote reader is allowed by accessstruct dds_security_garbage control it has to be checked if the local
 * writer is allowed to communicate with a particular local writer. This function will
 * check if the provided security end-point attributes are compatible, When the security
 * attributes are compatible then the function will register the writer and remote reader
 * match with the crypto factory and will also ask the crypto exchange to generate the
 * crypto tokens associate with the local writer which will be sent to the remote entity.
 * Note that the writer crypto tokens are used to encrypt the writer specific submessages
 * when submessage encoding or signing is configured and also the crypto tokens used
 * for encoding the payload of data or datafrag messages.
 *
 * @param[in] wr              The local writer.
 * @param[in] prd             The remote reader.
 * @param[in] relay_only      The "relay_only" returned by access control
 *                            operation check_remote_datareader()
 * @param[out] crypto_handle  The crypto handle associated with the match.
 *
 * @returns bool
 * @retval true   The local writer and remote reader are allowed to communicate.
 * @retval false  Otherwise.
 */
bool q_omg_security_match_remote_reader_enabled(struct ddsi_writer *wr, struct ddsi_proxy_reader *prd, bool relay_only, int64_t *crypto_handle);

/**
 * @brief Release the security information associated with the match between a writer and
 * a remote reader.
 *
 * This function releases the security resources that were allocated for this writer and remote
 * reader match. For example it will release the security tokens that where associated with this
 * writer and the remote reader.
 *
 * @param[in] gv       The global parameters.
 * @param[in] wr_guid  The guid of the writer.
 * @param[in] match    The writer-proxy_reader match.
 */
void q_omg_security_deregister_remote_reader_match(const struct ddsi_domaingv *gv, const ddsi_guid_t *wr_guid, struct ddsi_wr_prd_match *match);

/**
 * @brief Set the crypto tokens used for the secure communication from the remote reader to the writer.
 *
 * The remote reader instance will send the crypto tokens when the security settings determine that the
 * communication between the remote reader and the writer must be secure. When these tokens are received
 * this function will register these tokens with the crypto plugin and set the corresponding crypto handle returned
 * by the crypto plugin which is then used for decrypting messages received from that remote reader to the writer.
 *
 * @param[in] wr        The local writer.
 * @param[in] prd_guid  The guid of the remote reader.
 * @param[in] tokens    The crypto token received from the remote reader for the writer.
 */
void q_omg_security_set_remote_reader_crypto_tokens(struct ddsi_writer *wr, const ddsi_guid_t *prd_guid, const nn_dataholderseq_t *tokens);

/**
 * @brief Release all the security resources associated with the remote reader.
 *
 * Cleanup security resource associated with the remote reader.
 *
 * @param[in] prd       The remote reader.
 */
void q_omg_security_deregister_remote_reader(const struct ddsi_proxy_reader *prd);

/**
 * @brief Encode RTPS message.
 *
 * @param[in]     src_handle  Security handle of data source.
 * @param[in]     src_guid    GUID of the entity data source.
 * @param[in]     src_buf     Original RTPS message.
 * @param[in]     src_len     Original RTPS message size.
 * @param[out]    dst_buf     Encoded RTPS message.
 * @param[out]    dst_len     Encoded RTPS message size.
 * @param[in]     dst_handle  Security handle of data destination.
 *
 * @returns bool
 * @retval true   Encoding succeeded.
 * @retval false  Encoding failed.
 */
bool
q_omg_security_encode_rtps_message(
  const struct ddsi_domaingv *gv,
  int64_t                 src_handle,
  const ddsi_guid_t      *src_guid,
  const unsigned char    *src_buf,
  size_t                  src_len,
  unsigned char         **dst_buf,
  size_t                 *dst_len,
  int64_t                 dst_handle);

/**
 * @brief Encode payload when necessary.
 *
 * When encoding is necessary, *buf will be allocated and the vec contents
 * will change to point to that buffer.
 * It is expected that the vec contents is always aliased.
 *
 * If no encoding is necessary, nothing changes.
 *
 * encoding(    not needed) -> return( true), vec(untouched), buf(NULL)
 * encoding(needed&success) -> return( true), vec( buf(new))
 * encoding(needed&failure) -> return(false), vec(untouched), buf(NULL)
 *
 * @param[in]     wr   Writer that writes the payload.
 * @param[in,out] vec  An iovec that contains the payload.
 * @param[out]    buf  Buffer to contain the encoded payload.
 *
 * @returns bool
 * @retval true   Encoding succeeded or not necessary. Either way, vec
 *                contains the payload that should be send.
 * @retval false  Encoding was necessary, but failed.
 */
bool encode_payload(struct ddsi_writer *wr, ddsrt_iovec_t *vec, unsigned char **buf);

/**
 * @brief Decode the payload of a Data submessage.
 *
 * When decoding is necessary, the payloadp memory will be replaced
 * by the decoded payload. This means that the original submessage
 * now contains payload that can be deserialized.
 *
 * If no decoding is necessary, nothing changes.
 *
 * @param[in]     gv          Global information.
 * @param[in]     sampleinfo  Sample information.
 * @param[in,out] payloadp    Pointer to payload memory.
 * @param[in]     payloadsz   Size of payload.
 * @param[in,out] submsg_len  Size of submessage.
 *
 * @returns bool
 * @retval true   Decoding succeeded or not necessary. Either way, payloadp
 *                contains the data that should be deserialized.
 * @retval false  Decoding was necessary, but failed.
 */
bool decode_Data(const struct ddsi_domaingv *gv, struct nn_rsample_info *sampleinfo, unsigned char *payloadp, uint32_t payloadsz, size_t *submsg_len);

/**
 * @brief Decode the payload of a DataFrag submessage.
 *
 * When decoding is necessary, the payloadp memory will be replaced
 * by the decoded payload. This means that the original submessage
 * now contains payload that can be deserialized.
 *
 * If no decoding is necessary, nothing changes.
 *
 * @param[in]     gv          Global information.
 * @param[in]     sampleinfo  Sample information.
 * @param[in,out] payloadp    Pointer to payload memory.
 * @param[in]     payloadsz   Size of payload.
 * @param[in,out] submsg_len  Size of submessage.
 *
 * @returns bool
 * @retval true   Decoding succeeded or not necessary. Either way, payloadp
 *                contains the data that should be deserialized.
 * @retval false  Decoding was necessary, but failed.
 */
bool decode_DataFrag(const struct ddsi_domaingv *gv, struct nn_rsample_info *sampleinfo, unsigned char *payloadp, uint32_t payloadsz, size_t *submsg_len);

/**
 * @brief Encode datareader submessage when necessary.
 *
 * When encoding is necessary, the original submessage will be replaced
 * by a new encoded submessage.
 * If the encoding fails, the original submessage will be removed.
 *
 * If no encoding is necessary, nothing changes.
 *
 * @param[in,out] msg       Complete message.
 * @param[in,out] sm_marker Submessage location within message.
 * @param[in]     pwr       Writer for which the message is intended.
 * @param[in]     rd_guid   Origin reader guid.
 */
void encode_datareader_submsg(struct nn_xmsg *msg, struct nn_xmsg_marker sm_marker, const struct ddsi_proxy_writer *pwr, const struct ddsi_guid *rd_guid);

/**
 * @brief Encode datawriter submessage when necessary.
 *
 * When encoding is necessary, the original submessage will be replaced
 * by a new encoded submessage.
 * If the encoding fails, the original submessage will be removed.
 *
 * If no encoding is necessary, nothing changes.
 *
 * @param[in,out] msg       Complete message.
 * @param[in,out] sm_marker Submessage location within message.
 * @param[in]     wr        Origin writer guid.
 */
void encode_datawriter_submsg(struct nn_xmsg *msg, struct nn_xmsg_marker sm_marker, struct ddsi_writer *wr);

/**
 * @brief Check if given submessage is properly decoded.
 *
 * When decoding is necessary, it should be checked if a plain submessage was
 * actually decoded. Otherwise data can be injected just by inserting a plain
 * submessage directly.
 *
 * @param[in] e         Entity information.
 * @param[in] c         Proxy endpoint information.
 * @param[in] proxypp   Related proxy participant.
 * @param[in] rst       Receiver information.
 * @param[in] prev_smid Previously handled submessage ID.
 *
 * @returns bool
 * @retval true   Decoding succeeded or was not necessary.
 * @retval false  Decoding was necessary, but not detected.
 */
bool
validate_msg_decoding(
  const struct ddsi_entity_common *e,
  const struct ddsi_proxy_endpoint_common *c,
  const struct ddsi_proxy_participant *proxypp,
  const struct receiver_state *rst,
  SubmessageKind_t prev_smid);

/**
 * @brief Decode not only SecPrefix, but also the SecBody and SecPostfix
 * sub-messages.
 *
 * When encrypted, the original SecBody will be replaced by the decrypted
 * submessage. Then the normal sequence can continue as if there was no
 * encrypted data.
 *
 * @param[in]     rst         Receiver information.
 * @param[in,out] submsg      Pointer to SecPrefix/(SecBody|Submsg)/SecPostfix.
 * @param[in]     submsg_size Size of SecPrefix submessage.
 * @param[in]     msg_end     End of the complete message.
 * @param[in]     src_prefix  Prefix of the source entity.
 * @param[in]     dst_prefix  Prefix of the destination entity.
 * @param[in]     byteswap    Do the bytes need swapping?
 *
 * @returns bool
 * @retval true   Decoding succeeded.
 * @retval false  Decoding failed.
 */
bool
decode_SecPrefix(
  const struct receiver_state *rst,
  unsigned char *submsg,
  size_t submsg_size,
  unsigned char * const msg_end,
  const ddsi_guid_prefix_t * const src_prefix,
  const ddsi_guid_prefix_t * const dst_prefix,
  int byteswap);

/**
 * @brief Decode the RTPS message.
 *
 * When encrypted, the original buffers and information will be replaced
 * by the decrypted RTPS message. Then the normal sequence can continue
 * as if there was no encrypted data.
 *
 * @param[in]     thrst         Thread information.
 * @param[in]     gv          Global information.
 * @param[in,out] rmsg        Message information.
 * @param[in,out] hdr         Message header.
 * @param[in,out] buff        Message buffer.
 * @param[in,out] sz          Message size.
 * @param[in]     rbpool      Buffers pool.
 * @param[in]     isstream    Is message a stream variant?
 *
 * @returns nn_rtps_msg_state_t
 * @retval NN_RTPS_MSG_STATE_PLAIN    No decoding was necessary.
 * @retval NN_RTPS_MSG_STATE_ENCODED  Decoding succeeded.
 * @retval NN_RTPS_MSG_STATE_ERROR    Decoding failed.
 */
nn_rtps_msg_state_t decode_rtps_message(struct thread_state * const thrst, struct ddsi_domaingv *gv, struct nn_rmsg **rmsg, Header_t **hdr, unsigned char **buff, size_t *sz, struct nn_rbufpool *rbpool, bool isstream);

/**
 * @brief Send the RTPS message securely.
 *
 * @param[in]     conn          Connection to use.
 * @param[in]     dst           Possible destination information.
 * @param[in]     niov          Number of io vectors.
 * @param[in]     iov           Array of io vectors.
 * @param[in]     flags         Connection write flags.
 * @param[in,out] msg_len       Submessage containing length.
 * @param[in]     dst_one       Is there only one specific destination?
 * @param[in]     sec_info      Security information for handles.
 * @param[in]     conn_write_cb Function to call to do the actual writing.
 *
 * @returns ssize_t
 * @retval negative/zero    Something went wrong.
 * @retval positive         Secure writing succeeded.
 */
ssize_t
secure_conn_write(
    const struct ddsi_domaingv *gv,
    ddsi_tran_conn_t conn,
    const ddsi_locator_t *dst,
    size_t niov,
    const ddsrt_iovec_t *iov,
    uint32_t flags,
    MsgLen_t *msg_len,
    bool dst_one,
    nn_msg_sec_info_t *sec_info,
    ddsi_tran_write_fn_t conn_write_cb);


/**
 * @brief Loads the security plugins with the given configuration.
 *        This function tries to load the plugins only once. Returns the same
 *        result on subsequent calls.
 *        It logs the reason and returns error if can not load a plugin.
 *
 * @param[in] qos   Participant qos which owns the Property list
 *                             that contains security configurations and
 *                             plugin properties that are required for loading libraries
 * @returns dds_return_t
 * @retval DDS_RETCODE_OK   All plugins are successfully loaded
 * @retval DDS_RETCODE_ERROR  One or more security plugins are not loaded.
 */
dds_return_t q_omg_security_load( struct dds_security_context *security_context, const dds_qos_t *qos, struct ddsi_domaingv *gv );


void q_omg_security_init( struct ddsi_domaingv *gv );

void q_omg_security_stop (struct ddsi_domaingv *gv);

void q_omg_security_deinit (struct dds_security_context *sc );

void q_omg_security_free (struct ddsi_domaingv *gv);

bool q_omg_is_security_loaded(  struct dds_security_context *sc );

#else /* DDS_HAS_SECURITY */

#include "dds/ddsi/q_unused.h"

DDS_INLINE_EXPORT inline bool q_omg_security_enabled(void)
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_participant_is_access_protected(UNUSED_ARG(const struct ddsi_participant *pp))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_participant_is_rtps_protected(UNUSED_ARG(const struct ddsi_participant *pp))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_participant_is_liveliness_protected(UNUSED_ARG(const struct ddsi_participant *pp))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_participant_is_discovery_protected(UNUSED_ARG(const struct ddsi_participant *pp))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_participant_is_secure(UNUSED_ARG(const struct ddsi_participant *pp))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_proxy_participant_is_secure(UNUSED_ARG(const struct ddsi_proxy_participant *proxypp))
{
  return false;
}

DDS_INLINE_EXPORT inline unsigned determine_subscription_writer(UNUSED_ARG(const struct ddsi_reader *rd))
{
  return NN_ENTITYID_SEDP_BUILTIN_SUBSCRIPTIONS_WRITER;
}

DDS_INLINE_EXPORT inline unsigned determine_publication_writer(UNUSED_ARG(const struct ddsi_writer *wr))
{
  return NN_ENTITYID_SEDP_BUILTIN_PUBLICATIONS_WRITER;
}

#ifdef DDS_HAS_TOPIC_DISCOVERY
DDS_INLINE_EXPORT inline unsigned determine_topic_writer(UNUSED_ARG(const struct ddsi_topic *tp))
{
  return NN_ENTITYID_SEDP_BUILTIN_TOPIC_WRITER;
}
#endif

DDS_INLINE_EXPORT inline bool is_proxy_participant_deletion_allowed(UNUSED_ARG(struct ddsi_domaingv * const gv), UNUSED_ARG(const struct ddsi_guid *guid), UNUSED_ARG(const ddsi_entityid_t pwr_entityid))
{
  return true;
}

DDS_INLINE_EXPORT inline bool q_omg_is_similar_participant_security_info(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(struct ddsi_proxy_participant *proxypp))
{
  return true;
}

DDS_INLINE_EXPORT inline bool q_omg_participant_allow_unauthenticated(UNUSED_ARG(struct ddsi_participant *pp))
{
  return true;
}

DDS_INLINE_EXPORT inline bool q_omg_security_check_create_participant(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(uint32_t domain_id))
{
  return true;
}

DDS_INLINE_EXPORT inline void q_omg_security_deregister_participant(UNUSED_ARG(struct ddsi_participant *pp))
{
}

DDS_INLINE_EXPORT inline bool q_omg_security_check_create_topic(UNUSED_ARG(const struct ddsi_domaingv *gv), UNUSED_ARG(const ddsi_guid_t *pp_guid), UNUSED_ARG(const char *topic_name), UNUSED_ARG(const struct dds_qos *qos))
{
  return true;
}

DDS_INLINE_EXPORT inline int64_t q_omg_security_get_local_participant_handle(UNUSED_ARG(const struct ddsi_participant *pp))
{
  return 0;
}

DDS_INLINE_EXPORT inline bool q_omg_security_check_create_writer(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(uint32_t domain_id), UNUSED_ARG(const char *topic_name), UNUSED_ARG(const struct dds_qos *writer_qos))
{
  return true;
}

DDS_INLINE_EXPORT inline void q_omg_security_register_writer(UNUSED_ARG(struct ddsi_writer *wr))
{
}

DDS_INLINE_EXPORT inline void q_omg_security_deregister_writer(UNUSED_ARG(struct ddsi_writer *wr))
{
}

DDS_INLINE_EXPORT inline bool q_omg_security_check_create_reader(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(uint32_t domain_id), UNUSED_ARG(const char *topic_name), UNUSED_ARG(const struct dds_qos *reader_qos))
{
  return true;
}

DDS_INLINE_EXPORT inline void q_omg_security_register_reader(UNUSED_ARG(struct ddsi_reader *rd))
{
}

DDS_INLINE_EXPORT inline void q_omg_security_deregister_reader(UNUSED_ARG(struct ddsi_reader *rd))
{
}

DDS_INLINE_EXPORT inline bool q_omg_security_is_remote_rtps_protected(UNUSED_ARG(const struct ddsi_proxy_participant *proxypp), UNUSED_ARG(ddsi_entityid_t entityid))
{
  return false;
}

DDS_INLINE_EXPORT inline void q_omg_security_init_remote_participant(UNUSED_ARG(struct ddsi_proxy_participant *proxypp))
{
}

DDS_INLINE_EXPORT inline int64_t q_omg_security_check_remote_participant_permissions(UNUSED_ARG(uint32_t domain_id), UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(struct ddsi_proxy_participant *proxypp))
{
  return 0LL;
}

DDS_INLINE_EXPORT inline bool q_omg_security_register_remote_participant(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(struct ddsi_proxy_participant *proxypp), UNUSED_ARG(int64_t identity_handle), UNUSED_ARG(int64_t shared_secret))
{
  return true;
}

DDS_INLINE_EXPORT inline void q_omg_security_deregister_remote_participant(UNUSED_ARG(struct ddsi_proxy_participant *proxypp))
{
}

DDS_INLINE_EXPORT inline void q_omg_security_participant_send_tokens(UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(struct ddsi_proxy_participant *proxypp))
{
}

DDS_INLINE_EXPORT inline int64_t q_omg_security_get_remote_participant_handle(UNUSED_ARG(struct ddsi_proxy_participant *proxypp))
{
  return 0;
}

DDS_INLINE_EXPORT inline bool q_omg_security_match_remote_writer_enabled(UNUSED_ARG(struct ddsi_reader *rd), UNUSED_ARG(struct ddsi_proxy_writer *pwr), UNUSED_ARG(int64_t *crypto_handle))
{
  return true;
}

DDS_INLINE_EXPORT inline bool q_omg_security_match_remote_reader_enabled(UNUSED_ARG(struct ddsi_writer *wr), UNUSED_ARG(struct ddsi_proxy_reader *prd), UNUSED_ARG(bool relay_only), UNUSED_ARG(int64_t *crypto_handle))
{
  return true;
}

DDS_INLINE_EXPORT inline void q_omg_get_proxy_writer_security_info(UNUSED_ARG(struct ddsi_proxy_writer *pwr), UNUSED_ARG(const ddsi_plist_t *plist), UNUSED_ARG(nn_security_info_t *info))
{
}

DDS_INLINE_EXPORT inline bool q_omg_writer_is_discovery_protected(UNUSED_ARG(const struct ddsi_writer *wr))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_writer_is_submessage_protected(UNUSED_ARG(const struct ddsi_writer *wr))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_writer_is_payload_protected(UNUSED_ARG(const struct ddsi_writer *wr))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_security_check_remote_writer_permissions(UNUSED_ARG(const struct ddsi_proxy_writer *pwr), UNUSED_ARG(uint32_t domain_id), UNUSED_ARG(struct ddsi_participant *pp))
{
  return true;
}

DDS_INLINE_EXPORT inline void q_omg_security_deregister_remote_writer_match(UNUSED_ARG(const struct ddsi_proxy_writer *pwr), UNUSED_ARG(const struct ddsi_reader *rd), UNUSED_ARG(struct ddsi_rd_pwr_match *match))
{
}

DDS_INLINE_EXPORT inline void q_omg_get_proxy_reader_security_info(UNUSED_ARG(struct ddsi_proxy_reader *prd), UNUSED_ARG(const ddsi_plist_t *plist), UNUSED_ARG(nn_security_info_t *info))
{
}

DDS_INLINE_EXPORT inline bool q_omg_reader_is_discovery_protected(UNUSED_ARG(const struct ddsi_reader *rd))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_reader_is_submessage_protected(UNUSED_ARG(const struct ddsi_reader *rd))
{
  return false;
}


DDS_INLINE_EXPORT inline bool q_omg_security_check_remote_reader_permissions(UNUSED_ARG(const struct ddsi_proxy_reader *prd), UNUSED_ARG(uint32_t domain_id), UNUSED_ARG(struct ddsi_participant *pp), UNUSED_ARG(bool *relay_only))
{
  *relay_only = false;
  return true;
}

DDS_INLINE_EXPORT inline void set_proxy_participant_security_info(UNUSED_ARG(struct ddsi_proxy_participant *prd), UNUSED_ARG(const ddsi_plist_t *plist))
{
}

DDS_INLINE_EXPORT inline void set_proxy_reader_security_info(UNUSED_ARG(struct ddsi_proxy_reader *prd), UNUSED_ARG(const ddsi_plist_t *plist))
{
}

DDS_INLINE_EXPORT inline void set_proxy_writer_security_info(UNUSED_ARG(struct ddsi_proxy_writer *pwr), UNUSED_ARG(const ddsi_plist_t *plist))
{
}

DDS_INLINE_EXPORT inline bool
decode_Data(
  UNUSED_ARG(const struct ddsi_domaingv *gv),
  UNUSED_ARG(struct nn_rsample_info *sampleinfo),
  UNUSED_ARG(unsigned char *payloadp),
  UNUSED_ARG(uint32_t payloadsz),
  UNUSED_ARG(size_t *submsg_len))
{
  return true;
}

DDS_INLINE_EXPORT inline bool
decode_DataFrag(
  UNUSED_ARG(const struct ddsi_domaingv *gv),
  UNUSED_ARG(struct nn_rsample_info *sampleinfo),
  UNUSED_ARG(unsigned char *payloadp),
  UNUSED_ARG(uint32_t payloadsz),
  UNUSED_ARG(size_t *submsg_len))
{
  return true;
}

DDS_INLINE_EXPORT inline void
encode_datareader_submsg(
  UNUSED_ARG(struct nn_xmsg *msg),
  UNUSED_ARG(struct nn_xmsg_marker sm_marker),
  UNUSED_ARG(const struct ddsi_proxy_writer *pwr),
  UNUSED_ARG(const struct ddsi_guid *rd_guid))
{
}

DDS_INLINE_EXPORT inline void
encode_datawriter_submsg(
  UNUSED_ARG(struct nn_xmsg *msg),
  UNUSED_ARG(struct nn_xmsg_marker sm_marker),
  UNUSED_ARG(struct ddsi_writer *wr))
{
}

DDS_INLINE_EXPORT inline bool
validate_msg_decoding(
  UNUSED_ARG(const struct ddsi_entity_common *e),
  UNUSED_ARG(const struct ddsi_proxy_endpoint_common *c),
  UNUSED_ARG(struct ddsi_proxy_participant *proxypp),
  UNUSED_ARG(struct receiver_state *rst),
  UNUSED_ARG(SubmessageKind_t prev_smid))
{
  return true;
}

DDS_INLINE_EXPORT inline int
decode_SecPrefix(
  UNUSED_ARG(struct receiver_state *rst),
  UNUSED_ARG(unsigned char *submsg),
  UNUSED_ARG(size_t submsg_size),
  UNUSED_ARG(unsigned char * const msg_end),
  UNUSED_ARG(const ddsi_guid_prefix_t * const src_prefix),
  UNUSED_ARG(const ddsi_guid_prefix_t * const dst_prefix),
  UNUSED_ARG(int byteswap))
{
  /* Just let the parsing ignore the security sub-messages. */
  return true;
}

DDS_INLINE_EXPORT inline nn_rtps_msg_state_t
decode_rtps_message(
  UNUSED_ARG(struct thread_state * const thrst),
  UNUSED_ARG(struct ddsi_domaingv *gv),
  UNUSED_ARG(struct nn_rmsg **rmsg),
  UNUSED_ARG(Header_t **hdr),
  UNUSED_ARG(unsigned char **buff),
  UNUSED_ARG(size_t *sz),
  UNUSED_ARG(struct nn_rbufpool *rbpool),
  UNUSED_ARG(bool isstream))
{
  return NN_RTPS_MSG_STATE_PLAIN;
}

DDS_INLINE_EXPORT inline dds_return_t q_omg_security_load( UNUSED_ARG( struct dds_security_context *security_context ), UNUSED_ARG( const dds_qos_t *property_seq), UNUSED_ARG ( struct ddsi_domaingv *gv ) )
{
  return DDS_RETCODE_ERROR;
}

DDS_INLINE_EXPORT inline bool q_omg_is_security_loaded(  UNUSED_ARG( struct dds_security_context *sc )) { return false; }

DDS_INLINE_EXPORT inline void q_omg_security_deregister_remote_reader_match(UNUSED_ARG(const struct ddsi_proxy_reader *prd), UNUSED_ARG(const struct ddsi_writer *wr), UNUSED_ARG(struct ddsi_wr_prd_match *match))
{
}

DDS_INLINE_EXPORT inline bool q_omg_plist_keyhash_is_protected(UNUSED_ARG(const ddsi_plist_t *plist))
{
  return false;
}

DDS_INLINE_EXPORT inline bool q_omg_is_endpoint_protected(UNUSED_ARG(const ddsi_plist_t *plist))
{
  return false;
}

DDS_INLINE_EXPORT inline void q_omg_log_endpoint_protection(UNUSED_ARG(struct ddsi_domaingv * const gv), UNUSED_ARG(const ddsi_plist_t *plist))
{
}

#endif /* DDS_HAS_SECURITY */

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_OMG_SECURITY_H */
