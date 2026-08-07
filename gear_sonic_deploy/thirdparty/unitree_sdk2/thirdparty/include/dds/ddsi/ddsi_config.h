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
#ifndef DDSI_CONFIG_H
#define DDSI_CONFIG_H

#include <stdio.h>

#include "dds/export.h"
#include "dds/features.h"
#include "dds/ddsrt/sched.h"
#include "dds/ddsrt/random.h"
#include "dds/ddsi/ddsi_portmapping.h"
#include "dds/ddsi/ddsi_locator.h"

#if defined (__cplusplus)
extern "C" {
#endif

struct ddsi_config;

/**
 * @brief Default-initialize a configuration (unstable)
 *
 * @param[out]  cfg The configuration struct to be initialized.
 */
DDS_EXPORT void ddsi_config_init_default (struct ddsi_config *cfg);

enum ddsi_standards_conformance {
  DDSI_SC_PEDANTIC,
  DDSI_SC_STRICT,
  DDSI_SC_LAX
};

#define DDSI_SC_PEDANTIC_P(config) ((config).standards_conformance <= DDSI_SC_PEDANTIC)
#define DDSI_SC_STRICT_P(config) ((config).standards_conformance <= DDSI_SC_STRICT)

enum ddsi_besmode {
  DDSI_BESMODE_FULL,
  DDSI_BESMODE_WRITERS,
  DDSI_BESMODE_MINIMAL
};

enum ddsi_retransmit_merging {
  DDSI_REXMIT_MERGE_NEVER,
  DDSI_REXMIT_MERGE_ADAPTIVE,
  DDSI_REXMIT_MERGE_ALWAYS
};

enum ddsi_boolean_default {
  DDSI_BOOLDEF_DEFAULT,
  DDSI_BOOLDEF_FALSE,
  DDSI_BOOLDEF_TRUE
};

#ifdef DDS_HAS_SHM
enum ddsi_shm_loglevel {
  DDSI_SHM_OFF = 0,
  DDSI_SHM_FATAL,
  DDSI_SHM_ERROR,
  DDSI_SHM_WARN,
  DDSI_SHM_INFO,
  DDSI_SHM_DEBUG,
  DDSI_SHM_VERBOSE
};
#endif

#define DDSI_PARTICIPANT_INDEX_AUTO -1
#define DDSI_PARTICIPANT_INDEX_NONE -2

/* ddsi_config_listelem must be an overlay for all used listelem types */
struct ddsi_config_listelem {
  struct ddsi_config_listelem *next;
};

#ifdef DDS_HAS_NETWORK_PARTITIONS
struct networkpartition_address {
  struct networkpartition_address *next;
  ddsi_locator_t loc;
};

struct ddsi_config_networkpartition_listelem {
  struct ddsi_config_networkpartition_listelem *next;
  char *name;
  char *address_string;
  struct networkpartition_address *uc_addresses;
  struct networkpartition_address *asm_addresses;
#ifdef DDS_HAS_SSM
  struct networkpartition_address *ssm_addresses;
#endif
};

struct ddsi_config_ignoredpartition_listelem {
  struct ddsi_config_ignoredpartition_listelem *next;
  char *DCPSPartitionTopic;
};

struct ddsi_config_partitionmapping_listelem {
  struct ddsi_config_partitionmapping_listelem *next;
  char *networkPartition;
  char *DCPSPartitionTopic;
  struct ddsi_config_networkpartition_listelem *partition;
};
#endif /* DDS_HAS_NETWORK_PARTITIONS */

#ifdef DDS_HAS_NETWORK_CHANNELS
struct ddsi_config_channel_listelem {
  struct ddsi_config_channel_listelem *next;
  char   *name;
  int    priority;
  int64_t resolution;
#ifdef DDS_HAS_BANDWIDTH_LIMITING
  uint32_t data_bandwidth_limit;
  uint32_t auxiliary_bandwidth_limit;
#endif
  int    diffserv_field;
  struct thread_state *channel_reader_thrst;  /* keeping an handle to the running thread for this channel */
  struct nn_dqueue *dqueue; /* The handle of teh delivery queue servicing incoming data for this channel*/
  struct xeventq *evq; /* The handle of the event queue servicing this channel*/
  uint32_t queueId; /* the index of the networkqueue serviced by this channel*/
  struct ddsi_tran_conn * transmit_conn; /* the connection used for sending data out via this channel */
};
#endif /* DDS_HAS_NETWORK_CHANNELS */

struct ddsi_config_maybe_int32 {
  int isdefault;
  int32_t value;
};

struct ddsi_config_maybe_uint32 {
  int isdefault;
  uint32_t value;
};

struct ddsi_config_thread_properties_listelem {
  struct ddsi_config_thread_properties_listelem *next;
  char *name;
  ddsrt_sched_t sched_class;
  struct ddsi_config_maybe_int32 schedule_priority;
  struct ddsi_config_maybe_uint32 stack_size;
};

struct ddsi_config_peer_listelem
{
  struct ddsi_config_peer_listelem *next;
  char *peer;
};

struct ddsi_config_prune_deleted_ppant {
  int64_t delay;
  int enforce_delay;
};

/* allow multicast bits (default depends on network type): */
#define DDSI_AMC_FALSE 0u
#define DDSI_AMC_SPDP 1u
#define DDSI_AMC_ASM 2u
#ifdef DDS_HAS_SSM
#define DDSI_AMC_SSM 4u
#define DDSI_AMC_TRUE (DDSI_AMC_SPDP | DDSI_AMC_ASM | DDSI_AMC_SSM)
#else
#define DDSI_AMC_TRUE (DDSI_AMC_SPDP | DDSI_AMC_ASM)
#endif
#define DDSI_AMC_DEFAULT 0x80000000u

/* FIXME: this should be fully dynamic ... but this is easier for a quick hack */
enum ddsi_transport_selector {
  DDSI_TRANS_DEFAULT, /* actually UDP, but this is so we can tell what has been set */
  DDSI_TRANS_UDP,
  DDSI_TRANS_UDP6,
  DDSI_TRANS_TCP,
  DDSI_TRANS_TCP6,
  DDSI_TRANS_RAWETH,
  DDSI_TRANS_NONE /* FIXME: see FIXME above ... :( */
};

enum ddsi_many_sockets_mode {
  DDSI_MSM_NO_UNICAST,
  DDSI_MSM_SINGLE_UNICAST,
  DDSI_MSM_MANY_UNICAST
};

#ifdef DDS_HAS_SECURITY
struct ddsi_plugin_library_properties {
  char *library_path;
  char *library_init;
  char *library_finalize;
};

struct ddsi_authentication_properties {
  char *identity_certificate;
  char *identity_ca;
  char *private_key;
  char *password;
  char *trusted_ca_dir;
  char *crl;
  int include_optional_fields;
};

struct ddsi_access_control_properties {
  char *permissions;
  char *permissions_ca;
  char *governance;
};

struct ddsi_config_omg_security {
  struct ddsi_authentication_properties authentication_properties;
  struct ddsi_access_control_properties access_control_properties;
  struct ddsi_plugin_library_properties authentication_plugin;
  struct ddsi_plugin_library_properties access_control_plugin;
  struct ddsi_plugin_library_properties cryptography_plugin;
};

struct ddsi_config_omg_security_listelem {
  struct ddsi_config_omg_security_listelem *next;
  struct ddsi_config_omg_security cfg;
};
#endif /* DDS_HAS_SECURITY */

#ifdef DDS_HAS_SSL
struct ddsi_config_ssl_min_version {
  int major;
  int minor;
};
#endif

struct ddsi_config_socket_buf_size {
  struct ddsi_config_maybe_uint32 min, max;
};

struct ddsi_config_network_interface {
  int automatic;
  char *name;
  char *address;
  int prefer_multicast;
  int presence_required;
  enum ddsi_boolean_default multicast;
  struct ddsi_config_maybe_int32 priority;
};

struct ddsi_config_network_interface_listelem {
  struct ddsi_config_network_interface_listelem *next;
  struct ddsi_config_network_interface cfg;
};

enum ddsi_config_entity_naming_mode {
  DDSI_ENTITY_NAMING_DEFAULT_EMPTY,
  DDSI_ENTITY_NAMING_DEFAULT_FANCY
};

/* Expensive checks (compiled in when NDEBUG not defined, enabled only if flag set in xchecks) */
#define DDSI_XCHECK_WHC 1u
#define DDSI_XCHECK_RHC 2u
#define DDSI_XCHECK_XEV 4u

struct ddsi_config
{
  int valid;
  uint32_t tracemask;
  uint32_t enabled_xchecks;
  char *pcap_file;

  /* interfaces */
  struct ddsi_config_network_interface_listelem *network_interfaces;

  /* deprecated interface support */
  char *depr_networkAddressString;
  int depr_prefer_multicast;
  char *depr_assumeMulticastCapable;

  char **networkRecvAddressStrings;
  uint32_t allowMulticast;
  char *externalAddressString;
  char *externalMaskString;
  FILE *tracefp;
  char *tracefile;
  int tracingAppendToFile;
  enum ddsi_transport_selector transport_selector;
  enum ddsi_boolean_default compat_use_ipv6;
  enum ddsi_boolean_default compat_tcp_enable;
  int dontRoute;
  int enableMulticastLoopback;
  uint32_t domainId;
  struct ddsi_config_maybe_uint32 extDomainId; // domain id advertised in discovery
  char *domainTag;
  int participantIndex;
  int maxAutoParticipantIndex;
  char *spdpMulticastAddressString;
  char *defaultMulticastAddressString;
  int64_t spdp_interval;
  int64_t spdp_response_delay_max;
  int64_t lease_duration;
  int64_t const_hb_intv_sched;
  int64_t const_hb_intv_sched_min;
  int64_t const_hb_intv_sched_max;
  int64_t const_hb_intv_min;
  enum ddsi_retransmit_merging retransmit_merging;
  int64_t retransmit_merging_period;
  int squash_participants;
  int liveliness_monitoring;
  int noprogress_log_stacktraces;
  int64_t liveliness_monitoring_interval;
  int prioritize_retransmit;
  enum ddsi_boolean_default multiple_recv_threads;
  unsigned recv_thread_stop_maxretries;

  unsigned primary_reorder_maxsamples;
  unsigned secondary_reorder_maxsamples;

  unsigned delivery_queue_maxsamples;

  uint16_t fragment_size;
  uint32_t max_msg_size;
  uint32_t max_rexmit_msg_size;
  uint32_t init_transmit_extra_pct;
  uint32_t max_rexmit_burst_size;

  int publish_uc_locators; /* Publish discovery unicast locators */
  int enable_uc_locators; /* If false, don't even try to create a unicast socket */

#ifdef DDS_HAS_TOPIC_DISCOVERY
  int enable_topic_discovery_endpoints;
#endif

  /* TCP transport configuration */
  int tcp_nodelay;
  int tcp_port;
  int64_t tcp_read_timeout;
  int64_t tcp_write_timeout;
  int tcp_use_peeraddr_for_unicast;

#ifdef DDS_HAS_SSL
  /* SSL support for TCP */
  int ssl_enable;
  int ssl_verify;
  int ssl_verify_client;
  int ssl_self_signed;
  char * ssl_keystore;
  char * ssl_rand_file;
  char * ssl_key_pass;
  char * ssl_ciphers;
  struct ddsi_config_ssl_min_version ssl_min_version;
#endif

#ifdef DDS_HAS_NETWORK_CHANNELS
  struct ddsi_config_channel_listelem *channels;
  struct ddsi_config_channel_listelem *max_channel; /* channel with highest prio; always computed */
#endif /* DDS_HAS_NETWORK_CHANNELS */
#ifdef DDS_HAS_NETWORK_PARTITIONS
  struct ddsi_config_networkpartition_listelem *networkPartitions;
  unsigned nof_networkPartitions;
  struct ddsi_config_ignoredpartition_listelem *ignoredPartitions;
  struct ddsi_config_partitionmapping_listelem *partitionMappings;
#endif /* DDS_HAS_NETWORK_PARTITIONS */
  struct ddsi_config_peer_listelem *peers;
  struct ddsi_config_thread_properties_listelem *thread_properties;

  /* debug/test/undoc features: */
  int xmit_lossiness;           /**<< fraction of packets to drop on xmit, in units of 1e-3 */
  uint32_t rmsg_chunk_size;          /**<< size of a chunk in the receive buffer */
  uint32_t rbuf_size;                /* << size of a single receiver buffer */
  enum ddsi_besmode besmode;
  int meas_hb_to_ack_latency;
  int unicast_response_to_spdp_messages;
  int synchronous_delivery_priority_threshold;
  int64_t synchronous_delivery_latency_bound;

  /* Write cache */

  int whc_batch;
  uint32_t whc_lowwater_mark;
  uint32_t whc_highwater_mark;
  struct ddsi_config_maybe_uint32 whc_init_highwater_mark;
  int whc_adaptive;

  unsigned defrag_unreliable_maxsamples;
  unsigned defrag_reliable_maxsamples;
  unsigned accelerate_rexmit_block_size;
  int64_t responsiveness_timeout;
  uint32_t max_participants;
  int64_t writer_linger_duration;
  int multicast_ttl;
  struct ddsi_config_socket_buf_size socket_rcvbuf_size;
  struct ddsi_config_socket_buf_size socket_sndbuf_size;
  int64_t ack_delay;
  int64_t nack_delay;
  int64_t preemptive_ack_delay;
  int64_t schedule_time_rounding;
  int64_t auto_resched_nack_delay;
  int64_t ds_grace_period;
#ifdef DDS_HAS_BANDWIDTH_LIMITING
  uint32_t auxiliary_bandwidth_limit; /* bytes/second */
#endif
  uint32_t max_queued_rexmit_bytes;
  unsigned max_queued_rexmit_msgs;
  unsigned ddsi2direct_max_threads;
  int late_ack_mode;
  int retry_on_reject_besteffort;
  int generate_keyhash;
  uint32_t max_sample_size;

  /* compability options */
  enum ddsi_standards_conformance standards_conformance;
  int explicitly_publish_qos_set_to_default;
  enum ddsi_many_sockets_mode many_sockets_mode;
  int assume_rti_has_pmd_endpoints;

  struct ddsi_portmapping ports;

  int monitor_port;

  int enable_control_topic;
  int initial_deaf;
  int initial_mute;
  int64_t initial_deaf_mute_reset;

  int use_multicast_if_mreqn;
  struct ddsi_config_prune_deleted_ppant prune_deleted_ppant;
  int redundant_networking;

#ifdef DDS_HAS_SECURITY
  struct ddsi_config_omg_security_listelem *omg_security_configuration;
#endif

#ifdef DDS_HAS_SHM
  int enable_shm;
  char *shm_locator;
  char *iceoryx_service;
  enum ddsi_shm_loglevel shm_log_lvl;
#endif

  enum ddsi_config_entity_naming_mode entity_naming_mode;
  ddsrt_prng_seed_t entity_naming_seed;

#if defined (__cplusplus)
public:
  ddsi_config() {
    ddsi_config_init_default (this);
  }
#endif
};

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_CONFIG_H */
