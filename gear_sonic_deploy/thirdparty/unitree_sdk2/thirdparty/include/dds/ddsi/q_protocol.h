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
#ifndef NN_PROTOCOL_H
#define NN_PROTOCOL_H

#include "dds/ddsrt/endian.h"
#include "dds/ddsrt/misc.h"
#include "dds/ddsi/q_feature_check.h"

#include "dds/ddsi/q_rtps.h"
#include "dds/ddsi/ddsi_time.h"
#include "dds/ddsi/ddsi_locator.h"

#if defined (__cplusplus)
extern "C" {
#endif

typedef struct {
  uint8_t id[4];
} nn_protocolid_t;
typedef struct {
  int32_t high;
  uint32_t low;
} nn_sequence_number_t;
#define NN_SEQUENCE_NUMBER_UNKNOWN_HIGH -1
#define NN_SEQUENCE_NUMBER_UNKNOWN_LOW 0
#define NN_SEQUENCE_NUMBER_UNKNOWN ((seqno_t) (((uint64_t)NN_SEQUENCE_NUMBER_UNKNOWN_HIGH << 32) | NN_SEQUENCE_NUMBER_UNKNOWN_LOW))
/* C99 disallows flex array in nested struct, so only put the
   header in.  (GCC and Clang allow it, but there are other
   compilers in the world as well.) */
typedef struct nn_sequence_number_set_header {
  nn_sequence_number_t bitmap_base;
  uint32_t numbits;
} nn_sequence_number_set_header_t;
/* SequenceNumberSet size is base (2 words) + numbits (1 word) +
   bitmap ((numbits+31)/32 words), and this at 4 bytes/word */
#define NN_SEQUENCE_NUMBER_SET_MAX_BITS (256u)
#define NN_SEQUENCE_NUMBER_SET_BITS_SIZE(numbits) ((unsigned) (4 * (((numbits) + 31) / 32)))
#define NN_SEQUENCE_NUMBER_SET_SIZE(numbits) (sizeof (nn_sequence_number_set_header_t) + NN_SEQUENCE_NUMBER_SET_BITS_SIZE (numbits))
typedef uint32_t nn_fragment_number_t;
typedef struct nn_fragment_number_set_header {
  nn_fragment_number_t bitmap_base;
  uint32_t numbits;
} nn_fragment_number_set_header_t;
/* FragmentNumberSet size is base (2 words) + numbits (1 word) +
   bitmap ((numbits+31)/32 words), and this at 4 bytes/word */
#define NN_FRAGMENT_NUMBER_SET_MAX_BITS (256u)
#define NN_FRAGMENT_NUMBER_SET_BITS_SIZE(numbits) ((unsigned) (4 * (((numbits) + 31) / 32)))
#define NN_FRAGMENT_NUMBER_SET_SIZE(numbits) (sizeof (nn_fragment_number_set_header_t) + NN_FRAGMENT_NUMBER_SET_BITS_SIZE (numbits))
typedef uint32_t nn_count_t;

#define NN_STATUSINFO_DISPOSE      0x1u
#define NN_STATUSINFO_UNREGISTER   0x2u
#define NN_STATUSINFO_STANDARDIZED (NN_STATUSINFO_DISPOSE | NN_STATUSINFO_UNREGISTER)
#define NN_STATUSINFO_OSPL_AUTO    0x10000000u /* OSPL extension, not on the wire */
#define NN_STATUSINFOX_OSPL_AUTO   0x1         /* statusinfo word 2, OSPL L_AUTO flag on the wire */

#define NN_GUID_PREFIX_UNKNOWN_INITIALIZER {{0,0,0,0, 0,0,0,0, 0,0,0,0}}

#define NN_DISC_BUILTIN_ENDPOINT_PARTICIPANT_ANNOUNCER (1u << 0)
#define NN_DISC_BUILTIN_ENDPOINT_PARTICIPANT_DETECTOR (1u << 1)
#define NN_DISC_BUILTIN_ENDPOINT_PUBLICATION_ANNOUNCER (1u << 2)
#define NN_DISC_BUILTIN_ENDPOINT_PUBLICATION_DETECTOR (1u << 3)
#define NN_DISC_BUILTIN_ENDPOINT_SUBSCRIPTION_ANNOUNCER (1u << 4)
#define NN_DISC_BUILTIN_ENDPOINT_SUBSCRIPTION_DETECTOR (1u << 5)
#define NN_DISC_BUILTIN_ENDPOINT_PARTICIPANT_PROXY_ANNOUNCER (1u << 6) /* undefined meaning */
#define NN_DISC_BUILTIN_ENDPOINT_PARTICIPANT_PROXY_DETECTOR (1u << 7) /* undefined meaning */
#define NN_DISC_BUILTIN_ENDPOINT_PARTICIPANT_STATE_ANNOUNCER (1u << 8) /* undefined meaning */
#define NN_DISC_BUILTIN_ENDPOINT_PARTICIPANT_STATE_DETECTOR (1u << 9) /* undefined meaning */
#define NN_BUILTIN_ENDPOINT_PARTICIPANT_MESSAGE_DATA_WRITER (1u << 10)
#define NN_BUILTIN_ENDPOINT_PARTICIPANT_MESSAGE_DATA_READER (1u << 11)
#define NN_BUILTIN_ENDPOINT_TL_SVC_REQUEST_DATA_WRITER (1u << 12)
#define NN_BUILTIN_ENDPOINT_TL_SVC_REQUEST_DATA_READER (1u << 13)
#define NN_BUILTIN_ENDPOINT_TL_SVC_REPLY_DATA_WRITER (1u << 14)
#define NN_BUILTIN_ENDPOINT_TL_SVC_REPLY_DATA_READER (1u << 15)
#define NN_DISC_BUILTIN_ENDPOINT_TOPICS_ANNOUNCER (1u << 28)
#define NN_DISC_BUILTIN_ENDPOINT_TOPICS_DETECTOR (1u << 29)

/* Security extensions: */
#define NN_BUILTIN_ENDPOINT_PUBLICATION_MESSAGE_SECURE_ANNOUNCER (1u<<16)
#define NN_BUILTIN_ENDPOINT_PUBLICATION_MESSAGE_SECURE_DETECTOR (1u<<17)
#define NN_BUILTIN_ENDPOINT_SUBSCRIPTION_MESSAGE_SECURE_ANNOUNCER (1u<<18)
#define NN_BUILTIN_ENDPOINT_SUBSCRIPTION_MESSAGE_SECURE_DETECTOR (1u<<19)
#define NN_BUILTIN_ENDPOINT_PARTICIPANT_MESSAGE_SECURE_ANNOUNCER (1u<<20)
#define NN_BUILTIN_ENDPOINT_PARTICIPANT_MESSAGE_SECURE_DETECTOR (1u<<21)
#define NN_BUILTIN_ENDPOINT_PARTICIPANT_STATELESS_MESSAGE_ANNOUNCER (1u<<22)
#define NN_BUILTIN_ENDPOINT_PARTICIPANT_STATELESS_MESSAGE_DETECTOR (1u<<23)
#define NN_BUILTIN_ENDPOINT_PARTICIPANT_VOLATILE_SECURE_ANNOUNCER (1u<<24)
#define NN_BUILTIN_ENDPOINT_PARTICIPANT_VOLATILE_SECURE_DETECTOR (1u<<25)
#define NN_DISC_BUILTIN_ENDPOINT_PARTICIPANT_SECURE_ANNOUNCER (1u << 26)
#define NN_DISC_BUILTIN_ENDPOINT_PARTICIPANT_SECURE_DETECTOR (1u << 27)

#define NN_BES_MASK_NON_SECURITY 0xf000ffffu

#define NN_LOCATOR_KIND_INVALID -1
#define NN_LOCATOR_KIND_RESERVED 0
#define NN_LOCATOR_KIND_UDPv4 1
#define NN_LOCATOR_KIND_UDPv6 2
#define NN_LOCATOR_KIND_TCPv4 4
#define NN_LOCATOR_KIND_TCPv6 8
#define NN_LOCATOR_KIND_SHEM 16
#define NN_LOCATOR_KIND_RAWETH 0x8000 /* proposed vendor-specific */
#define NN_LOCATOR_KIND_UDPv4MCGEN 0x4fff0000
#define NN_LOCATOR_PORT_INVALID 0

/* Only one specific version is grokked */
#define RTPS_MAJOR 2
#define RTPS_MINOR 1
#define RTPS_MINOR_MINIMUM 1

typedef struct Header {
  nn_protocolid_t protocol;
  nn_protocol_version_t version;
  nn_vendorid_t vendorid;
  ddsi_guid_prefix_t guid_prefix;
} Header_t;
#if DDSRT_ENDIAN == DDSRT_LITTLE_ENDIAN
#define NN_PROTOCOLID_AS_UINT32 (((uint32_t)'R' << 0) | ((uint32_t)'T' << 8) | ((uint32_t)'P' << 16) | ((uint32_t)'S' << 24))
#elif DDSRT_ENDIAN == DDSRT_BIG_ENDIAN
#define NN_PROTOCOLID_AS_UINT32 (((uint32_t)'R' << 24) | ((uint32_t)'T' << 16) | ((uint32_t)'P' << 8) | ((uint32_t)'S' << 0))
#else
#error "DDSRT_ENDIAN neither LITTLE nor BIG"
#endif
#define RTPS_MESSAGE_HEADER_SIZE (sizeof (Header_t))

typedef struct SubmessageHeader {
  uint8_t submessageId;
  uint8_t flags;
  uint16_t octetsToNextHeader;
} SubmessageHeader_t;
#define RTPS_SUBMESSAGE_HEADER_SIZE (sizeof (SubmessageHeader_t))
#define SMFLAG_ENDIANNESS 0x01u

typedef enum SubmessageKind {
  SMID_PAD = 0x01,
  SMID_ACKNACK = 0x06,
  SMID_HEARTBEAT = 0x07,
  SMID_GAP = 0x08,
  SMID_INFO_TS = 0x09,
  SMID_INFO_SRC = 0x0c,
  SMID_INFO_REPLY_IP4 = 0x0d,
  SMID_INFO_DST = 0x0e,
  SMID_INFO_REPLY = 0x0f,
  SMID_NACK_FRAG = 0x12,
  SMID_HEARTBEAT_FRAG = 0x13,
  SMID_DATA = 0x15,
  SMID_DATA_FRAG = 0x16,
  /* security-specific sub messages */
  SMID_SEC_BODY = 0x30,
  SMID_SEC_PREFIX = 0x31,
  SMID_SEC_POSTFIX = 0x32,
  SMID_SRTPS_PREFIX = 0x33,
  SMID_SRTPS_POSTFIX = 0x34,
  /* vendor-specific sub messages (0x80 .. 0xff) */
  SMID_ADLINK_MSG_LEN = 0x81,
  SMID_ADLINK_ENTITY_ID = 0x82
} SubmessageKind_t;

typedef struct InfoTimestamp {
  SubmessageHeader_t smhdr;
  ddsi_time_t time;
} InfoTimestamp_t;

typedef struct EntityId {
  SubmessageHeader_t smhdr;
  ddsi_entityid_t entityid;
} EntityId_t;

typedef struct InfoDST {
  SubmessageHeader_t smhdr;
  ddsi_guid_prefix_t guid_prefix;
} InfoDST_t;

typedef struct InfoSRC {
  SubmessageHeader_t smhdr;
  unsigned unused;
  nn_protocol_version_t version;
  nn_vendorid_t vendorid;
  ddsi_guid_prefix_t guid_prefix;
} InfoSRC_t;

#if DDSRT_ENDIAN == DDSRT_LITTLE_ENDIAN
#define CDR_BE      0x0000u
#define CDR_LE      0x0100u
#define PL_CDR_BE   0x0200u
#define PL_CDR_LE   0x0300u
#define CDR2_BE     0x0600u
#define CDR2_LE     0x0700u
#define D_CDR2_BE   0x0800u
#define D_CDR2_LE   0x0900u
#define PL_CDR2_BE  0x0a00u
#define PL_CDR2_LE  0x0b00u

#define CDR_ENC_LE(x) (((x) & 0x0100) == 0x0100)
#define CDR_ENC_IS_NATIVE(x) (CDR_ENC_LE ((x)))
#define CDR_ENC_IS_VALID(x) (!((x) > PL_CDR2_LE || (x) == 0x0400 || (x) == 0x0500))
#define CDR_ENC_TO_NATIVE(x) ((x) | 0x0100)
#else
#define CDR_BE      0x0000u
#define CDR_LE      0x0001u
#define PL_CDR_BE   0x0002u
#define PL_CDR_LE   0x0003u
#define CDR2_BE     0x0006u
#define CDR2_LE     0x0007u
#define D_CDR2_BE   0x0008u
#define D_CDR2_LE   0x0009u
#define PL_CDR2_BE  0x000au
#define PL_CDR2_LE  0x000bu

#define CDR_ENC_LE(x) (((x) & 0x0001) == 0x0001)
#define CDR_ENC_IS_NATIVE(x) (!CDR_ENC_LE ((x)))
#define CDR_ENC_IS_VALID(x) (!((x) > PL_CDR2_LE || (x) == 0x0004 || (x) == 0x0005))
#define CDR_ENC_TO_NATIVE(x) ((x) & ~0x0001)
#endif

/*
  Encoding version to be used for serialization. Encoding version 1
  represents the XCDR1 format as defined in the DDS XTypes specification,
  with PLAIN_CDR(1) that is backwards compatible with the CDR encoding
  used by non-XTypes enabled nodes.
*/
#define CDR_ENC_VERSION_UNDEF     0
#define CDR_ENC_VERSION_1         1
#define CDR_ENC_VERSION_2         2

#define CDR_ENC_FORMAT_PLAIN      0
#define CDR_ENC_FORMAT_DELIMITED  1
#define CDR_ENC_FORMAT_PL         2


typedef uint16_t nn_parameterid_t; /* spec says short */
typedef struct nn_parameter {
  nn_parameterid_t parameterid;
  uint16_t length; /* spec says signed short */
  /* char value[] */
} nn_parameter_t;

typedef struct Data_DataFrag_common {
  SubmessageHeader_t smhdr;
  uint16_t extraFlags;
  uint16_t octetsToInlineQos;
  ddsi_entityid_t readerId;
  ddsi_entityid_t writerId;
  nn_sequence_number_t writerSN;
} Data_DataFrag_common_t;

typedef struct Data {
  Data_DataFrag_common_t x;
} Data_t;
#define DATA_FLAG_INLINE_QOS 0x02u
#define DATA_FLAG_DATAFLAG 0x04u
#define DATA_FLAG_KEYFLAG 0x08u

typedef struct DataFrag {
  Data_DataFrag_common_t x;
  nn_fragment_number_t fragmentStartingNum;
  uint16_t fragmentsInSubmessage;
  uint16_t fragmentSize;
  uint32_t sampleSize;
} DataFrag_t;
#define DATAFRAG_FLAG_INLINE_QOS 0x02u
#define DATAFRAG_FLAG_KEYFLAG 0x04u

typedef struct MsgLen {
  SubmessageHeader_t smhdr;
  uint32_t length;
} MsgLen_t;

DDSRT_WARNING_MSVC_OFF(4200)
typedef struct AckNack {
  SubmessageHeader_t smhdr;
  ddsi_entityid_t readerId;
  ddsi_entityid_t writerId;
  nn_sequence_number_set_header_t readerSNState;
  uint32_t bits[];
  /* nn_count_t count; */
} AckNack_t;
DDSRT_WARNING_MSVC_ON(4200)
#define ACKNACK_FLAG_FINAL 0x02u
#define ACKNACK_SIZE(numbits) (offsetof (AckNack_t, bits) + NN_SEQUENCE_NUMBER_SET_BITS_SIZE (numbits) + 4)
#define ACKNACK_SIZE_MAX ACKNACK_SIZE (NN_SEQUENCE_NUMBER_SET_MAX_BITS)

DDSRT_WARNING_MSVC_OFF(4200)
typedef struct Gap {
  SubmessageHeader_t smhdr;
  ddsi_entityid_t readerId;
  ddsi_entityid_t writerId;
  nn_sequence_number_t gapStart;
  nn_sequence_number_set_header_t gapList;
  uint32_t bits[];
} Gap_t;
DDSRT_WARNING_MSVC_ON(4200)
#define GAP_SIZE(numbits) (offsetof (Gap_t, bits) + NN_SEQUENCE_NUMBER_SET_BITS_SIZE (numbits))
#define GAP_SIZE_MAX GAP_SIZE (NN_SEQUENCE_NUMBER_SET_MAX_BITS)

typedef struct InfoTS {
  SubmessageHeader_t smhdr;
  ddsi_time_t time;
} InfoTS_t;
#define INFOTS_INVALIDATE_FLAG 0x2u

typedef struct Heartbeat {
  SubmessageHeader_t smhdr;
  ddsi_entityid_t readerId;
  ddsi_entityid_t writerId;
  nn_sequence_number_t firstSN;
  nn_sequence_number_t lastSN;
  nn_count_t count;
} Heartbeat_t;
#define HEARTBEAT_FLAG_FINAL 0x02u
#define HEARTBEAT_FLAG_LIVELINESS 0x04u

typedef struct HeartbeatFrag {
  SubmessageHeader_t smhdr;
  ddsi_entityid_t readerId;
  ddsi_entityid_t writerId;
  nn_sequence_number_t writerSN;
  nn_fragment_number_t lastFragmentNum;
  nn_count_t count;
} HeartbeatFrag_t;

DDSRT_WARNING_MSVC_OFF(4200)
typedef struct NackFrag {
  SubmessageHeader_t smhdr;
  ddsi_entityid_t readerId;
  ddsi_entityid_t writerId;
  nn_sequence_number_t writerSN;
  nn_fragment_number_set_header_t fragmentNumberState;
  uint32_t bits[];
  /* nn_count_t count; */
} NackFrag_t;
DDSRT_WARNING_MSVC_ON(4200)
#define NACKFRAG_SIZE(numbits) (offsetof (NackFrag_t, bits) + NN_FRAGMENT_NUMBER_SET_BITS_SIZE (numbits) + 4)
#define NACKFRAG_SIZE_MAX NACKFRAG_SIZE (NN_FRAGMENT_NUMBER_SET_MAX_BITS)

typedef union Submessage {
  SubmessageHeader_t smhdr;
  AckNack_t acknack;
  Data_t data;
  DataFrag_t datafrag;
  InfoTS_t infots;
  InfoDST_t infodst;
  InfoSRC_t infosrc;
  Heartbeat_t heartbeat;
  HeartbeatFrag_t heartbeatfrag;
  Gap_t gap;
  NackFrag_t nackfrag;
} Submessage_t;

#define PARTICIPANT_MESSAGE_DATA_KIND_UNKNOWN 0x0u
#define PARTICIPANT_MESSAGE_DATA_KIND_AUTOMATIC_LIVELINESS_UPDATE 0x1u
#define PARTICIPANT_MESSAGE_DATA_KIND_MANUAL_LIVELINESS_UPDATE 0x2u
#define PARTICIPANT_MESSAGE_DATA_VENDOR_SPECIFIC_KIND_FLAG 0x8000000u

#define PID_VENDORSPECIFIC_FLAG 0x8000u
#define PID_UNRECOGNIZED_INCOMPATIBLE_FLAG 0x4000u

#define PID_PAD                                 0x0u
#define PID_SENTINEL                            0x1u
#define PID_DOMAIN_ID                           0xfu
#define PID_DOMAIN_TAG                          (0x14u | PID_UNRECOGNIZED_INCOMPATIBLE_FLAG)
#define PID_USER_DATA                           0x2cu
#define PID_TOPIC_NAME                          0x5u
#define PID_TYPE_NAME                           0x7u
#define PID_GROUP_DATA                          0x2du
#define PID_TOPIC_DATA                          0x2eu
#define PID_DURABILITY                          0x1du
#define PID_DURABILITY_SERVICE                  0x1eu
#define PID_DEADLINE                            0x23u
#define PID_LATENCY_BUDGET                      0x27u
#define PID_LIVELINESS                          0x1bu
#define PID_RELIABILITY                         0x1au
#define PID_LIFESPAN                            0x2bu
#define PID_DESTINATION_ORDER                   0x25u
#define PID_HISTORY                             0x40u
#define PID_RESOURCE_LIMITS                     0x41u
#define PID_OWNERSHIP                           0x1fu
#define PID_OWNERSHIP_STRENGTH                  0x6u
#define PID_PRESENTATION                        0x21u
#define PID_PARTITION                           0x29u
#define PID_TIME_BASED_FILTER                   0x4u
#define PID_TRANSPORT_PRIORITY                  0x49u
#define PID_PROTOCOL_VERSION                    0x15u
#define PID_VENDORID                            0x16u
#define PID_UNICAST_LOCATOR                     0x2fu
#define PID_MULTICAST_LOCATOR                   0x30u
#define PID_MULTICAST_IPADDRESS                 0x11u
#define PID_DEFAULT_UNICAST_LOCATOR             0x31u
#define PID_DEFAULT_MULTICAST_LOCATOR           0x48u
#define PID_METATRAFFIC_UNICAST_LOCATOR         0x32u
#define PID_METATRAFFIC_MULTICAST_LOCATOR       0x33u
#define PID_DEFAULT_UNICAST_IPADDRESS           0xcu
#define PID_DEFAULT_UNICAST_PORT                0xeu
#define PID_METATRAFFIC_UNICAST_IPADDRESS       0x45u
#define PID_METATRAFFIC_UNICAST_PORT            0xdu
#define PID_METATRAFFIC_MULTICAST_IPADDRESS     0xbu
#define PID_METATRAFFIC_MULTICAST_PORT          0x46u
#define PID_EXPECTS_INLINE_QOS                  0x43u
#define PID_PARTICIPANT_MANUAL_LIVELINESS_COUNT 0x34u
#define PID_PARTICIPANT_BUILTIN_ENDPOINTS       0x44u
#define PID_PARTICIPANT_LEASE_DURATION          0x2u
#define PID_CONTENT_FILTER_PROPERTY             0x35u
#define PID_PARTICIPANT_GUID                    0x50u
#define PID_PARTICIPANT_ENTITYID                0x51u
#define PID_GROUP_GUID                          0x52u
#define PID_GROUP_ENTITYID                      0x53u
#define PID_BUILTIN_ENDPOINT_SET                0x58u
#define PID_PROPERTY_LIST                       0x59u
#define PID_TYPE_MAX_SIZE_SERIALIZED            0x60u
#define PID_ENTITY_NAME                         0x62u
#define PID_KEYHASH                             0x70u
#define PID_STATUSINFO                          0x71u
#define PID_CONTENT_FILTER_INFO                 0x55u
#define PID_COHERENT_SET                        0x56u
#define PID_DIRECTED_WRITE                      0x57u
#define PID_ORIGINAL_WRITER_INFO                0x61u
#define PID_ENDPOINT_GUID                       0x5au /* !SPEC <=> ADLINK_ENDPOINT_GUID */
#define PID_DATA_REPRESENTATION                 0x73u
#define PID_TYPE_CONSISTENCY_ENFORCEMENT        0x74u
#define PID_TYPE_INFORMATION                    0x75u

/* Security related PID values. */
#define PID_IDENTITY_TOKEN                      0x1001u
#define PID_PERMISSIONS_TOKEN                   0x1002u
#define PID_DATA_TAGS                           0x1003u
#define PID_ENDPOINT_SECURITY_INFO              0x1004u
#define PID_PARTICIPANT_SECURITY_INFO           0x1005u
#define PID_IDENTITY_STATUS_TOKEN               0x1006u

#ifdef DDS_HAS_SSM
/* To indicate whether a reader favours the use of SSM.  Iff the
   reader favours SSM, it will use SSM if available. */
#define PID_READER_FAVOURS_SSM                  0x72u
#endif

/* Deprecated parameter IDs (accepted but ignored) */
#define PID_PERSISTENCE                         0x03u
#define PID_TYPE_CHECKSUM                       0x08u
#define PID_TYPE2_NAME                          0x09u
#define PID_TYPE2_CHECKSUM                      0x0au
#define PID_EXPECTS_ACK                         0x10u
#define PID_MANAGER_KEY                         0x12u
#define PID_SEND_QUEUE_SIZE                     0x13u
#define PID_RELIABILITY_ENABLED                 0x14u
#define PID_VARGAPPS_SEQUENCE_NUMBER_LAST       0x17u
#define PID_RECV_QUEUE_SIZE                     0x18u
#define PID_RELIABILITY_OFFERED                 0x19u

#define PID_ADLINK_BUILTIN_ENDPOINT_SET         (PID_VENDORSPECIFIC_FLAG | 0x0u)

/* parameter ids for READER_DATA_LIFECYCLE & WRITER_DATA_LIFECYCLE are
   undefined, but let's publish them anyway */
#define PID_ADLINK_READER_DATA_LIFECYCLE        (PID_VENDORSPECIFIC_FLAG | 0x2u)
#define PID_ADLINK_WRITER_DATA_LIFECYCLE        (PID_VENDORSPECIFIC_FLAG | 0x3u)

/* ENDPOINT_GUID is formally undefined, so in strictly conforming
   mode, we use our own */
#define PID_ADLINK_ENDPOINT_GUID                (PID_VENDORSPECIFIC_FLAG | 0x4u)

#define PID_ADLINK_SYNCHRONOUS_ENDPOINT         (PID_VENDORSPECIFIC_FLAG | 0x5u)

/* Relaxed QoS matching readers/writers are best ignored by
   implementations that don't understand them.  This also covers "old"
   DDSI2's, although they may emit an error. */
#define PID_ADLINK_RELAXED_QOS_MATCHING         (PID_VENDORSPECIFIC_FLAG | PID_UNRECOGNIZED_INCOMPATIBLE_FLAG | 0x6u)

#define PID_ADLINK_PARTICIPANT_VERSION_INFO     (PID_VENDORSPECIFIC_FLAG | 0x7u)

#define PID_ADLINK_NODE_NAME                    (PID_VENDORSPECIFIC_FLAG | 0x8u)
#define PID_ADLINK_EXEC_NAME                    (PID_VENDORSPECIFIC_FLAG | 0x9u)
#define PID_ADLINK_PROCESS_ID                   (PID_VENDORSPECIFIC_FLAG | 0xau)
#define PID_ADLINK_SERVICE_TYPE                 (PID_VENDORSPECIFIC_FLAG | 0xbu)
#define PID_ADLINK_ENTITY_FACTORY               (PID_VENDORSPECIFIC_FLAG | 0xcu)
#define PID_ADLINK_WATCHDOG_SCHEDULING          (PID_VENDORSPECIFIC_FLAG | 0xdu)
#define PID_ADLINK_LISTENER_SCHEDULING          (PID_VENDORSPECIFIC_FLAG | 0xeu)
#define PID_ADLINK_SUBSCRIPTION_KEYS            (PID_VENDORSPECIFIC_FLAG | 0xfu)
#define PID_ADLINK_READER_LIFESPAN              (PID_VENDORSPECIFIC_FLAG | 0x10u)
#define PID_ADLINK_TYPE_DESCRIPTION             (PID_VENDORSPECIFIC_FLAG | 0x12u)
#define PID_ADLINK_LAN                          (PID_VENDORSPECIFIC_FLAG | 0x13u)
#define PID_ADLINK_ENDPOINT_GID                 (PID_VENDORSPECIFIC_FLAG | 0x14u)
#define PID_ADLINK_GROUP_GID                    (PID_VENDORSPECIFIC_FLAG | 0x15u)
#define PID_ADLINK_EOTINFO                      (PID_VENDORSPECIFIC_FLAG | 0x16u)
#define PID_ADLINK_PART_CERT_NAME               (PID_VENDORSPECIFIC_FLAG | 0x17u)
#define PID_ADLINK_LAN_CERT_NAME                (PID_VENDORSPECIFIC_FLAG | 0x18u)
#define PID_CYCLONE_RECEIVE_BUFFER_SIZE         (PID_VENDORSPECIFIC_FLAG | 0x19u)
#define PID_CYCLONE_TYPE_INFORMATION            (PID_VENDORSPECIFIC_FLAG | 0x1au)
#define PID_CYCLONE_TOPIC_GUID                  (PID_VENDORSPECIFIC_FLAG | 0x1bu)
#define PID_CYCLONE_REQUESTS_KEYHASH            (PID_VENDORSPECIFIC_FLAG | 0x1cu)
#define PID_CYCLONE_REDUNDANT_NETWORKING        (PID_VENDORSPECIFIC_FLAG | 0x1du)

/* Names of the built-in topics */
#define DDS_BUILTIN_TOPIC_PARTICIPANT_NAME "DCPSParticipant"
#define DDS_BUILTIN_TOPIC_PUBLICATION_NAME "DCPSPublication"
#define DDS_BUILTIN_TOPIC_SUBSCRIPTION_NAME "DCPSSubscription"
#define DDS_BUILTIN_TOPIC_TOPIC_NAME "DCPSTopic"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_MESSAGE_NAME "DCPSParticipantMessage"
#define DDS_BUILTIN_TOPIC_TYPELOOKUP_REQUEST_NAME "DCPSTypeLookupRequest"
#define DDS_BUILTIN_TOPIC_TYPELOOKUP_REPLY_NAME "DCPSTypeLookupReply"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_SECURE_NAME "DCPSParticipantsSecure"
#define DDS_BUILTIN_TOPIC_PUBLICATION_SECURE_NAME "DCPSPublicationsSecure"
#define DDS_BUILTIN_TOPIC_SUBSCRIPTION_SECURE_NAME "DCPSSubscriptionsSecure"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_MESSAGE_SECURE_NAME "DCPSParticipantMessageSecure"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_STATELESS_MESSAGE_NAME "DCPSParticipantStatelessMessage"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_VOLATILE_MESSAGE_SECURE_NAME "DCPSParticipantVolatileMessageSecure"

/* Participant built-in topic qos properties */
#define DDS_BUILTIN_TOPIC_PARTICIPANT_PROPERTY_PROCESS_NAME "__ProcessName"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_PROPERTY_PID "__Pid"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_PROPERTY_HOSTNAME "__Hostname"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_PROPERTY_NETWORKADDRESSES "__NetworkAddresses"
#define DDS_BUILTIN_TOPIC_PARTICIPANT_DEBUG_MONITOR "__DebugMonitor"
#define DDS_BUILTIN_TOPIC_NULL_NAME NULL

#if defined (__cplusplus)
}
#endif

#endif /* NN_PROTOCOL_H */
