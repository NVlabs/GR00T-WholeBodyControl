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
#ifndef DDSI_PLIST_H
#define DDSI_PLIST_H

#include "dds/ddsrt/bswap.h"
#include "dds/ddsi/q_feature_check.h"
#include "dds/ddsi/ddsi_xqos.h"
#include "dds/ddsi/ddsi_keyhash.h"
#include "dds/ddsi/ddsi_tran.h" /* FIXME: eliminate */
#include "dds/ddsi/ddsi_typelookup.h"

#if defined (__cplusplus)
extern "C" {
#endif

#define PP_PROTOCOL_VERSION                     ((uint64_t)1 <<  0)
#define PP_VENDORID                             ((uint64_t)1 <<  1)
#define PP_UNICAST_LOCATOR                      ((uint64_t)1 <<  2)
#define PP_MULTICAST_LOCATOR                    ((uint64_t)1 <<  3)
#define PP_DEFAULT_UNICAST_LOCATOR              ((uint64_t)1 <<  4)
#define PP_DEFAULT_MULTICAST_LOCATOR            ((uint64_t)1 <<  5)
#define PP_METATRAFFIC_UNICAST_LOCATOR          ((uint64_t)1 <<  6)
#define PP_METATRAFFIC_MULTICAST_LOCATOR        ((uint64_t)1 <<  7)
#define PP_EXPECTS_INLINE_QOS                   ((uint64_t)1 <<  8)
#define PP_PARTICIPANT_MANUAL_LIVELINESS_COUNT  ((uint64_t)1 <<  9)
#define PP_PARTICIPANT_BUILTIN_ENDPOINTS        ((uint64_t)1 << 10)
#define PP_PARTICIPANT_LEASE_DURATION           ((uint64_t)1 << 11)
#define PP_CONTENT_FILTER_PROPERTY              ((uint64_t)1 << 12)
#define PP_PARTICIPANT_GUID                     ((uint64_t)1 << 13)
#define PP_PARTICIPANT_ENTITYID                 ((uint64_t)1 << 14)
#define PP_GROUP_GUID                           ((uint64_t)1 << 15)
#define PP_GROUP_ENTITYID                       ((uint64_t)1 << 16)
#define PP_BUILTIN_ENDPOINT_SET                 ((uint64_t)1 << 17)
#define PP_PROPERTIES                           ((uint64_t)1 << 18)
#define PP_TYPE_MAX_SIZE_SERIALIZED             ((uint64_t)1 << 19)
#define PP_CYCLONE_REDUNDANT_NETWORKING         ((uint64_t)1 << 20)
#define PP_KEYHASH                              ((uint64_t)1 << 21)
#define PP_STATUSINFO                           ((uint64_t)1 << 22)
#define PP_ORIGINAL_WRITER_INFO                 ((uint64_t)1 << 23)
#define PP_ENDPOINT_GUID                        ((uint64_t)1 << 24)
#define PP_ADLINK_PARTICIPANT_VERSION_INFO      ((uint64_t)1 << 26)
#define PP_ADLINK_TYPE_DESCRIPTION              ((uint64_t)1 << 27)
#define PP_COHERENT_SET                         ((uint64_t)1 << 28)
#ifdef DDS_HAS_SSM
#define PP_READER_FAVOURS_SSM                   ((uint64_t)1 << 29)
#endif
#define PP_DOMAIN_ID                            ((uint64_t)1 << 30)
#define PP_DOMAIN_TAG                           ((uint64_t)1 << 31)
/* Security extensions. */
#define PP_IDENTITY_TOKEN                       ((uint64_t)1 << 32)
#define PP_PERMISSIONS_TOKEN                    ((uint64_t)1 << 33)
#define PP_ENDPOINT_SECURITY_INFO               ((uint64_t)1 << 34)
#define PP_PARTICIPANT_SECURITY_INFO            ((uint64_t)1 << 35)
#define PP_IDENTITY_STATUS_TOKEN                ((uint64_t)1 << 36)
#define PP_DATA_TAGS                            ((uint64_t)1 << 37)
/* Other stuff */
#define PP_CYCLONE_RECEIVE_BUFFER_SIZE          ((uint64_t)1 << 38)
#define PP_CYCLONE_TOPIC_GUID                   ((uint64_t)1 << 39)
#define PP_CYCLONE_REQUESTS_KEYHASH             ((uint64_t)1 << 40)

/* Set for unrecognized parameters that are in the reserved space or
   in our own vendor-specific space that have the
   PID_UNRECOGNIZED_INCOMPATIBLE_FLAG set (see DDSI 2.1 9.6.2.2.1) */
#define PP_INCOMPATIBLE                         ((uint64_t)1 << 63)

#define NN_ADLINK_PARTICIPANT_VERSION_INFO_FIXED_CDRSIZE (24)

#define NN_ADLINK_FL_KERNEL_SEQUENCE_NUMBER     (1u << 0)
#define NN_ADLINK_FL_DISCOVERY_INCLUDES_GID     (1u << 1)
#define NN_ADLINK_FL_PTBES_FIXED_0              (1u << 2)
#define NN_ADLINK_FL_DDSI2_PARTICIPANT_FLAG     (1u << 3)
#define NN_ADLINK_FL_PARTICIPANT_IS_DDSI2       (1u << 4)
#define NN_ADLINK_FL_MINIMAL_BES_MODE           (1u << 5)
#define NN_ADLINK_FL_SUPPORTS_STATUSINFOX       (1u << 5)
/* SUPPORTS_STATUSINFOX: when set, also means any combination of
   write/unregister/dispose supported */

/* For locators one could patch the received message data to create
   singly-linked lists (parameter header -> offset of next entry in
   list relative to current), allowing aliasing of the data. But that
   requires modifying the data. For string sequences the length does
   the same thing. */
struct nn_locators_one {
  struct nn_locators_one *next;
  ddsi_locator_t loc;
};

typedef struct nn_locators {
  uint32_t n;
  struct nn_locators_one *first;
  struct nn_locators_one *last;
} nn_locators_t;

typedef uint32_t nn_ipv4address_t;

typedef uint32_t nn_port_t;

#ifdef DDS_HAS_SECURITY
typedef struct nn_tag {
  char *name;
  char *value;
} nn_tag_t;

typedef struct nn_tagseq {
  uint32_t n;
  nn_tag_t *tags;
} nn_tagseq_t;

typedef struct nn_datatags {
  nn_tagseq_t tags;
} nn_datatags_t;
#endif

#ifdef DDS_HAS_SSM
typedef struct nn_reader_favours_ssm {
  uint32_t state; /* default is false */
} nn_reader_favours_ssm_t;
#endif

#ifdef DDS_HAS_SECURITY
typedef struct nn_dataholder
{
  char *class_id;
  dds_propertyseq_t properties;
  dds_binarypropertyseq_t binary_properties;
} nn_dataholder_t;

typedef struct nn_dataholderseq {
  uint32_t n;
  nn_dataholder_t *tags;
} nn_dataholderseq_t;

typedef nn_dataholder_t nn_token_t;

/* Used for both nn_participant_security_info and nn_endpoint_security_info. */
typedef struct nn_security_info
{
  uint32_t security_attributes;
  uint32_t plugin_security_attributes;
} nn_security_info_t;

#define NN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_READ_PROTECTED                         (1u <<  0)
#define NN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_WRITE_PROTECTED                        (1u <<  1)
#define NN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_DISCOVERY_PROTECTED                    (1u <<  2)
#define NN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_SUBMESSAGE_PROTECTED                   (1u <<  3)
#define NN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_PAYLOAD_PROTECTED                      (1u <<  4)
#define NN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_KEY_PROTECTED                          (1u <<  5)
#define NN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_LIVELINESS_PROTECTED                   (1u <<  6)
#define NN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_VALID                                  (1u << 31)

#define NN_PLUGIN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_SUBMESSAGE_ENCRYPTED            (1u <<  0)
#define NN_PLUGIN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_PAYLOAD_ENCRYPTED               (1u <<  1)
#define NN_PLUGIN_ENDPOINT_SECURITY_ATTRIBUTES_FLAG_IS_SUBMESSAGE_ORIGIN_AUTHENTICATED (1u <<  2)

#define NN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_RTPS_PROTECTED                      (1u <<  0)
#define NN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_DISCOVERY_PROTECTED                 (1u <<  1)
#define NN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_LIVELINESS_PROTECTED                (1u <<  2)
#define NN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_VALID                               (1u << 31)

#define NN_PLUGIN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_RTPS_ENCRYPTED               (1u <<  0)
#define NN_PLUGIN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_DISCOVERY_ENCRYPTED          (1u <<  1)
#define NN_PLUGIN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_LIVELINESS_ENCRYPTED         (1u <<  2)
#define NN_PLUGIN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_RTPS_AUTHENTICATED           (1u <<  3)
#define NN_PLUGIN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_DISCOVERY_AUTHENTICATED      (1u <<  4)
#define NN_PLUGIN_PARTICIPANT_SECURITY_ATTRIBUTES_FLAG_IS_LIVELINESS_AUTHENTICATED     (1u <<  5)
#else
struct nn_security_info;
typedef struct nn_security_info nn_security_info_t;
#endif

typedef struct nn_adlink_participant_version_info
{
  uint32_t version;
  uint32_t flags;
  uint32_t unused[3];
  char *internals;
} nn_adlink_participant_version_info_t;

typedef struct ddsi_plist {
  uint64_t present;
  uint64_t aliased;

  dds_qos_t qos;

  nn_protocol_version_t protocol_version;
  nn_vendorid_t vendorid;
  nn_locators_t unicast_locators;
  nn_locators_t multicast_locators;
  nn_locators_t default_unicast_locators;
  nn_locators_t default_multicast_locators;
  nn_locators_t metatraffic_unicast_locators;
  nn_locators_t metatraffic_multicast_locators;

  unsigned char expects_inline_qos;
  nn_count_t participant_manual_liveliness_count;
  uint32_t participant_builtin_endpoints;
  dds_duration_t participant_lease_duration;
  /* nn_content_filter_property_t content_filter_property; */
  ddsi_guid_t participant_guid;
  ddsi_guid_t endpoint_guid;
  ddsi_guid_t group_guid;
  ddsi_guid_t topic_guid;
#if 0 /* reserved, rather than NIY */
  nn_entityid_t participant_entityid;
  nn_entityid_t group_entityid;
#endif
  uint32_t builtin_endpoint_set;
  /* int type_max_size_serialized; */
  ddsi_keyhash_t keyhash;
  uint32_t statusinfo;
  nn_adlink_participant_version_info_t adlink_participant_version_info;
  char *type_description;
  nn_sequence_number_t coherent_set_seqno;
#ifdef DDS_HAS_SECURITY
  nn_token_t identity_token;
  nn_token_t permissions_token;
  nn_security_info_t endpoint_security_info;
  nn_security_info_t participant_security_info;
  nn_token_t identity_status_token;
  nn_datatags_t data_tags;
#endif
#ifdef DDS_HAS_SSM
  nn_reader_favours_ssm_t reader_favours_ssm;
#endif
  uint32_t domain_id;
  char *domain_tag;
  uint32_t cyclone_receive_buffer_size;
  unsigned char cyclone_requests_keyhash;
  unsigned char cyclone_redundant_networking;
} ddsi_plist_t;


/***/

typedef struct ddsi_plist_src {
  nn_protocol_version_t protocol_version; /**< input protocol version */
  nn_vendorid_t vendorid;                 /**< vendor code for input */
  int encoding;                           /**< PL_CDR_LE or PL_CDR_BE */
  const unsigned char *buf;               /**< input buffer */
  size_t bufsz;                           /**< size of input buffer */
  bool strict;                            /**< whether to be strict in checking */
} ddsi_plist_src_t;

DDS_EXPORT extern const ddsi_plist_t ddsi_default_plist_participant;

/**
 * @brief Initialize global parameter-list parsing indices.
 *
 * These indices are derived from compile-time constant tables.  This only does the work
 * once; ideally it would be done at compile time instead.
 */
void ddsi_plist_init_tables (void);

/**
 * @brief Initialize a ddsi_plist_t as an empty object
 *
 * In principle, this only clears the "present" and "aliased" bitmasks.  A debug build
 * additionally initializes all other bytes to 0x55.
 *
 * @param[out] dest  plist_t to be initialized.
 */
DDS_EXPORT void ddsi_plist_init_empty (ddsi_plist_t *dest);

/**
 * @brief Extend "a" with selected entries present in "b"
 *
 * This copies into "a" any entries present in "b" that are included in "pmask" and
 * "qmask" and missing in "a".  It doesn't touch any entries already present in "a".
 * Calling this on an empty "a" with all bits set in "pmask" and "qmask" all is equivalent
 * to copying "b" into "a"; calling this with "pmask" and "qmask" both 0 copies nothing.
 *
 * @param[in,out] a       ddsi_plist_t to be extended
 * @param[in]     b       ddsi_plist_t from which to copy entries
 * @param[in]     pmask   subset of non-QoS part of b (if PP_X is set, include X)
 * @param[in]     qmask   subset of QoS part of b (if QP_X is set, include X)
 */
DDS_EXPORT void ddsi_plist_mergein_missing (ddsi_plist_t *a, const ddsi_plist_t *b, uint64_t pmask, uint64_t qmask);

/**
 * @brief Copy "src" to "dst"
 *
 * @param[out]    dst     destination, any contents are overwritten
 * @param[in]     src     source ddsi_plist_t
 */
DDS_EXPORT void ddsi_plist_copy (ddsi_plist_t *dst, const ddsi_plist_t *src);

/**
 * @brief Duplicate "src"
 *
 * @param[in]     src     ddsi_plist_t to be duplicated
 *
 * @returns a new (allocated using ddsrt_malloc) ddsi_plist_t containing a copy of "src".
 */
DDS_EXPORT ddsi_plist_t *ddsi_plist_dup (const ddsi_plist_t *src);

/**
 * @brief Initialize an ddsi_plist_t from a PL_CDR_{LE,BE} paylaod.
 *
 * @param[in]  pwanted
 *               PP_... flags indicating which non-QoS parameters are of interest, treated as
 *               a hint.  Parameters in the input that are outside the mask may or may not be
 *               ignored.
 * @param[in]  qwanted
 *               QP_... flags indicating which QoS settings are of interest, and is treated as
 *               a hint.  Parameters in the input that are outside the mask may or may not be
 *               ignored.
 * @param[in]  src
 *               Serialized payload to be parsed, validated and copied into dest
 *               - protocol_version is the version protocol version according to which the list
 *                 should be parsed
 *               - vendorid is the vendor id code for the source of the parameter list (for
 *                 handling vendor-specific parameters and compatibility workarounds)
 *               - encoding is PL_CDR_LE or PL_CDR_BE
 *               - buf is a pointer to the first parameter header
 *               - bufsz is the size in bytes of the input buffer
 * @param[in]  gv
 *               Global context, used for locator kind lookups and tracing
 * @param[out] dest
 *               Filled with the recognized parameters in the input if successful, otherwise
 *               initialized to an empty parameter list.  Where possible, pointers alias the
 *               input (indicated by the "aliased" bits in the plist/xqos structures), but
 *               some things cannot be aliased (e.g., the array of pointers to strings for a
 *               sequence of strings).
 *               Generally, ddsi_plist_fini should be called when done with the parameter list,
 *               even when ddsi_plist_unlias or ddsi_xqos_unlias hasn't been called.
 * @param[out] nextafterplist
 *               If non-NULL, *nextafterplist is set to the first byte following the parameter
 *               list sentinel on successful parse, or to NULL on failure
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *               All parameters valid (or ignored), dest and *nextafterplist have been set
 *               accordingly.
 * @retval DDS_INCONSISTENT_POLICY
 *               All individual settings are valid, but there are inconsistencies between
 *               dependent settings.
 * @retval DDS_RETCODE_BAD_PARAMETER
 *               Input contained invalid data; dest is cleared, *nextafterplist is NULL.
 * @retval DDS_RETCODE_UNSUPPORTED
 *               Input contained an unrecognized parameter with the "incompatible-if-unknown"
 *               flag set; dest is cleared, *nextafterplist is NULL.
 */
DDS_EXPORT dds_return_t ddsi_plist_init_frommsg (ddsi_plist_t *dest, char **nextafterplist, uint64_t pwanted, uint64_t qwanted, const ddsi_plist_src_t *src, struct ddsi_domaingv const * const gv);

/**
 * @brief Free memory owned by "ps"
 *
 * A ddsi_plist_t may own other allocated blocks of memory, depending on which fields are
 * set, their types and whether they are marked as "aliased".  This function releases any
 * such memory owned by "ps", but not "ps" itself.  Afterward, the contents of "ps" is
 * undefined and must not be used again without initialising it (either via
 * `ddsi_plist_init_empty`, `ddsi_plist_init_frommsg` or `ddsi_plist_copy`.
 *
 * @param[in] ps   ddsi_plist_t for which to free memory
 */
DDS_EXPORT void ddsi_plist_fini (ddsi_plist_t *ps);

/**
 * @brief Free memory owned by "plist" for a subset of the entries
 *
 * A ddsi_plist_t may own other allocated blocks of memory, depending on which fields are
 * set, their types and whether they are marked as "aliased".  This function releases any
 * such memory owned by "plist" for entries included in "pmask" and "qmask".  The
 * "present" and "aliased" bits are cleared accordingly.
 *
 * @param[in,out] plist  ddsi_plist_t for which to free memory
 * @param[in]     pmask  subset of non-QoS part of b (if PP_X is set, free X if present)
 * @param[in]     qmask  subset of QoS part of b (if QP_X is set, free X if present)
 */
DDS_EXPORT void ddsi_plist_fini_mask (ddsi_plist_t *plist, uint64_t pmask, uint64_t qmask);

/**
 * @brief Replace any memory "plist" aliases by copies it owns
 *
 * A ddsi_plist_t may can reference other memory without owning it.  This functions allows
 * one to replace any such aliased memory by copies, allowing one to free the original
 * copy.
 *
 * @param[in,out] plist  ddsi_plist_t for which to replace all aliased memory by owned
 *                       copies
 */
DDS_EXPORT void ddsi_plist_unalias (ddsi_plist_t *plist);

/**
 * @brief Add selected entries in "ps" to a message in native endianness.
 *
 * This functions appends to "m" a serialized copy of the the entries selected by
 * "pwanted"/"qwanted" and present in "ps".  Each copy is preceded by a 4-byte header with
 * a parameter id and length (conform the PL_CDR representation).  It does *not* add a
 * sentinel to allow adding additional data to the parameter list.  A sentinel can be
 * added using `nn_xmsg_addpar_sentinel`.
 *
 * @param[in,out] m        message to append the parameters to
 * @param[in]     ps       source
 * @param[in]     pwanted  subset of non-QoS part of ps (if PP_X is set, add X if present)
 * @param[in]     qwanted  subset of QoS part of ps (if QP_X is set, add X if present)
 */
DDS_EXPORT void ddsi_plist_addtomsg (struct nn_xmsg *m, const ddsi_plist_t *ps, uint64_t pwanted, uint64_t qwanted);

/**
 * @brief Add selected entries in "ps" to a message with selected endianness.
 *
 * This functions appends to "m" a serialized copy of the the entries selected by
 * "pwanted"/"qwanted" and present in "ps".  Each copy is preceded by a 4-byte header with
 * a parameter id and length (conform the PL_CDR representation).  It does *not* add a
 * sentinel to allow adding additional data to the parameter list.  A sentinel can be
 * added using `nn_xmsg_addpar_sentinel`.
 *
 * @param[in,out] m        message to append the parameters to
 * @param[in]     ps       source
 * @param[in]     pwanted  subset of non-QoS part of ps (if PP_X is set, add X if present)
 * @param[in]     qwanted  subset of QoS part of ps (if QP_X is set, add X if present)
 * @param[in]     bo       byte order
 */
DDS_EXPORT void ddsi_plist_addtomsg_bo (struct nn_xmsg *m, const ddsi_plist_t *ps, uint64_t pwanted, uint64_t qwanted, enum ddsrt_byte_order_selector bo);

/**
 * @brief Determine the set of entries in which "x" differs from "y"
 *
 * This computes the entries set in "x" but not set in "y", not set in "x" but set in "y",
 * or set in both "x" and "y" but to a different value.  It returns this set reduced to
 * only those included in "pmask"/"qmask", that is, if bit X is clear in "pmask", bit X
 * will be clear in "pdelta", etc.
 *
 * @param[out] pdelta    non-QoS entries that are different and not masked out
 * @param[out] qdelta    QoS entries that are different and not masked out
 * @param[in]  x         one of the two plists to compare
 * @param[in]  y         other plist to compare
 * @param[in]  pmask     subset of non-QoS part to be compared
 * @param[in]  qmask     subset of QoS part to be compared
 */
DDS_EXPORT void ddsi_plist_delta (uint64_t *pdelta, uint64_t *qdelta, const ddsi_plist_t *x, const ddsi_plist_t *y, uint64_t pmask, uint64_t qmask);

/**
 * @brief Formats plist using `ddsi_plist_print` and writes it to the trace.
 *
 * @param[in] cat        log category to use
 * @param[in] logcfg     logging configuration
 * @param[in] plist      parameter list to be logged
 */
DDS_EXPORT void ddsi_plist_log (uint32_t cat, const struct ddsrt_log_cfg *logcfg, const ddsi_plist_t *plist);

/**
 * @brief Formats plist into a buffer
 *
 * The representation is somewhat cryptic as all enumerated types are dumped as numbers
 * and timestamps are durations as nanoseconds with "infinite" represented as
 * 9223372036854775807 (INT64_MAX).
 *
 * @param[out] buf       buffer to store the formatted representation in
 * @param[in]  bufsize   size of buffer, if > 0, there will be a terminating 0 in buf on
 *                       return
 * @param[in]  plist     parameter list to be formatted as a string
 *
 * @returns number of bytes written to buf, excluding a terminating 0.
 */
DDS_EXPORT size_t ddsi_plist_print (char * __restrict buf, size_t bufsize, const ddsi_plist_t *plist);

struct nn_rsample_info;

/**
 * @brief Scan a PL_CDR-serialized parameter list, checking structure and copying some information to "dest".
 *
 * This checks that the serialized data is structured correctly (proper aligned headers,
 * declared lengths within bounds, a sentinel at the end).  It sets the `statusinfo` of
 * `dest` to the least significant two bits (UNREGISTER and DISPOSE) of a status info
 * parameter if present, else to 0.  A statusinfo parameter that is too short (less than 4
 * bytes) is treated as an invalid input.
 *
 * It sets the `complex_qos` flag if it encounters any parameter other than a statusinfo
 * limited to those two bits, keyhash or padding, and clears it otherwise.
 *
 * It clears the `bswap` flag in `dest` if the input is in native endianness, and sets it
 * otherwise.
 *
 * @param[out] dest     internal sample info of which some fields will be set
 * @param[in]  src      input description (see `ddsi_plist_init_frommsg`)
 * @param[out] keyhashp set to point to keyhash in inline QoS if present, else to NULL
 * @param[in]  gv       global context (for logging)
 *
 * @return pointer to the first byte following the sentinel if the input is well-formed, a
 * null pointer if it is not.
*/
DDS_EXPORT unsigned char *ddsi_plist_quickscan (struct nn_rsample_info *dest, const ddsi_keyhash_t **keyhashp, const ddsi_plist_src_t *src, struct ddsi_domaingv const * const gv);

/**
 * @brief Locate a specific parameter in a PL_CDR-serialized parameter list
 *
 * This scans the serialized data until it encounters the sentinel, recording whether the
 * specified parameter occurs and returning the size and address of it in `buf`.
 *
 * If `needle` is PID_SENTINEL, it will simply check well-formedness of the input and
 * `needlep` and `needlesz` must both be null pointers.  If `needle` is not PID_SENTINEL,
 * `needlep` and `needlesz` may not be null pointers.
 *
 * @param[in]  buf       serialized parameter list to scan
 * @param[in]  bufsz     length of serialized form
 * @param[in]  encoding  encoding of `buf`, either PL_CDR_LE or PL_CDR_BE
 * @param[in]  needle    parameter id to look for
 * @param[out] needlep   where to store the address of the `needle` value
 * @param[out] needlesz  where to store the size of the `needle` value
 *
 * @return Whether input was valid and if so, whether it contains the needle.
 *
 * @retval DDS_RETCODE_BAD_PARAMETER  invalid input
 * @retval DDS_RETCODE_NOT_FOUND      valid input, `needle` not present
 * @retval DDS_RETCODE_OK             valid input, `needle` is present
*/
DDS_EXPORT dds_return_t ddsi_plist_findparam_checking (const void *buf, size_t bufsz, uint16_t encoding, nn_parameterid_t needle, void **needlep, size_t *needlesz);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_PLIST_H */
