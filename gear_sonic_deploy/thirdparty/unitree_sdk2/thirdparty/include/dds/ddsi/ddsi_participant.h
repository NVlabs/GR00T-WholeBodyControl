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
#ifndef DDSI_PARTICIPANT_H
#define DDSI_PARTICIPANT_H

#include "dds/export.h"
#include "dds/features.h"

#include "dds/ddsrt/avl.h"
#include "dds/ddsrt/fibheap.h"
#include "dds/ddsrt/sync.h"
#include "dds/ddsi/ddsi_entity.h"

#if defined (__cplusplus)
extern "C" {
#endif

#define DPG_LOCAL 1
#define DPG_REMOTE 2

extern const ddsrt_fibheap_def_t ldur_fhdef;
extern const ddsrt_fibheap_def_t lease_fhdef_pp;

struct deleted_participant {
  ddsrt_avl_node_t avlnode;
  ddsi_guid_t guid;
  unsigned for_what;
  ddsrt_mtime_t t_prune;
};

struct deleted_participants_admin {
  ddsrt_mutex_t deleted_participants_lock;
  ddsrt_avl_tree_t deleted_participants;
  const ddsrt_log_cfg_t *logcfg;
  int64_t delay;
};

struct avail_entityid_set {
  struct inverse_uint32_set x;
};

enum ddsi_participant_state {
  DDSI_PARTICIPANT_STATE_INITIALIZING,
  DDSI_PARTICIPANT_STATE_OPERATIONAL,
  DDSI_PARTICIPANT_STATE_DELETE_STARTED,
  DDSI_PARTICIPANT_STATE_DELETING_BUILTINS
};

struct ddsi_participant
{
  struct ddsi_entity_common e;
  dds_duration_t lease_duration; /* constant */
  uint32_t bes; /* built-in endpoint set */
  unsigned is_ddsi2_pp: 1; /* true for the "federation leader", the ddsi2 participant itself in OSPL; FIXME: probably should use this for broker mode as well ... */
  struct ddsi_plist *plist; /* settings/QoS for this participant */
  struct xevent *spdp_xevent; /* timed event for periodically publishing SPDP */
  struct xevent *pmd_update_xevent; /* timed event for periodically publishing ParticipantMessageData */
  ddsi_locator_t m_locator; /* this is always a unicast address, it is set if it is in the many unicast mode */
  ddsi_tran_conn_t m_conn; /* this is connection to m_locator, if it is set, this is used */
  struct avail_entityid_set avail_entityids; /* available entity ids [e.lock] */
  ddsrt_mutex_t refc_lock;
  int32_t user_refc; /* number of non-built-in endpoints in this participant [refc_lock] */
  int32_t builtin_refc; /* number of built-in endpoints in this participant [refc_lock] */
  enum ddsi_participant_state state; /* current state of this participant [refc_lock] */
  ddsrt_fibheap_t ldur_auto_wr; /* Heap that contains lease duration for writers with automatic liveliness in this participant */
  ddsrt_atomic_voidp_t minl_man; /* clone of min(leaseheap_man) */
  ddsrt_fibheap_t leaseheap_man; /* keeps leases for this participant's writers (with liveliness manual-by-participant) */
#ifdef DDS_HAS_SECURITY
  struct participant_sec_attributes *sec_attr;
  nn_security_info_t security_info;
#endif
};

/* Interface for glue code between the OpenSplice kernel and the DDSI
   entities. These all return 0 iff successful. All GIDs supplied
   __MUST_BE_UNIQUE__. All hell may break loose if they aren't.

   All delete operations synchronously remove the entity being deleted
   from the various global hash tables on GUIDs. This ensures no new
   operations can be invoked by the glue code, discovery, protocol
   messages, &c.  The entity is then scheduled for garbage collection.

     There is one exception: a participant without built-in
     endpoints: that one synchronously reaches reference count zero
     and is then freed immediately.

     If ddsi_new_writer () and/or ddsi_new_reader () may be called in parallel to
     ddsi_delete_participant (), trouble ensues. The current glue code
     performs all local discovery single-threaded, and can't ever get
     into that issue.

   A garbage collector thread is used to perform the actual freeing of
   an entity, but it never does so before all threads have made
   sufficient progress to guarantee they are not using that entity any
   longer, with the exception of use via internal pointers in the
   entity data structures.

   An example of the latter is that (proxy) endpoints have a pointer
   to the owning (proxy) participant, but the (proxy) participant is
   reference counted to make this safe.

   The case of a proxy writer is particularly complicated is it has to
   pass through a multiple-stage delay in the garbage collector before
   it may be freed: first there is the possibility of a parallel
   delete or protocol message, then there is still the possibility of
   data in a delivery queue.  This is dealt by requeueing garbage
   collection and sending bubbles through the delivery queue. */

/* Set this flag in new_participant to prevent the creation SPDP, SEDP
   and PMD readers for that participant.  It doesn't really need it,
   they all share the information anyway.  But you do need it once. */
#define RTPS_PF_NO_BUILTIN_READERS 1u
/* Set this flag to prevent the creation of SPDP, SEDP and PMD
   writers.  It will then rely on the "privileged participant", which
   must exist at the time of creation.  It creates a reference to that
   "privileged participant" to ensure it won't disappear too early. */
#define RTPS_PF_NO_BUILTIN_WRITERS 2u
/* Set this flag to mark the participant as the "privileged
   participant", there can only be one of these.  The privileged
   participant MUST have all builtin readers and writers. */
#define RTPS_PF_PRIVILEGED_PP 4u
  /* Set this flag to mark the participant as is_ddsi2_pp. */
#define RTPS_PF_IS_DDSI2_PP 8u
  /* Set this flag to mark the participant as an local entity only. */
#define RTPS_PF_ONLY_LOCAL 16u


void ddsi_participant_add_wr_lease_locked (struct ddsi_participant * pp, const struct ddsi_writer * wr);
void ddsi_participant_remove_wr_lease_locked (struct ddsi_participant * pp, struct ddsi_writer * wr);
dds_return_t ddsi_participant_allocate_entityid (ddsi_entityid_t *id, uint32_t kind, struct ddsi_participant *pp);
void ddsi_participant_release_entityid (struct ddsi_participant *pp, ddsi_entityid_t id);
void ddsi_gc_participant_lease (struct gcreq *gcreq);
void ddsi_prune_deleted_participant_guids (struct deleted_participants_admin *admin, ddsrt_mtime_t tnow);
void ddsi_remove_deleted_participant_guid (struct deleted_participants_admin *admin, const struct ddsi_guid *guid, unsigned for_what);
void ddsi_remember_deleted_participant_guid (struct deleted_participants_admin *admin, const struct ddsi_guid *guid);
struct ddsi_participant *ddsi_ref_participant (struct ddsi_participant *pp, const struct ddsi_guid *guid_of_refing_entity);
void ddsi_unref_participant (struct ddsi_participant *pp, const struct ddsi_guid *guid_of_refing_entity);
struct deleted_participants_admin *ddsi_deleted_participants_admin_new (const ddsrt_log_cfg_t *logcfg, int64_t delay);
void ddsi_deleted_participants_admin_free (struct deleted_participants_admin *admin);
int ddsi_is_deleted_participant_guid (struct deleted_participants_admin *admin, const struct ddsi_guid *guid, unsigned for_what);



/**
 * @brief Create a new participant in the domain
 *
 * @param[out] ppguid
 *               On successful return: the GUID of the new participant;
 *               Undefined on error.
 * @param[in]  flags
 *               Zero or more of:
 *               - RTPS_PF_NO_BUILTIN_READERS   do not create discovery readers in new ppant
 *               - RTPS_PF_NO_BUILTIN_WRITERS   do not create discvoery writers in new ppant
 *               - RTPS_PF_PRIVILEGED_PP        FIXME: figure out how to describe this ...
 *               - RTPS_PF_IS_DDSI2_PP          FIXME: OSPL holdover - there is no DDSI2E here
 *               - RTPS_PF_ONLY_LOCAL           FIXME: not used, it seems
 * @param[in]  plist
 *               Parameters/QoS for this participant
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *               Success, there is now a local participant with the GUID stored in
 *               *ppguid
 * @retval DDS_RETCODE_OUT_OF_RESOURCES
 *               Failed to allocate a new GUID (note: currently this will always
 *               happen after 2**24-1 successful calls to new_participant ...).
 * @retval DDS_RETCODE_OUT_OF_RESOURCES
 *               The configured maximum number of participants has been reached.
*/
DDS_EXPORT dds_return_t ddsi_new_participant (struct ddsi_guid *ppguid, struct ddsi_domaingv *gv, unsigned flags, const struct ddsi_plist *plist);

/**
 * @brief Initiate the deletion of the participant:
 * - dispose/unregister built-in topic
 * - list it as one of the recently deleted participants
 * - remote it from the GUID hash tables
 * - schedule the scare stuff to really delete it via the GC
 *
 * It is ok to call delete_participant without deleting all DDSI-level
 * readers/writers: those will simply be deleted.  (New ones can't be
 * created anymore because the participant can no longer be located via
 * the hash tables).
 *
 * @param[in]  ppguid
 *               GUID of the participant to be deleted.
 *
 * @returns A dds_return_t indicating success or failure.
 *
 * @retval DDS_RETCODE_OK
 *               Success, it is no longer visible and GC events have
 *               been scheduled for eventual deleting of all remaining
 *               readers and writers and freeing of memory
 * @retval DDS_RETCODE_BAD_PARAMETER
 *               ppguid lookup failed.
*/
DDS_EXPORT dds_return_t ddsi_delete_participant (struct ddsi_domaingv *gv, const struct ddsi_guid *ppguid);

/**
 * @brief Updates the parameters list for this participant and trigger sending
 * an updated SPDP message.
 *
 * @param[in] pp The participant
 * @param[in] plist The new parameters
 */
DDS_EXPORT void ddsi_update_participant_plist (struct ddsi_participant *pp, const struct ddsi_plist *plist);

/**
 * @brief Gets the interval for PMD messages, which is the minimal lease duration for writers
 * with auto liveliness in this participant, or the participants lease duration if shorter
 *
 * @param[in] pp The participant
 * @returns The PMD interval of the participant
 */
DDS_EXPORT dds_duration_t ddsi_participant_get_pmd_interval (struct ddsi_participant *pp);

/**
 * @brief To obtain the builtin writer to be used for publishing SPDP, SEDP, PMD stuff for
 * PP and its endpoints, given the entityid. If PP has its own writer, use it; else use the
 * privileged participant.
 *
 * @param[in] pp The participant
 * @param[in] entityid The entity ID of the writer
 * @returns The built-in writer
 */
DDS_EXPORT struct ddsi_writer *ddsi_get_builtin_writer (const struct ddsi_participant *pp, unsigned entityid);

#if defined (__cplusplus)
}
#endif

#endif /* DDSI_PARTICIPANT_H */
