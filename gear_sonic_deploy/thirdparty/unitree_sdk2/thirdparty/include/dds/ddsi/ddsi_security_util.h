/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSI_SECURITY_UTIL_H
#define DDSI_SECURITY_UTIL_H

#include "dds/features.h"

#ifdef DDS_HAS_SECURITY

#include "dds/ddsi/ddsi_plist.h"
#include "dds/security/core/dds_security_utils.h"

#if defined (__cplusplus)
extern "C" {
#endif

void g_omg_shallow_copy_StringSeq(DDS_Security_StringSeq *dst, const ddsi_stringseq_t *src);
void g_omg_shallow_free_StringSeq(DDS_Security_StringSeq *obj);
void q_omg_copy_PropertySeq(DDS_Security_PropertySeq *dst, const dds_propertyseq_t *src);
void q_omg_shallow_copyin_PropertySeq(DDS_Security_PropertySeq *dst, const dds_propertyseq_t *src);
void q_omg_shallow_copyout_PropertySeq(dds_propertyseq_t *dst, const DDS_Security_PropertySeq *src);
void q_omg_shallow_free_PropertySeq(DDS_Security_PropertySeq *obj);
void q_omg_shallow_copyin_BinaryPropertySeq(DDS_Security_BinaryPropertySeq *dst, const dds_binarypropertyseq_t *src);
void q_omg_shallow_copyout_BinaryPropertySeq(dds_binarypropertyseq_t *dst, const DDS_Security_BinaryPropertySeq *src);
void q_omg_shallow_free_BinaryPropertySeq(DDS_Security_BinaryPropertySeq *obj);
void q_omg_shallow_copy_PropertyQosPolicy(DDS_Security_PropertyQosPolicy *dst, const dds_property_qospolicy_t *src);
void q_omg_shallow_copy_security_qos(DDS_Security_Qos *dst, const struct dds_qos *src);
void q_omg_shallow_free_PropertyQosPolicy(DDS_Security_PropertyQosPolicy *obj);
void q_omg_shallow_free_security_qos(DDS_Security_Qos *obj);
void q_omg_security_dataholder_copyin(nn_dataholder_t *dh, const DDS_Security_DataHolder *holder);
void q_omg_security_dataholder_copyout(DDS_Security_DataHolder *holder, const nn_dataholder_t *dh);
void q_omg_shallow_copyin_DataHolder(DDS_Security_DataHolder *dst, const nn_dataholder_t *src);
void q_omg_shallow_copyout_DataHolder(nn_dataholder_t *dst, const DDS_Security_DataHolder *src);
void q_omg_shallow_free_DataHolder(DDS_Security_DataHolder *obj);
void q_omg_shallow_free_nn_dataholder(nn_dataholder_t *holder);
void q_omg_shallow_copyin_DataHolderSeq(DDS_Security_DataHolderSeq *dst, const nn_dataholderseq_t *src);
void q_omg_copyin_DataHolderSeq(DDS_Security_DataHolderSeq *dst, const nn_dataholderseq_t *src);
void q_omg_shallow_copyout_DataHolderSeq(nn_dataholderseq_t *dst, const DDS_Security_DataHolderSeq *src);
void q_omg_shallow_free_DataHolderSeq(DDS_Security_DataHolderSeq *obj);
void q_omg_shallow_free_nn_dataholderseq(nn_dataholderseq_t *obj);
void q_omg_shallow_copy_ParticipantBuiltinTopicDataSecure(DDS_Security_ParticipantBuiltinTopicDataSecure *dst, const ddsi_guid_t *guid, const ddsi_plist_t *plist);
void q_omg_shallow_free_ParticipantBuiltinTopicDataSecure(DDS_Security_ParticipantBuiltinTopicDataSecure *obj);
void q_omg_shallow_copy_SubscriptionBuiltinTopicDataSecure(DDS_Security_SubscriptionBuiltinTopicDataSecure *dst, const ddsi_guid_t *guid, const struct dds_qos *qos, const nn_security_info_t *secinfo);
void q_omg_shallow_free_SubscriptionBuiltinTopicDataSecure(DDS_Security_SubscriptionBuiltinTopicDataSecure *obj);
void q_omg_shallow_copy_PublicationBuiltinTopicDataSecure(DDS_Security_PublicationBuiltinTopicDataSecure *dst, const ddsi_guid_t *guid, const struct dds_qos *qos, const nn_security_info_t *secinfo);
void q_omg_shallow_free_PublicationBuiltinTopicDataSecure(DDS_Security_PublicationBuiltinTopicDataSecure *obj);
void q_omg_shallow_copy_TopicBuiltinTopicData(DDS_Security_TopicBuiltinTopicData *dst, const char *topic_name, const char *type_name);
void q_omg_shallow_free_TopicBuiltinTopicData(DDS_Security_TopicBuiltinTopicData *obj);

#if defined (__cplusplus)
}
#endif


#endif /* DDS_HAS_SECURITY */

#endif /* DDSI_SECURITY_UTIL_H */
