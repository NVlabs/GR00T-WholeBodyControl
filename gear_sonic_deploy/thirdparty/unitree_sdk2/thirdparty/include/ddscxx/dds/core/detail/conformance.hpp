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
#ifndef CYCLONEDDS_DDS_CORE_DETAIL_CONFORMANCE_HPP_
#define CYCLONEDDS_DDS_CORE_DETAIL_CONFORMANCE_HPP_

// Implementation

/**
 * @file
 * @internal
 * @note Values 'set' in this file should be mirrored in etc/doxygen_isocpp2_common.cfg
 * in order to ensure the doxygen documentation includes all supported QoS
 * and features and ting.
 */

#define OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT                FULL
// #define OMG_DDS_MULTI_TOPIC_SUPPORT                         FULL
#define OMG_DDS_PERSISTENCE_SUPPORT                         FULL
#define OMG_DDS_OWNERSHIP_SUPPORT                           FULL
#define OMG_DDS_OBJECT_MODEL_SUPPORT                        FULL
#define OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT   FULL
// #define OMG_DDS_X_TYPES_BUILTIN_TOPIC_TYPES_SUPPORT         FULL

#define OMG_DDS_HAS_PRETTY_PRINT_COUT 1
// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_DETAIL_CONFORMANCE_HPP_ */
