#ifndef OMG_DDS_SUB_PACKAGE_INCLUDE_HPP_
#define OMG_DDS_SUB_PACKAGE_INCLUDE_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Corp.
 * Copyright 2010, Real-Time Innovations, Inc.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dds/core/detail/conformance.hpp>

/* Depends on:
 *   <dds/sub/Subscriber.hpp>
 *   <dds/sub/status/DataStatus.hpp>
 *   <dds/sub/Sample.hpp>
 *   <dds/sub/LoanedSamples.hpp>
 */
#include <dds/sub/DataReader.hpp>

/////////////////////////////////////////////////////////////////////////////
// -- Status Includes
#include <dds/sub/status/DataState.hpp>


/////////////////////////////////////////////////////////////////////////////
// -- QoS Includes
#include <dds/sub/qos/DataReaderQos.hpp>
#include <dds/sub/qos/SubscriberQos.hpp>

/////////////////////////////////////////////////////////////////////////////
// -- Sub Includes

#include <dds/sub/Rank.hpp>
#include <dds/sub/GenerationCount.hpp>

/* Depends on:
 *   Forward declaration of AnyDataReader
 */
#include <dds/sub/AnyDataReaderListener.hpp>


/*
 * Depends on:
 *  <dds/sub/GenerationCount.hpp>
 *  <dds/sub/Rank.hpp>
 */

#include <dds/sub/SampleInfo.hpp>

#include <dds/sub/LoanedSamples.hpp>
#include <dds/sub/SharedSamples.hpp>

/* Depends on:
 *  <dds/sub/qos/SubscriberQos.hpp>
 *  <dds/sub/qos/DataReaderQos.hpp>
 */
#include <dds/sub/Subscriber.hpp>


/* Depends on:
 *    <dds/sub/Subscriber.hpp>
 */
#include <dds/sub/CoherentAccess.hpp>

/* Depends on:
 * <dds/sub/AnyDataReaderListener.hpp>
 * <dds/sub/Subscriber.hpp>
 */
#include <dds/sub/SubscriberListener.hpp>


/* Depends on:
 *    <dds/sub/AnyDataReaderListener.hpp>
 */
#include <dds/sub/DataReaderListener.hpp>

/* Depends on:
 *  <dds/sub/Subscriber.hpp>
 *  <dds/sub/DataReader.hpp>
 */
#include <dds/sub/AnyDataReader.hpp>

/////////////////////////////////////////////////////////////////////////////
// -- Condition Includes
/* Depends on:
 *   <dds/sub/DataReader.hpp>
 */
#include <dds/sub/cond/ReadCondition.hpp>
#include <dds/sub/Query.hpp>

/* Depends on:
 *    <dds/sub/status/DataState.hpp>
 *    <dds/sub/cond/ReadCondition.hpp>
 *    <dds/sub/Query.hpp>
 */
#include <dds/sub/cond/QueryCondition.hpp>

/* Depends on:
 *   <dds/sub/Subscriber.hpp>
 *   <dds/sub/status/ReaderState.hpp>
 */
#include <dds/sub/find.hpp>
#include <dds/sub/discovery.hpp>

//
// Pretty Print Utilities
//
// #if (OMG_DDS_HAS_PRETTY_PRINT_COUT == 1)
std::ostream& operator << (std::ostream& os, const dds::sub::status::DataState& s);
std::ostream& operator << (std::ostream& os, const dds::sub::Rank& r);
std::ostream& operator << (std::ostream& os, const dds::sub::SampleInfo& si);

#include <dds/sub/detail/ddssub.hpp>

// #endif

#endif /* OMG_DDS_SUB_PACKAGE_INCLUDE_HPP_ */
