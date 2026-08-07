#ifndef OMG_DDS_CORE_MACROS_HPP_
#define OMG_DDS_CORE_MACROS_HPP_

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

#include <dds/core/detail/macros.hpp>

// == Static Assert =========================================================
#define OMG_DDS_STATIC_ASSERT OMG_DDS_STATIC_ASSERT_DETAIL
// ==========================================================================


// == Logging Macros ========================================================
#define OMG_DDS_LOG(kind, msg) OMG_DDS_LOG_DETAIL(kind, msg)
// ==========================================================================


// == State Width Constants =================================================
/* By default, there can be up to 32 unique statuses. However, a
 * DDS implementation may extend this number if necessary.
 */
#define OMG_DDS_DEFAULT_STATE_BIT_COUNT OMG_DDS_DEFAULT_STATE_BIT_COUNT_DETAIL
#define OMG_DDS_DEFAULT_STATUS_COUNT    OMG_DDS_DEFAULT_STATUS_COUNT_DETAIL

#ifndef OMG_DDS_STATE_BIT_COUNT
#define OMG_DDS_STATE_BIT_COUNT         OMG_DDS_DEFAULT_STATE_BIT_COUNT
#endif  // !defined(OMG_DDS_STATE_BIT_COUNT)

#ifndef OMG_DDS_STATUS_COUNT
#define OMG_DDS_STATUS_COUNT            OMG_DDS_DEFAULT_STATUS_COUNT
#endif  // !defined(OMG_DDS_STATUS_COUNT)
// ==========================================================================


// == DLL Export Macros =====================================================
#define OMG_DDS_API OMG_DDS_API_DETAIL
// ==========================================================================

#define TEMP_TYPE void*

#define OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_DEFAULT(TYPE) \
public:\
    virtual ~TYPE() = default;\
    TYPE(const TYPE& tocopy) = default;\
    TYPE(TYPE&& tomove) = default;\
    virtual TYPE& operator=(const TYPE& tocopy) = default;\
    virtual TYPE& operator=(TYPE&& tomove) { if (this != &tomove) *this = tomove; return *this;}

#define OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_EXPLICIT(TYPE) \
public:\
    virtual ~TYPE() {}\
    TYPE(const TYPE& tocopy): dds::core::Reference<DELEGATE>(dynamic_cast<const dds::core::Reference<DELEGATE>& >(tocopy)) {}\
    TYPE(TYPE&& tomove): dds::core::Reference<DELEGATE>(dynamic_cast<dds::core::Reference<DELEGATE>&& >(tomove)) {}\
    virtual TYPE& operator=(const TYPE& tocopy) {this->dds::core::Reference<DELEGATE>::operator=(dynamic_cast<const dds::core::Reference<DELEGATE>& >(tocopy)); return *this;}\
    virtual TYPE& operator=(TYPE&& tomove) { if (this != &tomove) this->dds::core::Reference<DELEGATE>::operator=(dynamic_cast<dds::core::Reference<DELEGATE>&& >(tomove)); return *this;}

#endif /* OMG_DDS_CORE_MACROS_HPP_*/
