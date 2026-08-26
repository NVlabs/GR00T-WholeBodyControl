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
#ifndef OMG_DDS_CORE_XTYPES_ANNOTATIONS_HPP_
#define OMG_DDS_CORE_XTYPES_ANNOTATIONS_HPP_

#include <dds/core/xtypes/TAnnotation.hpp>
#include <dds/core/xtypes/detail/Annotation.hpp>

namespace dds
{
namespace core
{
namespace xtypes
{
typedef TAnnotation<detail::Annotation> Annotation;


typedef TIdAnnotation<detail::IdAnnotation> IdAnnotation;

typedef TKeyAnnotation<detail::KeyAnnotation> KeyAnnotation;

typedef TSharedAnnotation<detail::SharedAnnotation> SharedAnnotation;

typedef TNestedAnnotation<detail::NestedAnnotation> NestedAnnotation;

typedef TExtensibilityAnnotation<detail::ExtensibilityAnnotation> ExtensibilityAnnotation;

typedef TMustUnderstandAnnotation<detail::MustUnderstandAnnotation> MustUnderstandAnnotation;

typedef TVerbatimAnnotation<detail::VerbatimAnnotation> VerbatimAnnotation;

typedef TBitsetAnnotation<detail::BitsetAnnotation> BitsetAnnotation;

typedef TBitBoundAnnotation<detail::BitBoundAnnotation> BitBoundAnnotation;

namespace annotation
{
// These functions can be used to get cached instances,
// to avoid the proliferation of small annotation objects.
dds::core::xtypes::IdAnnotation Id(uint32_t);
dds::core::xtypes::KeyAnnotation Key();
dds::core::xtypes::SharedAnnotation Shared();
dds::core::xtypes::NestedAnnotation Nested();
dds::core::xtypes::ExtensibilityAnnotation Extensibility(dds::core::xtypes::ExtensibilityAnnotation::ExtensibilityKind kind);
dds::core::xtypes::ExtensibilityAnnotation Final();
dds::core::xtypes::ExtensibilityAnnotation Extensible();
dds::core::xtypes::ExtensibilityAnnotation Mutable();
dds::core::xtypes::MustUnderstandAnnotation MustUnderstand();
dds::core::xtypes::VerbatimAnnotation Verbatim(const std::string& text);
dds::core::xtypes::BitsetAnnotation Bitset();
dds::core::xtypes::BitsetAnnotation BitBound(uint32_t bound);

}
}
}
}
#endif /* OMG_DDS_CORE_XTYPES_ANNOTATIONS_HPP_ */
