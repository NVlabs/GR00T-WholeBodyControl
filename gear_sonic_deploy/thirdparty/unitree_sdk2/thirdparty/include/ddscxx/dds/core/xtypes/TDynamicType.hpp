#ifndef OMG_DDS_CORE_XTYPES_T_DYNAMIC_TYPE_HPP_
#define OMG_DDS_CORE_XTYPES_T_DYNAMIC_TYPE_HPP_

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

#include <string>
#include <vector>
#include <dds/core/conformance.hpp>
#include <dds/core/Reference.hpp>
#include <dds/core/xtypes/Annotations.hpp>

#if defined (OMG_DDS_X_TYPES_DYNANIC_TYPE_SUPPORT)

namespace dds
{
namespace core
{
namespace xtypes
{
template <typename DELEGATE>
class TDynamicType;

template <typename T>
bool isPrimitiveType(const TDynamicType<T>& t);

template <typename T>
bool isConstructedType(const TDynamicType<T>& t);

template <typename T>
bool isCollectionType(const TDynamicType<T>& t);

template <typename T>
bool isAggregationType(const TDynamicType<T>& t);
}
}
}


/**
 * Base class for all dynamic types.
 */
template <typename DELEGATE>
class dds::core::xtypes::TDynamicType : public dds::core::Reference<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE(TDynamicType, dds::core::Reference, DELEGATE)

protected:
    TDynamicType(const std::string& name, TypeKind kind);
    TDynamicType(const std::string& name, TypeKind kind, const Annotation& annotation);
    TDynamicType(const std::string& name, TypeKind kind, const std::vector<Annotation>& annotations);
    template <typename FWI>
    TDynamicType(const std::string& name, TypeKind kind, const FWI& annotation_begin, const FWI& annotation_end);
    TDynamicType(const DyamicType& other);
    ~TDynamicType();
public:
    /**
     * Get the type kind.
     */
    TypeKind kind() const;

    /**
     * Get the type name.
     */
    const std::string& name() const;

    const std::vector<Annotation>& annotations() const;

public:
    bool operator == (const DynamicType& that) const;

};


#endif  // defined(OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT)


#endif // !defined(OMG_DDS_CORE_XTYPES_T_DYNAMIC_TYPE_HPP_)
