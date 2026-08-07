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
#ifndef OMG_DDS_CORE_XTYPES_TANNOTATIONS_HPP_
#define OMG_DDS_CORE_XTYPES_TANNOTATIONS_HPP_

#include <dds/core/Reference.hpp>

namespace dds
{
namespace core
{
namespace xtypes
{

struct AnnotationKind_def
{
    enum type
    {
        ID_ANNOTATION_TYPE,
        OPTIONAL_ANNOTATION_TYPE,
        KEY_ANNOTATION_TYPE,
        SHARED_ANNOTATION_TYPE,
        NESTED_ANNOTATION_TYPE,
        EXTENSIBILITY_ANNOTATION_TYPE,
        MUST_UNDERSTAND_ANNOTATION_TYPE,
        VERBATIM_ANNOTATION_TYPE,
        BITSET_ANNOTATION_TYPE
    };
};

typedef dds::core::safe_enum<AnnotationKind_def> AnnotationKind;

struct ExtensibilityKind_def
{
    enum type
    {
        FINAL,
        EXTENSIBLE,
        MUTABLE
    };
};
typedef dds::core::safe_enum<ExtensibilityKind_def> ExtensibilityKind;


template <typename DELEGATE>
class TAnnotation;

template <typename DELEGATE>
class TIdAnnotation;

template <typename DELEGATE>
class TKeyAnnotation;

template <typename DELEGATE>
class TSharedAnnotation;

template <typename DELEGATE>
class TNestedAnnotation;

template <typename DELEGATE>
class TExtensibilityAnnotation;

template <typename DELEGATE>
class TMustUnderstandAnnotation;

template <typename DELEGATE>
class TVerbatimAnnotation;

template <typename DELEGATE>
class TBitsetAnnotation;

template <typename DELEGATE>
class TBitBoundAnnotation;

}
}
}

template <typename DELEGATE>
class dds::core::xtypes::TAnnotation : public dds::core::Reference<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE(TAnnotation, dds::core::Reference, DELEGATE)

public:
    TAnnotation();
protected:
    TAnnotation(const TypeKind& kind);
public:
    TypeKind kind() const;
};
template <typename DELEGATE>
class dds::core::xtypes::TIdAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{
public:
    TIdAnnotation(uint32_t id);
public:
    uint32_t id() const;
};
template <typename DELEGATE>
class dds::core::xtypes::TKeyAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{
public:
    TKeyAnnotation();
};
template <typename DELEGATE>
class dds::core::xtypes::TSharedAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{
public:
    TSharedAnnotation();
};
template <typename DELEGATE>
class dds::core::xtypes::TNestedAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{
public:
    TNestedAnnotation();
};

template <typename DELEGATE>
class dds::core::xtypes::TExtensibilityAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{

public:
    TExtensibilityAnnotation(ExtensibilityKind xkind);

public:
    ExtensibilityKind extensibility_kind() const;
};

template <typename DELEGATE>
class dds::core::xtypes::TMustUnderstandAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{
public:
    TMustUnderstandAnnotation();
};

template <typename DELEGATE>
class dds::core::xtypes::TVerbatimAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{
public:
    TVerbatimAnnotation(const std::string& text);
public:
    const std::string& verbatim_text() const;
};

template <typename DELEGATE>
class dds::core::xtypes::TBitsetAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{
public:
    TBitsetAnnotation();

};

template <typename DELEGATE>
class  dds::core::xtypes::TBitBoundAnnotation : public dds::core::xtypes::TAnnotation<DELEGATE>
{
public:
    TBitBoundAnnotation(uint32_t bound);
};


#endif /* OMG_DDS_CORE_XTYPES_TANNOTATIONS_HPP_ */
