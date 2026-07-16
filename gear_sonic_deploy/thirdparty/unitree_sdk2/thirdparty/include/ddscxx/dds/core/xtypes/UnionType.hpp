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
#ifndef OMG_DDS_CORE_XTYPES_T_UNION_TYPE_HPP_
#define OMG_DDS_CORE_XTYPES_T_UNION_TYPE_HPP_

#include <dds/core/xtypes/DynamicType.hpp>
#include <dds/core/xtypes/UnionCase.hpp>
#include <dds/core/xtypes/PrimitiveTypes.hpp>
#include <dds/core/xtypes/detail/UnionType.hpp>
#include <vector>

namespace dds
{
namespace core
{
namespace xtypes
{
template <typename DELEGATE>
class TUnionForwardDeclaration;

template <typename T, template <typename Q> class DELEGATE = detail::UnionType >
class UnionType;
}
}
}

/**
 * Declares a forward declaration for a union type.
 */
template <typename DELEGATE>
class dds::core::xtypes::TUnionForwardDeclaration : public dds::core::xtypes::DynamicType<DELEGATE>
{
public:
    TUnionForwardDeclaration(const std::string& name);
};

template <typename T, typename DELEGATE>
class dds::core::xtypes::UnionType  : public dds::core::xtypes::DynamicType<DELEGATE<T> >
{
public:

    UnionType(
        const std::string& name,
        const TPrimitiveType<T>& discriminator_type,
        const std::vector<UnionCase<T> >& cases);

    UnionType(
        const std::string& name,
        const TPrimitiveType<T>& discriminator_type,
        const std::vector<UnionCase<T> >& cases,
        const Annotation& annotation);

    UnionType(
        const std::string& name,
        const TPrimitiveType<T>& discriminator_type,
        const std::vector<UnionCase<T> >& cases,
        const std::vector<Annotation>& annotations);

public:

    const std::vector<UnionCase<T> >& members() const;
    const MemberType& member(uint32_t id) const;
    const MemberType& member(const std::string& name) const;

    const std::vector<Annotation>& annotations() const;

    UnionType add_member(const UnionCase<T>& member) const;
    UnionType remove_member(const UnionCase<T>& member) const;

    UnionType add_annotation(const Annotation& annotation) const;
    UnionType remove_annotation(const Annotation& annotation) const;

};

#endif /* OMG_DDS_CORE_XTYPES_T_STRUCT_TYPE_HPP_ */
