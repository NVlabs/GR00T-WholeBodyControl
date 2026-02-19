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
#ifndef OMG_DDS_CORE_XTYPES_TDYNAMICDATA_HPP_
#define OMG_DDS_CORE_XTYPES_TDYNAMICDATA_HPP_

#include <dds/core/Reference.hpp>
#include <dds/core/xtypes/DynamicType.hpp>
#include <dds/core/xtypes/MemberType.hpp>

namespace dds
{
namespace core
{
namespace xtypes
{
template <typename DELEGATE>
class TDynamicData;

template <typename DDT, typename T>
void value(DDT& dd, uint32_t mid, const T& v);

template <typename DDT, typename T>
T value(const DDT& dd, const std::string& mid, const T& v);

template <typename DDT, typename T>
T value(const DDT& dd, uint32_t mid);

template <typename DDT, typename T>
T value(const DDT& dd, const std::string& mid);


}
}
}

/**
 * This class is used to read/write data for DynamicTypes. It allows reading and
 * writing of samples in a type-safe manner but without any compile-time knowledge
 * of the type being read or written.
 */
template <typename DELEGATE>
class dds::core::xtypes::TDynamicData : dds::core::Reference<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE(TDynamicData, dds::core::Reference, DELEGATE)
public:
    TDynamicData(const DynamicType& type);

public:
    template <typename T>
    void value(uint32_T mid, const T& v) const;

    template <typename T>
    T value(const std::string& mid, const T& v) const;

    template <typename T>
    T value(uint32_T mid) const;

    template <typename T>
    T value(const std::string& mid) const;

    DynamicType type() const;

    MemberType member_type(uint32_t id) const;
    MemberType member_type(const std::string& name) const;

    uint32_t member_id(const std::string& name) const;
};


#endif /* OMG_DDS_CORE_XTYPES_TDYNAMICDATA_HPP_ */
