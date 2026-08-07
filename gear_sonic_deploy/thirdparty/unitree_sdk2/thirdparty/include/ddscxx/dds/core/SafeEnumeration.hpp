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
#ifndef OMG_DDS_CORE_SAFEENUMERATION_HPP_
#define OMG_DDS_CORE_SAFEENUMERATION_HPP_

namespace dds
{
namespace core
{

/**
 * safe_enum provides a wrapper for enumerated types in a typesafe
 * manner.
 *
 * safe_enums allow specification of the underlying type,
 * do not implictly convert to integers, and resolve scoping issues.
 */
template<typename def, typename inner = typename def::Type>
class safe_enum : public def
{
    typedef typename def::Type type;
    inner val;

public:

    safe_enum(type v) : val(v) {}
    inner underlying() const
    {
        return val;
    }

    bool operator == (const safe_enum& s) const
    {
        return this->val == s.val;
    }
    bool operator != (const safe_enum& s) const
    {
        return this->val != s.val;
    }
    bool operator < (const safe_enum& s) const
    {
        return this->val <  s.val;
    }
    bool operator <= (const safe_enum& s) const
    {
        return this->val <= s.val;
    }
    bool operator > (const safe_enum& s) const
    {
        return this->val >  s.val;
    }
    bool operator >= (const safe_enum& s) const
    {
        return this->val >= s.val;
    }
};


}
}



#endif /* OMG_DDS_CORE_SAFEENUMERATION_HPP_ */
