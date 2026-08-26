#ifndef OMG_DDS_CORE_OPTIONAL_HPP_
#define OMG_DDS_CORE_OPTIONAL_HPP_


/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Inc.
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

namespace dds
{
namespace core
{

/**
 * The optional class is used to wrap attributes annotated in the idl with the
 * \@optional annotation. This class provides a simple and safe way of
 * accessing, setting and resetting the stored attribute.
 *
 * IDL:
 * @code
 * struct RadarTrack {
 *     string id;
 *     long x;
 *     long y;
 *     long z; //@Optional
 * };
 * @endcode
 *
 * C++ Representation:
 * @code{.cpp}
 * class RadarTrack {
 * public:
 *     RadarTrack();
 *     RadarTrack(const std::string& id,
 *                int32_t x,
 *                int32_t y,
 *                int32_t z);
 * public:
 *     std::string& id() const;
 *     void id(const std::string& s);
 *
 *     int32_t x() const;
 *     void x(int32_t v);
 *
 *     int32_t y() const;
 *     void y(int32_t v);
 *
 *     dds::core::optional<int32_t>& z() const;
 *     void z(int32_t v);
 *     void z(const dds::core::optional<int32_t>& z)
 * };
 * @endcode
 */
template <typename T, template <typename Q> class DELEGATE>
class optional : public dds::core::Value< DELEGATE<T> >
{
public:
    optional(const T& t);

public:
    /**
     * Returns true only if the attribute is set.
     */
    bool is_set() const;

    /**
     * Reset the attribute.
     */
    void reset();

    /**
     *  Get the attribute. An exception is thrown if the attribute is not set.
     */
    const T& get() const;

    /**
     *  Get the attribute. An exception is thrown if the attribute is not set.
     */
    T& get();
};

}
}
#endif /* OMG_DDS_CORE_OPTIONAL_HPP_ */
