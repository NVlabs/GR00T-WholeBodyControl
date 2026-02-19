#ifndef OMG_DDS_TOPIC_TOPIC_TRAITS_HPP_
#define OMG_DDS_TOPIC_TOPIC_TRAITS_HPP_

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

//==============================================================================
namespace dds
{
namespace topic
{
template <typename T>
struct is_topic_type;

/** @cond */
template <typename T>
struct topic_type_support;
/** @endcond */

/**
 * @brief
 * Support functionality to get the default type_name of a Topic type.
 *
 * @code{.cpp}
 * std::string typeName = dds::topic::topic_type_name<Foo::Bar>::value();
 * @endcode
 */
template <typename T>
struct topic_type_name
{
    static std::string value()
    {
        return "Undefined";
    }
};

}
}

//==============================================================================
/**
 * @brief
 * Support functionality to check if a given object type is a Topic.
 *
 * @code{.cpp}
 * if (dds::topic::is_topic_type<Foo::Bar>::value) {
 *     // Foo::Bar type is considered a Topic
 * } else {
 *     // Foo::Bar type is NOT considered a Topic
 * }
 * @endcode
 */
template <typename T>
struct dds::topic::is_topic_type
{
    enum {value = 0 };
};


/** @cond
 * IsoCpp doesn't know the TypeSupport concept. The type is automatically
 * registered when an Topic is created. So, what does this function do?
 */
template <typename T>
struct dds::topic::topic_type_support { };
/** @endcond */


#define REGISTER_TOPIC_TYPE(TOPIC_TYPE) \
    namespace dds { namespace topic { \
    template<> struct is_topic_type<TOPIC_TYPE> { \
        enum { value = 1 }; \
    }; } }

#endif /* OMG_DDS_TOPIC_TOPIC_TRAITS_HPP_ */
