#ifndef OMG_DDS_CORE_TYPES_HPP_
#define OMG_DDS_CORE_TYPES_HPP_

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

// ISO C++ Includes
#include <string>
#include <vector>

// DDS Includes
#include <dds/core/detail/inttypes.hpp>
#include <dds/core/macros.hpp>
#include <dds/core/detail/conformance.hpp>

namespace dds
{
namespace core
{
/**
 * Use a std::vector<uint8_t> to hold a sequence of bytes.
 */
typedef std::vector<uint8_t> ByteSeq;

/**
 * Use a std::vector<std::string> to hold a sequence of bytes.
 */
typedef std::vector<std::string> StringSeq;

// DDS Null-Reference
/**
 * @brief This class is used to create dds::core::null objects.
 */
class OMG_DDS_API null_type { };
/**
 * This is the DDS Null-Reference.<br>
 * A dds reference object that doesn't reference to anything can be compared with this object.
 * @code{.cpp}
 * dds::domain::DomainParticipant participant = dds::core::null;
 * ...
 * if (participant == dds::core::null) {
 *     // The participant is not yet properly created.
 *     // Using it now will trigger the dds::core::NullReferenceError exception.
 * }
 * @endcode
 */
extern const null_type OMG_DDS_API null;

/** @cond
 * Duplicate in CorePolicy.hpp. Why?
 */
namespace policy
{
typedef uint32_t QosPolicyId;
}
/** @endcond */
}
}

#endif /* OMG_DDS_CORE_TYPES_HPP_ */
