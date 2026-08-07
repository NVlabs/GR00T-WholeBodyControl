#ifndef OMG_DDS_SUB_SAMPLE_HPP_
#define OMG_DDS_SUB_SAMPLE_HPP_

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

#include <dds/sub/detail/Sample.hpp>

namespace dds
{
namespace sub
{
template <typename T, template <typename Q> class DELEGATE = dds::sub::detail::Sample >
class Sample;
}
}

/**
 * @todo RTF Issue - added include here
 * @note This include needs to be here to enable MSVC compilation,
 * or TSample.hpp must set the above default DELEGATE instead of it being here.
 */
#include <dds/sub/TSample.hpp>

#endif /* OMG_DDS_SUB_SAMPLE_HPP_ */
