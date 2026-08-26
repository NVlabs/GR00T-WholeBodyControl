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
#ifndef OMG_DDS_SUB_DATA_READER_HPP_
#define OMG_DDS_SUB_DATA_READER_HPP_

#include <dds/sub/detail/DataReader.hpp>

namespace dds {
    namespace sub {

        template < typename T,
           template <typename Q> class DELEGATE = dds::sub::detail::DataReader >
           class DataReader;
    }
}

#include <dds/sub/TDataReader.hpp>


// = Manipulators
namespace dds
{
namespace sub
{
namespace functors
{
typedef dds::sub::functors::detail::MaxSamplesManipulatorFunctor      MaxSamplesManipulatorFunctor;
typedef dds::sub::functors::detail::ContentFilterManipulatorFunctor   ContentFilterManipulatorFunctor;
typedef dds::sub::functors::detail::StateFilterManipulatorFunctor   StateFilterManipulatorFunctor;
typedef dds::sub::functors::detail::InstanceManipulatorFunctor     InstanceManipulatorFunctor;
typedef dds::sub::functors::detail::NextInstanceManipulatorFunctor   NextInstanceManipulatorFunctor;
}
}
}

namespace dds
{
namespace sub
{

template <typename SELECTOR>
SELECTOR& read(SELECTOR& selector);

template <typename SELECTOR>
SELECTOR& take(SELECTOR& selector);

inline dds::sub::functors::MaxSamplesManipulatorFunctor
max_samples(uint32_t n);

inline dds::sub::functors::ContentFilterManipulatorFunctor
content(const dds::sub::Query& query);

inline dds::sub::functors::StateFilterManipulatorFunctor
state(const dds::sub::status::DataState& s);

inline dds::sub::functors::InstanceManipulatorFunctor
instance(const dds::core::InstanceHandle& h);

inline dds::sub::functors::NextInstanceManipulatorFunctor
next_instance(const dds::core::InstanceHandle& h);

}
}


#endif /* OMG_DDS_SUB_DATA_READER_HPP_ */
