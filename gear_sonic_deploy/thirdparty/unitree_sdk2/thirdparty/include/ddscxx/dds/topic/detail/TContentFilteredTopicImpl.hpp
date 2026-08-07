/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_DDS_TOPIC_TCONTENTFILTEREDTOPIC_HPP_
#define CYCLONEDDS_DDS_TOPIC_TCONTENTFILTEREDTOPIC_HPP_

/**
 * @file
 */

/*
 * OMG PSM class declaration
 */
#include <dds/topic/TContentFilteredTopic.hpp>

// Implementation

namespace dds
{
namespace topic
{
template <typename T, template <typename Q> class DELEGATE>
ContentFilteredTopic<T, DELEGATE>::ContentFilteredTopic(const Topic<T>& topic,
                                                        const std::string& name,
                                                        const dds::topic::Filter& filter) :
        ::dds::core::Reference< DELEGATE<T> >(
                new dds::topic::detail::ContentFilteredTopic<T>(topic, name, filter))
{
    this->delegate()->init(::dds::core::Reference< DELEGATE<T> >::impl_);
}

template <typename T, template <typename Q> class DELEGATE>
ContentFilteredTopic<T, DELEGATE>::~ContentFilteredTopic()
{
    // Nothing to be done yet....
}

template <typename T, template <typename Q> class DELEGATE>
const std::string& ContentFilteredTopic<T, DELEGATE>::filter_expression() const
{
    return this->delegate()->filter_expression();
}

template <typename T, template <typename Q> class DELEGATE>
const dds::core::StringSeq ContentFilteredTopic<T, DELEGATE>::filter_parameters() const
{
    return this->delegate()->filter_parameters();
}

template <typename T, template <typename Q> class DELEGATE>
template <typename FWDIterator>
void ContentFilteredTopic<T, DELEGATE>::filter_parameters(const FWDIterator& begin, const FWDIterator& end)
{
    this->delegate()->filter_parameters(begin, end);
}

template <typename T, template <typename Q> class DELEGATE>
const dds::topic::Topic<T>& ContentFilteredTopic<T, DELEGATE>::topic() const
{
    return this->delegate()->topic();
}


}
}

// End of implementation

#endif /* CYCLONEDDS_DDS_TOPIC_TCONTENTFILTEREDTOPIC_HPP_ */
