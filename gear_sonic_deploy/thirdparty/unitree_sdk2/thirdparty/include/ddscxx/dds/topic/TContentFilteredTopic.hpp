#ifndef OMG_DDS_T_TOPIC_CONTENT_FILTERED_TOPIC_HPP_
#define OMG_DDS_T_TOPIC_CONTENT_FILTERED_TOPIC_HPP_

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

#include <vector>

#include <dds/core/detail/conformance.hpp>
#include <dds/core/types.hpp>
#include <dds/topic/Topic.hpp>
#include <dds/topic/Filter.hpp>

#ifdef OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT

namespace dds
{
namespace topic
{
template <typename T, template <typename Q> class DELEGATE>
class ContentFilteredTopic;
}
}


/**
 * @brief
 * ContentFilteredTopic is a specialization of TopicDescription that allows
 * for content-based subscriptions.
 *
 * ContentFilteredTopic describes a more sophisticated subscription which
 * indicates that the Subscriber does not necessarily want to see all values of each
 * instance published under the Topic. Rather, it only wants to see the values whose
 * contents satisfy certain criteria. Therefore this class must be used to request
 * content-based subscriptions.
 *
 * The selection of the content is done using the SQL based filter with parameters to
 * adapt the filter clause.
 *
 * <b><i>Example</i></b>
 * @code{.cpp}
 * // Default creation of a Topic
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
 *
 * // Creation of a ContentFilteredTopic (assume Foo::Bar contains long_1 element).
 * std::vector<std::string> params;
 * params.push_back("1");
 * dds::topic::Filter filter("long_1=%0", params);
 * dds::topic::ContentFilteredTopic<Foo::Bar> cfTopic(topic,
 *                                                    "ContentFilteredTopicName",
 *                                                    filter);
 *
 * // The ContentFilteredTopic can be used to create readers
 * dds::sub::Subscriber subscriber(participant);
 * dds::sub::DataReader<Foo::Bar> reader(subscriber, cfTopic);
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_TopicDefinition "Topic Definition"
 */
template <typename T, template <typename Q> class DELEGATE>
class dds::topic::ContentFilteredTopic : public dds::topic::TTopicDescription< DELEGATE<T> >
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC_T(ContentFilteredTopic, dds::topic::TTopicDescription, T, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(ContentFilteredTopic)

    /**
     * Convenience typedef for the type of the data sample.
     */
    typedef T                               DataType;

public:
    /**
     * Creates a ContentFilteredTopic be used as to perform content-based
     * subscriptions.
     *
     * The ContentFilteredTopic only relates to samples published under that
     * Topic, filtered according to their content. The filtering is done by
     * means of evaluating a logical expression that involves the values of
     * some of the data-fields in the sample.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @param topic  the related Topic
     * @param name   the name of the ContentFilteredTopic
     * @param filter the filter expression
     * @throw dds::core::Exception
     */
    ContentFilteredTopic(const Topic<T>& topic, const std::string& name, const dds::topic::Filter& filter);

    /** @cond */
    virtual ~ContentFilteredTopic();
    /** @endcond */

public:
    /**
     * Get the filter expression.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @return the filter expression
     */
    const std::string& filter_expression() const;

    /**
     * Get the filter expression parameters.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @return the filter parameters as a sequence
     */
    const dds::core::StringSeq filter_parameters() const;

    /**
     * Sets the query parameters.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @param begin Iterator pointing to the beginning of the parameters to set
     * @param end   Iterator pointing to the end of the parameters to set
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     */
    template <typename FWDIterator>
    void filter_parameters(const FWDIterator& begin, const FWDIterator& end);

    /**
     * Return the associated Topic.
     *
     * @return the Topic
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::NullReferenceError
     *                  The entity was not properly created and references to dds::core::null.
     */
    const dds::topic::Topic<T>& topic() const;
};

#endif  // OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT


#endif /* OMG_DDS_T_TOPIC_CONTENT_FILTERED_TOPIC_HPP_ */
