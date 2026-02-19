#ifndef DDS_TOPIC_TFILTER__HPP_
#define DDS_TOPIC_TFILTER__HPP_

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

#include <dds/core/types.hpp>
#include <dds/core/Value.hpp>

namespace dds
{
namespace topic
{
template <typename D>
class TFilter;
}
}

/**
 * @brief
 * Filter objects contain SQL expressions that allow the application to specify
 * a filter on the locally available data.
 *
 * A Filter is used to create a ContentFilteredTopic.
 *
 * @anchor anchor_dds_topic_filter_expression
 * \par SQL Expression
 * The SQL query string is set by expression which must be a subset of the
 * SQL query language. In this query expression, parameters may be used, which must
 * be set in the sequence of strings defined by the parameter query_parameters. A
 * parameter is a string which can define an integer, float, string or enumeration. The
 * number of values in query_parameters must be equal or greater than the highest
 * referenced %n token in the query_expression (e.g. if %1 and %8 are used as
 * parameters in the query_expression, the query_parameters should at least
 * contain n+1 = 9 values).<br>
 * Look @ref DCPS_Queries_and_Filters "here" for the specific query expression syntax.
 *
 * @see dds::topic::ContentFilteredTopic
 * @see for more information: @ref DCPS_Modules_TopicDefinition "Topic Definition"
 */
template <typename D>
class dds::topic::TFilter: public dds::core::Value<D>
{
public:
    /**
     * Iterator for the query expression parameters.
     */
    typedef typename D::iterator iterator;

    /**
     * Iterator for the query expression parameters.
     */
    typedef typename D::const_iterator const_iterator;

public:
    /**
     * Create a Filter based on a query expression.
     *
     * @param  query_expression an @ref anchor_dds_topic_filter_expression "SQL expression"
     * @throw                   dds::core::Exception
     */
    TFilter(const std::string& query_expression);

    /**
     * Create a Filter based on a query expression and an iterable parameter container.
     *
     * @param  query_expression an @ref anchor_dds_topic_filter_expression "SQL expression"
     * @param params_begin     Iterator pointing to the beginning of the parameters to set
     * @param params_end       Iterator pointing to the end of the parameters to set
     * @throw                   dds::core::Exception
     */
    template <typename FWIterator>
    TFilter(const std::string& query_expression, const FWIterator& params_begin,
            const FWIterator& params_end);

    /**
     * Create a Filter based on a query expression and parameter vector.
     *
     * @param  query_expression an @ref anchor_dds_topic_filter_expression "SQL expression"
     * @param params           Vector containing SQL expression parameters
     * @throw                   dds::core::Exception
     */
    TFilter(const std::string& query_expression,
            const std::vector<std::string>& params);

    /**
     * Get the query expression.
     *
     * @return std::string The @ref anchor_dds_topic_filter_expression "SQL expression".
     * @throw              dds::core::Exception
     */
    const std::string& expression() const;

    /**
     * Provides the begin iterator to the SQL expression parameter list.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @return dds::topic::Filter::const_iterator
     *              The begin iterator
     * @throw  dds::core::Exception
     */
    const_iterator begin() const;

    /**
     * The end iterator to the SQL expression parameter list.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @return dds::topic::Filter::const_iterator
     *              The end iterator
     * @throw  dds::core::Exception
     */
    const_iterator end() const;

    /**
     * Provides the begin iterator to the SQL expression parameter list.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @return dds::topic::Filter::iterator
     *              The begin iterator
     * @throw  dds::core::Exception
     */
    iterator begin();

    /**
     * The end iterator to the SQL expression parameter list.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @return dds::topic::Filter::iterator
     *              The end iterator
     * @throw  dds::core::Exception
     */
    iterator end();

    /**
     * Sets the query parameters.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @param begin Iterator pointing to the beginning of the parameters to set
     * @param end   Iterator pointing to the end of the parameters to set
     * @throw        dds::core::Exception
     */
    template <typename FWIterator>
    void parameters(const FWIterator& begin, const FWIterator end);

    /**
     * Adds a parameter to the query.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @param param The parameter to add
     * @throw       dds::core::Exception
     */
    void add_parameter(const std::string& param);

    /**
     * Gets the number of parameters in the query of the filter.
     *
     * See @ref anchor_dds_topic_filter_expression "SQL expression info"
     *
     * @return uint32_t The number of parameters in the query
     * @throw  dds::core::Exception
     */
    uint32_t parameters_length() const;
};

#endif /* DDS_TOPIC_TFILTER__HPP_ */
