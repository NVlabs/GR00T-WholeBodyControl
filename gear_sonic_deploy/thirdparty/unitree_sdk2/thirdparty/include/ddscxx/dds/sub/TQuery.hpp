#ifndef DDS_CORE_TQUERY_HPP_
#define DDS_CORE_TQUERY_HPP_

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
#include <dds/core/Reference.hpp>
#include <dds/sub/AnyDataReader.hpp>

namespace dds
{
namespace sub
{
template <typename DELEGATE>
class TQuery;
}
}

/**
 * @brief
 * Query objects contain expressions that allow the application to specify
 * a filter on the locally available data.
 *
 * A Query is used in a QueryCondition or DataReader::Selector and is
 * associated with one DataReader.
 *
 * @anchor anchor_dds_sub_query_expression
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
 * @see dds::sub::cond::QueryCondition
 * @see dds::sub::DataReader::Selector
 * @see dds::sub::DataReader::ManipulatorSelector
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "Subscription concept"
 */
template <typename DELEGATE>
class dds::sub::TQuery: public virtual dds::core::Reference<DELEGATE> {
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TQuery, dds::core::Reference, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TQuery)
    OMG_DDS_COMPLETE_RULE_OF_FIVE_VIRTUAL_EXPLICIT(TQuery)

public:
  /**
   * Iterator for the query expression parameters.
   */
  typedef typename DELEGATE::iterator iterator;

  /**
   * Iterator for the query expression parameters.
   */
  typedef typename DELEGATE::const_iterator const_iterator;

public:
  /**
   * Create a dds::sub::Query associated with an dds::sub::AnyDataReader.
   *
   * @param  dr The AnyDataReader to associate with the Query.
   * @param  expression an @ref anchor_dds_sub_query_expression "SQL expression"
   * @throw  dds::core::Exception
   */
  TQuery(const dds::sub::AnyDataReader& dr, const std::string& expression);

  /**
   * Create a dds::sub::Query associated with an dds::sub::AnyDataReader.
   *
   * @param  dr The AnyDataReader to associate with the Query.
   * @param  expression an @ref anchor_dds_sub_query_expression "SQL expression"
   * @param params_begin Iterator pointing to the beginning of the parameters to set
   * @param params_end   Iterator pointing to the end of the parameters to set
   * @throw  dds::core::Exception
   */
  template<typename FWIterator>
  TQuery(const dds::sub::AnyDataReader& dr, const std::string& expression,
         const FWIterator& params_begin, const FWIterator& params_end);

  /**
   * Create a dds::sub::Query associated with an dds::sub::AnyDataReader.
   *
   * @param  dr The AnyDataReader to associate with the Query.
   * @param  expression an @ref anchor_dds_sub_query_expression "SQL expression"
   * @param params Vector containing SQL expression parameters
   * @throw  dds::core::Exception
   */
  TQuery(const dds::sub::AnyDataReader& dr, const std::string& expression,
         const std::vector<std::string>& params);

  /**
   * Get expression.
   *
   * @return std::string The @ref anchor_dds_sub_query_expression "SQL expression".
   * @throw  dds::core::Exception
   */
  const std::string& expression() const;

  /**
   * Set new expression.
   *
   * @param  expr an ref anchor_dds_sub_query_expression "SQL expression"
   * @throw  dds::core::Exception
   */
  void expression(const std::string& expr);

  /**
   * Provides the begin iterator to the SQL expression parameter list.
   *
   * See @ref anchor_dds_sub_query_expression "SQL expression info"
   *
   * @return dds::sub::Query::const_iterator
   *              The begin iterator
   * @throw  dds::core::Exception
   */
  const_iterator begin() const;

  /**
   * The end iterator to the SQL expression parameter list.
   *
   * See @ref anchor_dds_sub_query_expression "SQL expression info"
   *
   * @return dds::sub::Query::const_iterator
   *              The end iterator
   * @throw  dds::core::Exception
   */
  const_iterator end() const;

  /**
   * Provides the begin iterator to the SQL expression parameter list.
   *
   * See @ref anchor_dds_sub_query_expression "SQL expression info"
   *
   * @return dds::sub::Query::iterator
   *              The begin iterator
   * @throw  dds::core::Exception
   */
  iterator begin();

  /**
   * The end iterator to the SQL expression parameter list.
   *
   * See @ref anchor_dds_sub_query_expression "SQL expression info"
   *
   * @return dds::sub::TQuery::iterator
   *              The end iterator
   * @throw  dds::core::Exception
   */
  iterator end();

  /**
   * Sets the query parameters.
   *
   * See @ref anchor_dds_sub_query_expression "SQL expression info"
   *
   * @param begin Iterator pointing to the beginning of the parameters to set
   * @param end   Iterator pointing to the end of the parameters to set
   * @throw  dds::core::Exception
   */
  template<typename FWIterator>
  void parameters(const FWIterator& begin, const FWIterator end);

  /**
   * Adds a parameter to the query.
   *
   * See @ref anchor_dds_sub_query_expression "SQL expression info"
   *
   * @param param The parameter to add
   * @throw  dds::core::Exception
   */
  void add_parameter(const std::string& param);

  /**
   * Gets the number of parameters in the query.
   *
   * See @ref anchor_dds_sub_query_expression "SQL expression info"
   *
   * @return uint32_t The number of parameters in the query
   * @throw  dds::core::Exception
   */
  uint32_t parameters_length() const;

  /**
   * This operation returns the DataReader associated with the Query.
   *
   * Note that there is exactly one DataReader associated with each Query.
   *
   * @return dds::sub::AnyDataReader The associated DataReader
   * @throw  dds::core::Exception
   */
  const AnyDataReader& data_reader() const;
};

#endif /* DDS_CORE_TQUERY_HPP_ */
