#ifndef OMG_DDS_SUB_TQUERY_CONDITION_HPP_
#define OMG_DDS_SUB_TQUERY_CONDITION_HPP_

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

#include <dds/core/detail/conformance.hpp>
#include <dds/sub/cond/detail/QueryCondition.hpp>
#include <dds/sub/cond/TReadCondition.hpp>


#ifdef OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT

namespace dds
{
namespace sub
{
namespace cond
{
template <typename DELEGATE>
class TQueryCondition;
}
}
}

/**
 * @brief
 * QueryCondition objects are specialized ReadCondition objects that allow
 * the application to also specify a filter on the locally available data.
 *
 * The query (query_expression) is similar to an SQL WHERE clause can be
 * parameterized by arguments. See dds::sub::Query for more query information.
 *
 * See the @ref anchor_dds_core_cond_waitset_examples "WaitSet examples" for some examples.<br>
 * Although the WaitSet examples use the StatusCondition, the basic usage of this Condition
 * with a WaitSet is the same.
 *
 * @see dds::sub::Query
 * @see dds::core::cond::Condition
 * @see dds::sub::cond::ReadCondition
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "WaitSet concept"
 * @see for more information: @ref DCPS_Modules_Infrastructure_Waitset "Subscription concept"
 * @see for more information: @ref anchor_dds_sub_query_expression "SQL expression info"
 * @see for more information: @ref anchor_dds_core_cond_waitset_examples "WaitSet examples"
 */
template <typename DELEGATE>
class dds::sub::cond::TQueryCondition :
    public dds::sub::cond::TReadCondition<DELEGATE>,
    public dds::sub::TQuery<DELEGATE>
{
public:
    OMG_DDS_REF_TYPE_PROTECTED_DC(TQueryCondition, dds::sub::cond::TReadCondition, DELEGATE)
    OMG_DDS_IMPLICIT_REF_BASE(TQueryCondition)

public:
    // Random access iterators
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
     * Creates a QueryCondition instance.
     *
     * This will create an QueryCondition that is associated with a dds::sub::Query,
     * which is again associated with a dds::sub::DataReader.
     *
     *
     * @param query   The query to filter on the locally available data.
     * @param status  A mask, which selects only those samples with the desired
     *                sample/view/instance states.
     * @throw dds::core::Exception
     */
    TQueryCondition(const dds::sub::Query& query,
                    const dds::sub::status::DataState& status);

    /**
     * Creates a QueryCondition instance.
     *
     * This will create an QueryCondition that is associated with a dds::sub::Query,
     * which is again associated with a dds::sub::DataReader.
     *
     * The supplied functor will be called when this QueryCondition is triggered
     * and either the inherited dds::core::cond::Condition::dispatch() is called or the
     * dds::core::cond::WaitSet::dispatch() on the WaitSet to which this QueryCondition is
     * attached to.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info" <br>
     * See @ref anchor_dds_sub_cond_readcondition_state_mask "State mask info" in ReadCondition.
     *
     * @param query   The query to filter on the locally available data.
     * @param status  A mask, which selects only those samples with the desired
     *                sample/view/instance states.
     * @param  functor The functor to be called when the QueryCondition triggers.
     * @throw  dds::core::Exception
     */
    template <typename FUN>
    TQueryCondition(const dds::sub::Query& query,
                    const dds::sub::status::DataState& status, FUN functor);

    /**
     * Creates a QueryCondition instance.
     *
     * This will create an QueryCondition that is associated with an dds::sub::AnyDataReader,
     * with parameters that essentially represent a query.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info" <br>
     * See @ref anchor_dds_sub_cond_readcondition_state_mask "State mask info" in ReadCondition.
     *
     * @param dr         Query associated DataReader.
     * @param expression for more information: @ref anchor_dds_sub_query_expression "Query expression"
     * @param params     for more information: @ref anchor_dds_sub_query_expression "Query expression parameters"
     * @param status     A mask, which selects only those samples with the desired
     *                   sample/view/instance states.
     * @throw dds::core::Exception
     */
    TQueryCondition(const dds::sub::AnyDataReader& dr,
                    const std::string& expression,
                    const std::vector<std::string>& params,
                    const dds::sub::status::DataState& status);

     /**
     * Creates a QueryCondition instance.
     *
     * This will create an QueryCondition that is associated with an dds::sub::AnyDataReader,
     * with parameters that essentially represent a query.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info" <br>
     * See @ref anchor_dds_sub_cond_readcondition_state_mask "State mask info" in ReadCondition.
     *
     * @param dr         Query associated DataReader.
     * @param status     A mask, which selects only those samples with the desired
     *                   sample/view/instance states.
     * @throw dds::core::Exception
     */
    TQueryCondition(const dds::sub::AnyDataReader& dr,
                    const dds::sub::status::DataState& status);
    /**
     * Creates a QueryCondition instance.
     *
     * This will create an QueryCondition that is associated with an dds::sub::AnyDataReader,
     * with parameters that essentially represent a query.
     *
     * The supplied functor will be called when this QueryCondition is triggered
     * and either the inherited dds::core::cond::Condition::dispatch() is called or the
     * dds::core::cond::WaitSet::dispatch() on the WaitSet to which this QueryCondition is
     * attached to.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info" <br>
     * See @ref anchor_dds_sub_cond_readcondition_state_mask "State mask info" in ReadCondition.
     *
     * @param dr         Query associated DataReader.
     * @param expression for more information: @ref anchor_dds_sub_query_expression "Query expression"
     * @param params     for more information: @ref anchor_dds_sub_query_expression "Query expression parameters"
     * @param status     A mask, which selects only those samples with the desired
     *                   sample/view/instance states.
     * @throw dds::core::Exception
     */
    template <typename FUN>
    TQueryCondition(const dds::sub::AnyDataReader& dr,
                    const std::string& expression,
                    const std::vector<std::string>& params,
                    const dds::sub::status::DataState& status,
                    FUN functor);

public:
    /**
     * Set the Query expression.
     *
     * @param expr for more information: @ref anchor_dds_sub_query_expression "SQL expression"
     * @throw  dds::core::Exception
     */
    void expression(const std::string& expr);

    /**
     * Get the Query expression.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info"
     *
     * @return std::string The query expression
     * @throw  dds::core::Exception
     */
    const std::string& expression();

    /**
     * Provides the beginning iterator of the query parameter list.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info"
     *
     * @return dds::sub::cond::TQueryCondition::const_iterator
     *              The beginning iterator
     * @throw  dds::core::Exception
     */
    const_iterator begin() const;

    /**
     * Provides the end iterator of the query parameter list.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info"
     *
     * @return dds::sub::cond::TQueryCondition::const_iterator
     *              The end iterator
     * @throw  dds::core::Exception
     */
    const_iterator end() const;

    /**
     * Provides the beginning iterator of the query parameter list.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info"
     *
     * @return dds::sub::cond::TQueryCondition::iterator
     *              The beginning iterator
     * @throw  dds::core::Exception
     */
    iterator begin();

    /**
     * Provides the end iterator of the query parameter list.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info"
     *
     * @return dds::sub::cond::TQueryCondition::iterator
     *              The end iterator
     * @throw  dds::core::Exception
     */
    iterator end();

    /**
     * Sets the query parameters.
     *
     * See @ref anchor_dds_sub_query_expression "SQL expression info"
     *
     * @param  begin Iterator pointing to the beginning of the parameters to set
     * @param  end   Iterator pointing to the end of the parameters to set
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
     * This operation returns the DataReader associated with the QueryCondition.
     *
     * Note that there is exactly one DataReader associated with each QueryCondition.
     *
     * @cond
     * Note also that since QueryCondition overrides this call since it inherits
     * two different implementations from its two different parents. Not offering
     * an implementation here would results in an ambiguity.
     * @endcond
     *
     * @return dds::sub::AnyDataReader The associated DataReader
     * @throw  dds::core::Exception
     */
    const dds::sub::AnyDataReader& data_reader() const;
};

#endif  // OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT

#endif /* OMG_DDS_SUB_TQUERY_CONDITION_HPP_ */
