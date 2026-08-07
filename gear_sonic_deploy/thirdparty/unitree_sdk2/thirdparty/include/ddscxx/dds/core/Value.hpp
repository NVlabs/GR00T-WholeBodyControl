#ifndef OMG_DDS_CORE_VALUE_HPP_
#define OMG_DDS_CORE_VALUE_HPP_

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


namespace dds
{
namespace core
{
template <typename D>
class Value;
}
}

/**
 * @brief
 * This class is the base for various value-type dds objects.
 *
 * QoS, Policy, Statuses, and Topic samples are all modeled as value-types.
 *
 * All objects that have a value-type have a deep-copy assignment and copy
 * construction semantics.
 * It should also be pointed out that value-types are not 'pure-value-types' in
 * the sense that they are immutable (as in functional programming languages).
 *
 * The DDS-PSM-Cxx makes value-types mutable to limit the number of copies as well
 * as to limit the time-overhead necessary to change a value-type
 * (note that for immutable value-types the only form of change is to create a new
 * value-type).
 */
template <typename D>
class dds::core::Value
{
protected:
    Value();
    Value(const Value& p) = default;  //check on this != &p?
    Value(Value&& p) = default;  //check on this != &p?

public:
    /** @cond
     * The following Constructors are not really relevant for the API.
     * So, leave them from the doxygen generated API documentation for clarity.
     */

    /**
     * Create a value-type object of one internal value
     *
     * @param arg VALUETYPE value
     */
    template <typename ARG>
    Value(const ARG& arg);

    /**
     * Create a value-type object of two internal value
     *
     * @param arg1 VALUETYPES value
     * @param arg2 VALUETYPES value
     */
    template <typename ARG1, typename ARG2>
    Value(const ARG1& arg1, const ARG2& arg2);

    /**
     * Create a value-type object of three internal value
     *
     * @param arg1 VALUETYPES value
     * @param arg2 VALUETYPES value
     * @param arg3 VALUETYPES value
     */
    template <typename ARG1, typename ARG2, typename ARG3>
    Value(const ARG1& arg1, const ARG2& arg2, const ARG3& arg3);

    /**
     * Create a value-type object of four internal value
     *
     * @param arg1 VALUETYPES value
     * @param arg2 VALUETYPES value
     * @param arg3 VALUETYPES value
     * @param arg4 VALUETYPES value
     */
    template <typename ARG1, typename ARG2, typename ARG3, typename ARG4>
    Value(const ARG1& arg1, const ARG2& arg2, const ARG3& arg3, const ARG4& arg4);

    /**
     * Create a value-type object of five internal value
     *
     * @param arg1 VALUETYPES value
     * @param arg2 VALUETYPES value
     * @param arg3 VALUETYPES value
     * @param arg4 VALUETYPES value
     * @param arg5 VALUETYPES value
     */
    template <typename ARG1, typename ARG2, typename ARG3, typename ARG4, typename ARG5>
    Value(const ARG1& arg1, const ARG2& arg2, const ARG3& arg3, const ARG4& arg4, const ARG5& arg5);

    /**
     * Create a value-type object of six internal value
     *
     * @param arg1 VALUETYPES value
     * @param arg2 VALUETYPES value
     * @param arg3 VALUETYPES value
     * @param arg4 VALUETYPES value
     * @param arg5 VALUETYPES value
     * @param arg6 VALUETYPES value
     */
    template <typename ARG1, typename ARG2, typename ARG3, typename ARG4, typename ARG5, typename ARG6>
    Value(const ARG1& arg1, const ARG2& arg2, const ARG3& arg3, const ARG4& arg4, const ARG5& arg5, const ARG6& arg6);
    /** @endcond */

public:
    /** @cond */
    ~Value() = default;
    /** @endcond */

public:
    /**
     * Assigns new delegate to this Value
     * @param other Value
     */
    Value& operator=(const Value& other);

    /**
     * Moves delegate to this Value
     * @param other Value
     */
    Value& operator=(Value&& other);

    /**
     * Compare this Value with another Value
     *
     * @param other Value
     * @return true if equal
     */
    bool operator==(const Value& other) const;

    /**
     * Compare this Value with another Value
     *
     * @param other Value
     * @return true if not equal
     */
    bool operator !=(const Value& other) const;

public:
    /**
     * The operator->() is provided to be able to directly invoke
     * functions on the delegate.
     *
     * The decision to provide direct access to
     * the delegate was motivated by the need for providing a way that
     * was not invasive with respect to the CXXDDS API and yet would allow
     * for vendor-specific extension.
     * Thus vendor-specific extensions can be invoked on the Value
     * and on all its subclasses as follows:
     * @code{.cpp}
     * my_dds_value.standard_function();
     * my_dds_value->vendor_specific_extension();
     * @endcode
     *
     * @return a reference to delegate.
     */
    D* operator->();

    /** @copydoc dds::core::Value::operator->() */
    const D* operator->() const;

    /** @cond
     * Functions possibly needed for delegate implementation, but not recommended
     * for application usage: exclude from the API documentation for clarity.
     *
     * Returns an object to the underlying delegate.
     */
    const D& delegate() const;
    D& delegate();
    operator D& ();
    operator const D& () const;
    /** @endcond */

protected:
    D d_;
};


#endif /* OMG_DDS_CORE_VALUE_HPP_ */
