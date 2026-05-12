#ifndef OMG_DDS_CORE_TIME_HPP_
#define OMG_DDS_CORE_TIME_HPP_

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

namespace dds
{
namespace core
{
class Duration;
class Time;
}
}

/**
* Time represents a time value and can:
*
* * Be incremented by Duration expressed as seconds,
*   nanoseconds, milliseconds, or Duration objects.
*
* * Be converted to and from Times expressed in
*   milliseconds (or other units) as integer types.
*/
class OMG_DDS_API dds::core::Time
{
public:
    static const Time invalid();       // {-1, 0xffffffff}

public:
    /**
     * Create a Time from a number of microseconds
     *
     * @param microseconds number of microseconds
     */
    static const Time from_microsecs(int64_t microseconds);
    /**
     * Create a Time from a number of milliseconds
     *
     * @param milliseconds number of miliseconds
     */
    static const Time from_millisecs(int64_t milliseconds);
    /**
     * Create a Time from a number of seconds
     *
     * @param seconds number of seconds
     */
    static const Time from_secs(double seconds);

public:
    /**
     * Create a Time of zero seconds.
     */
    Time();
    /**
     * Create a Time elapsing a specific amount of time.
     */
    explicit Time(int64_t sec, uint32_t nanosec = 0);

public:
    /**
     * @return number of seconds
     */
    int64_t sec() const;

    /**
     * Set number of seconds
     * @param s number of seconds
     */
    void    sec(int64_t s);

    /**
     * @return number of nanoseconds
     */
    uint32_t nanosec() const;

    /**
     * Set number of nanoseconds
     * @param ns number of nanoseconds
     */
    void    nanosec(uint32_t ns);

public:
    /**
     * Returns an integer indicating the result of a comparison
      * of two Times:
     *   1 if this Time is greater than the comparator (that)
     *  -1 if the Time is less than the comparator (that)
     *   0 if the Time matches the comparator (that)
     *
     * @param that Time to compare
     * @return comparison result
     */
    int compare(const Time& that) const;

    /**
     * @param that Time to compare
     * @return true if the Time is greater than the comparator
     */
    bool operator >(const Time& that) const;

    /**
     * @param that Time to compare
     * @return true if the Time is greater than or equal to the comparator
     */
    bool operator >=(const Time& that) const;

    /**
     * @param that Time to compare
     * @return true if the Time is not equal to the comparator
     */
    bool operator !=(const Time& that) const;

    /**
     * @param that Time to compare
     * @return true if the Time is equal to the comparator
     */
    bool operator ==(const Time& that) const;
    /**
     * @param that Time to compare
     * @return true if the Time is less than or equal to the comparator
     */
    bool operator <=(const Time& that) const;
    /**
     * @param that Time to compare
     * @return true if the Time is less than the comparator
     */
    bool operator <(const Time& that) const;

public:
    /**
     * @param a_ti Duration to add
     * @return Time value + Duration
     */
    Time& operator+=(const Duration& a_ti);
    /**
     * @param a_ti Duration to subtract
     * @return Time value - Duration
     */
    Time& operator-=(const Duration& a_ti);

public:
    /**
     * Returns this Time in milliseconds.
     *
     * @return this Time in milliseconds
     */
    int64_t to_millisecs() const;

    /**
     * Returns this Time in micro-seconds.
     *
     * @return this Time in micro-seconds
     */
    int64_t to_microsecs() const;

    /**
     * Returns this Time in seconds.
     *
     * @return this Time in seconds
     */
    double to_secs() const;

private:
    int64_t sec_;
    uint32_t nsec_;
};

// Time arithmetic operators.
/**
 * Add a Duration to a Time value
 * @param lhs Time
 * @param rhs Duration
 * @return Time * Duration
 */
const dds::core::Time OMG_DDS_API operator +(const dds::core::Time& lhs,      const dds::core::Duration& rhs);

/**
 * Add a Duration to a Time value
 * @param lhs Duration
 * @param rhs Time
 * @return Duration + Time
 */
const dds::core::Time OMG_DDS_API operator +(const dds::core::Duration& lhs,  const dds::core::Time& rhs);

/**
 * Subtract a Duration from a Time value
 * @param lhs Time
 * @param rhs Duration
 * @return Time - Duration
 */
const dds::core::Time OMG_DDS_API operator -(const dds::core::Time& lhs,      const dds::core::Duration& rhs);


#endif /* OMG_DDS_CORE_TIME_HPP_ */
