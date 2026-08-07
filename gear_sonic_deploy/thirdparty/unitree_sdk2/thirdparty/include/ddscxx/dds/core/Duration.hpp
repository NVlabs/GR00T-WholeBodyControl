#ifndef OMG_DDS_CORE_DURATION_HPP_
#define OMG_DDS_CORE_DURATION_HPP_

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

/**
 * @file
 * This class represents a time interval.
 */

/**
 * Duration represents a time interval and can -
 * * Can be incremented by durations expressed as seconds, nanoseconds
 *   milliseconds, or Duration objects.
 *
 * * Can be converted to and from Durations expressed in
 *   milliseconds (or other units) as integer types.
 */
class OMG_DDS_API Duration
{
public:
    /**
     * Create a Duration elapsing zero seconds.
     */
    static const Duration zero();       // {0, 0}

    /**
     * Create an infinite Duration.
     */
    static const Duration infinite();   // {0x7fffffff, 0x7fffffff}

public:
    /**
     * Create a Duration elapsing the default amount of time (zero seconds).
     */
    Duration();

    /**
     * Create a Duration elapsing a specific amount of time.
     *
     * @param sec     Amount of seconds for the Duration.
     * @param nanosec Amount of nanoseconds for the Duration.
     */
    explicit Duration(int32_t sec, uint32_t nanosec = 0);
#if __cplusplus >= 199711L
    /** @copydoc dds::core::Duration::Duration(int32_t sec, uint32_t nanosec = 0) */
    explicit Duration(int64_t sec, uint32_t nanosec = 0);
#endif

public:
    /**
     * Create a Duration from a number of microseconds
     * @param microseconds number of microseconds
     */
    static const Duration from_microsecs(int64_t microseconds);

    /**
     * Create a Duration from a number of milliseconds
     * @param milliseconds number of milliseconds
     */
    static const Duration from_millisecs(int64_t milliseconds);

    /**
     * Create a Duration from a number of seconds
     * @param seconds number of seconds
     */
    static const Duration from_secs(double seconds);

public:
    /**
     * Get seconds part of the Duration.
     * @return number of seconds
     */
    int64_t sec() const;
    /**
     * Set number of seconds
     * @param s number of seconds
     */
    void sec(int64_t s);

    /**
     * Get nanoseconds part of the Duration.
     * @return number of nanoseconds
     */
    uint32_t nanosec() const;
    /**
     * Set number of nanoseconds
     * @param ns number of nanoseconds
     */
    void nanosec(uint32_t ns);

public:
    /**
     * Returns an integer value for a comparison of two Durations:
     * 1 if this Duration is greater than the comparator (that)
     * -1 if the Duration is less than the comparator (that)
     * 0 if the Duration matches the comparator (that)
     *
     * @param that Duration to compare
     *
     * @return comparison result
     */
    int compare(const Duration& that) const;

    /**
     * Returns true if the Duration is greater than the comparator
     *
     * @param that Duration to compare
     * @return comparison result
     */
    bool operator >(const Duration& that) const;

    /**
     * Returns true if the Duration is greater than or equal to the comparator
     * @param that the Duration to compare
     * @return true if that is larger than or equal to this
     */
    bool operator >=(const Duration& that) const;

    /**
     * Returns true if the Duration is not equal to the comparator
     *
     * @param that Duration to compare
     * @return comparison result
     */
    bool operator !=(const Duration& that) const;

    /**
     * Returns true if the Duration is equal to the comparator
     *
     * @param that Duration to compare
     * @return comparison result
     */
    bool operator ==(const Duration& that) const;

    /**
     * Returns true if the Duration is less than or equal to the comparator
     *
     * @param that Duration to compare
     * @return comparison result
     */
    bool operator <=(const Duration& that) const;

    /**
     * Returns true if the Duration is less than the comparator
     *
     * @param that Duration to compare
     * @return comparison result
     */
    bool operator <(const Duration& that) const;

public:
    /**
     * Add a Duration to this Duration
     *
     * @param a_ti Duration to add
     * @return this Duration + a_ti
     */
    Duration& operator+=(const Duration& a_ti);

    /**
     * Subtract a Duration from this Duration
     *
     * @param a_ti Duration to subtract
     * @return this Duration - a_ti
     */
    Duration& operator-=(const Duration& a_ti);

    /**
     * Multiply this Duration by a factor
     *
     * @param factor the factor to multiply this Duration by
     * @return this Duration * factor
     */
    Duration& operator*=(uint64_t factor);

    /**
     * Add a Duration to Duration
     *
     * @param other a Duration
     * @return Duration + other
     */
    const Duration operator +(const Duration& other) const;

    /**
     * Subtract a Duration from Duration
     *
     * @param other a Duration
     * @return the Duration - other
     */
    const Duration operator -(const Duration& other) const;
public:
    /**
     * Returns this Duration in milliseconds.
     *
     * @return the duration in milliseconds
     */
    int64_t to_millisecs() const;

    /**
     * Returns this Duration in micro-seconds.
     *
     * @return the duration in micro-seconds
     */
    int64_t to_microsecs() const;

    /**
     * Returns this Duration in seconds.
     *
     * @return the duration in seconds
     */
    double to_secs() const;

private:
    int32_t sec_;
    uint32_t nsec_;
};

/**
 * Multiply Duration by a factor
 *
 * @param lhs factor by which to multiply
 * @param rhs Duration to multiply
 *
 * @return factor * Duration
 */
const Duration OMG_DDS_API operator *(uint64_t lhs,
                                      const Duration& rhs);

/**
 * Multiply Duration by a factor
 *
 * @param lhs Duration to multiply
 * @param rhs factor by which to multiply
 *
 * @return Duration * factor
 */
const Duration OMG_DDS_API operator *(const Duration& lhs,
                                      uint64_t rhs);

/**
 * Divide Duration by a factor
 *
 * @param lhs Duration to divide
 * @param rhs factor by which to divide
 */
const Duration OMG_DDS_API operator /(const Duration& lhs,
                                      uint64_t rhs);

}
} /* namespace dds / namespace core  */
#endif /* OMG_DDS_CORE_DURATION_HPP_ */
