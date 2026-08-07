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


/**
 * @file
 *
 * Unfortunately, we need this because we can not change the Duration.hpp and Time.hpp
 * and we don't want this piece of code repeated in every function within their
 * implementation files.
 */

#ifndef CYCLONEDDS_CORE_TIME_HELPER_HPP_
#define CYCLONEDDS_CORE_TIME_HELPER_HPP_

#include <sstream>
#include <dds/dds.h>
#include <org/eclipse/cyclonedds/core/ReportUtils.hpp>

#define MS 1000
#define MiS 1000000
#define NS 1000000000

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace core
{
namespace timehelper
{


/**
* Check if the TIMEISH value supplied is at all suitable for arithmetic jiggery
* pokery. Invalidity encompasses (but is not restricted to) a -1 seconds part
* an 'infinity' Duration, an 'invalid' Time (or Duration).
* @param t the TIMEISH thing to check
* @return true if the argument is not suitable for doing sums with.
*/
template <typename TIMEISH>
bool is_valid_for_arithmetic(const TIMEISH& t)
{
    return (t.sec() != -1 // Invalid
            && t.sec() != 0x7FFFFFFF // Infinity
            && t.nanosec() < 1000000000);  // Invalid & infinity are > 10^9
}

/**
* Check a TIMEISH is valid for doing sums with.
* @param t the TIMEISH thing to check
* @param context Some sort of clue to the receiver about what was
* called or what you were trying to do. Must be a literal or c_str.
* Defaults to "".
* @param function String to be concateneated onto context.
* Must be a literal or c_str. Defaults to "".
* @throws a dds::core::InvalidDataError if not valid.
* @see OSPL_CONTEXT_LITERAL
* @see is_valid_for_arithmetic
*/
template <typename TIMEISH>
void validate(const TIMEISH& t, const char* context = "timehelper", const char* function = "validate")
{
    if(! is_valid_for_arithmetic<TIMEISH>(t))
    {
        std::stringstream message("dds::core::InvalidDataError");
        message << "Value invalid for arithmetic operations" << context << function
                << " seconds=" << t.sec() << " (" << std::hex << t.sec()
                << ") nanoseconds=" << t.nanosec() << " (" << std::hex << t.nanosec() << ")";
        ISOCPP_THROW_EXCEPTION(ISOCPP_ILLEGAL_OPERATION_ERROR, message.str().c_str());
    }
}

}
}
}
}
}

#endif /* CYCLONEDDS_CORE_TIME_HELPER_HPP_ */
