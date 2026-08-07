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
#ifndef CYCLONEDDS_DDS_CORE_DETAIL_MACROS_HPP_
#define CYCLONEDDS_DDS_CORE_DETAIL_MACROS_HPP_

/**
 * @file
 */

// Implementation

#include <iostream>
#include <string.h>

// == Constants
#define OMG_DDS_DEFAULT_STATE_BIT_COUNT_DETAIL static_cast<size_t>(32)
#define OMG_DDS_DEFAULT_STATUS_COUNT_DETAIL    static_cast<size_t>(32)
// ==========================================================================

#define OSPL_DEFAULT_TO_CXX11
#define OSPL_USE_CXX11

// == Static Assert
#define OMG_DDS_STATIC_ASSERT_DETAIL(condition) static_assert(condition, #condition)
// ==========================================================================

// DLL Export Macros
#include "dds/core/detail/export.hpp"

// ==========================================================================

// Logging Macros
#if 0
#include <dds/core/detail/maplog.hpp>
#define OMG_DDS_LOG_DETAIL(kind, msg) \
    if (dds::core::detail::maplog(kind) >= os_reportVerbosity) os_report(dds::core::detail::maplog(kind),"isocpp-OMG_DDS_LOG",__FILE__,__LINE__,0,msg)
//  std::cout << "[" << kind << "]: " << msg << std::endl;
// ==========================================================================
#endif

// C++ 11 features
// Slightly pathological - we could (for example) want to use boost traits
// and 'STL' implementations  but compile with -std=c++11, so set a macro for
// C++11 compile being on. This way we can always use language features
#define OSPL_DDS_CXX11
#include <cstring>

#if defined (_MSC_VER) && ( _MSC_VER < 1700)
// http://msdn.microsoft.com/en-us/library/vstudio/hh567368.aspx
// 'Visual C++ in Visual Studio 2010 ... "final" was ... supported, but
// under the different spelling "sealed". The Standard spelling and
// semantics of "override" and "final" are now completely supported.'
#  define OSPL_DDS_FINAL sealed
#else
#  define OSPL_DDS_FINAL final
#endif

#if defined (_MSC_VER) && (_MSC_VER <= 1800)
// See: http://msdn.microsoft.com/en-us/library/vstudio/hh567368.aspx
// "These are now supported, but with this exception: For defaulted functions,
// the use of = default to request member-wise move constructors and move
// assignment operators is not supported."
// ('now' is currently VS 2013 - _MSC_VER == 1800).
#  define OSPL_CXX11_NO_FUNCTION_DEFAULTS
#endif

#define OSPL_ENUM enum class
#define OSPL_ENUM_LABEL(_escope,_etype,_elabel) _escope::_etype::_elabel

/**
 * @brief Macro to disable unused argument warnings
 */
#define DDSCXX_UNUSED_ARG(a) (void)(a)

/**
 * @brief Macro to disable specific compiler warnings
 */
#if defined(_MSC_VER)
#define DDSCXX_WARNING_MSVC_OFF(x) \
  __pragma(warning(push)) \
  __pragma(warning(disable: ## x))
#define DDSCXX_WARNING_MSVC_ON(x) \
  __pragma(warning(pop))
#else
#define DDSCXX_WARNING_MSVC_OFF(x)
#define DDSCXX_WARNING_MSVC_ON(x)
#endif

// End of implementation

#endif /* CYCLONEDDS_DDS_CORE_DETAIL_MACROS_HPP_ */
