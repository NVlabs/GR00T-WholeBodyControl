/*
 * Copyright(c) 2006 to 2022 ZettaScale Technology and others
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
  */

#ifndef CYCLONEDDS_CORE_MISSING_HPP_
#define CYCLONEDDS_CORE_MISSING_HPP_

//since these helpers are only introduced from c++14 onwards

#if __cplusplus < 201402L
namespace std {
  template< bool B, class T, class F >
  using conditional_t = typename std::conditional<B, T, F>::type;

  template<bool cond, class T = void>
  using enable_if_t = typename std::enable_if<cond, T>::type;

  template<class T = void>
  using decay_t = typename std::decay<T>::type;
}
#endif /* _cplusplus */

#endif /* CYCLONEDDS_CORE_MISSING_HPP_ */
