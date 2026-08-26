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
#ifndef OMG_DDS_CORE_XTYPES_DETAIL_PRIMITIVE_TYPES_HPP_
#define OMG_DDS_CORE_XTYPES_DETAIL_PRIMITIVE_TYPES_HPP_

#include <dds/core/xtypes/TypeKind.hpp>

// This template specialization have the intent to provide guidelines
// implementors of this specification. Notice that this define only a subset
// of primitive types, thus more work for you to do...
namespace dds {
  namespace core {
    namespace xtypes {
      // Notice that const char* const is used instead of std::string
      // to limit initialization issues with ctors.
      template<>
      struct dynamic_type_traits<uint8_t> {
        static const TypeKind TYPE_ID = TypeKind::UINT_8_TYPE;
        static const  char* const NAME; // "uint8_t"
      };

      template<>
      struct dynamic_type_traits<int16_t> {
        static const TypeKind TYPE_ID = TypeKind::INT_16_TYPE;
        static const  char* const NAME; // "uint16_t"
      };


      template<>
      struct dynamic_type_traits<uint16_t> {
        static const TypeKind TYPE_ID = TypeKind::UINT_16_TYPE;
        static const  char* const NAME; // "int16_t"
      };

      template<>
      struct dynamic_type_traits<int32_t> {
        static const TypeKind TYPE_ID = TypeKind::INT_32_TYPE;
        static const  char* const NAME; // "int16_t"
      };

      template<>
      struct dynamic_type_traits<uint32_t> {
        static const TypeKind TYPE_ID = TypeKind::UINT_32_TYPE;
        static const  char* const NAME; // "uint32_t"
      };

    }
  }
}


#endif /* OMG_DDS_CORE_XTYPES_DETAIL_PRIMITIVE_TYPES_HPP_ */
