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
#ifndef OMG_DDS_CORE_XTYPES_DETAIL_ANNOTATIONS_HPP_
#define OMG_DDS_CORE_XTYPES_DETAIL_ANNOTATIONS_HPP_

namespace dds {
  namespace core {
    namespace xtypes {
      namespace detail {
        class Annotation { };

        class IdAnnotation : public  Annotation { };

        class KeyAnnotation : public  Annotation { };

        class SharedAnnotation : public  Annotation  { };

        class NestedAnnotation : public  Annotation  { };

        class ExtensibilityAnnotation : public  Annotation  { };

        class MustUnderstandAnnotation : public  Annotation { };

        class VerbatimAnnotation : public  Annotation { };

        class BitsetAnnotation : public  Annotation { };

        class BitBoundAnnotation : public  Annotation  { };
      }
    }
  }
}
#endif /* OMG_DDS_CORE_XTYPES_DETAIL_ANNOTATIONS_HPP_ */
