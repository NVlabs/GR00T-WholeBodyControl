/*
 * Copyright(c) 2021 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CDR_ENUMS_HPP_
#define CDR_ENUMS_HPP_

namespace org {
namespace eclipse {
namespace cyclonedds {
namespace core {
namespace cdr {

/**
 * @brief
 * Entity extensibility descriptors.
 *
 * @enum extensibility Describes the extensibility of entities.
 *
 * This value is set for entities and their parents.
 *
 * @var extensibility::ext_final The entity representation is complete, no fields can be added or removed.
 * @var extensibility::ext_appendable The entity representation can be extended, no fields can be removed.
 * @var extensibility::ext_mutable The entity representation can be modified, fields can be removed or added.
 */
enum class extensibility {
  ext_final,
  ext_appendable,
  ext_mutable
};

/**
 * @brief
 * Encoding version descriptors.
 *
 * @enum encoding_version Describes the CDR encoding version of entities.
 *
 * @var encoding_version::basic_cdr Basic CDR encoding, does not support any xtypes functionality.
 * @xml encoding_version::xml XML encoding.
 * @var encoding_version::xcdr_v1 Version 1 Xtypes CDR encoding (deprecated).
 * @var encoding_version::xcdr_v2 Version 2 XTypes CDR encoding.
 */
enum class encoding_version {
  basic_cdr,
  xml,
  xcdr_v1,
  xcdr_v2
};

typedef uint32_t allowable_encodings_t;

}
}
}
}
} /* namespace org / eclipse / cyclonedds / core / cdr */
#endif
