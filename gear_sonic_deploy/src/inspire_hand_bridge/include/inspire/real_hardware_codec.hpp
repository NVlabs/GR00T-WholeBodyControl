#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "inspire/contract.hpp"

namespace inspire::real {

inline constexpr float kNormalizedLower = 0.F;
inline constexpr float kNormalizedUpper = 1.F;
// Serial positions encode open fraction: 0 = closed, 1 = open.
// Reuse the canonical coordinate bounds instead of maintaining a second table.
inline constexpr HandVector kCanonicalHardwareOpen = kLowerLimits;
inline constexpr HandVector kCanonicalHardwareClosed = kUpperLimits;

inline void validate_normalized(const HandVector& values, const std::string& field) {
  for (std::size_t index = 0; index < kHandDof; ++index) {
    if (!std::isfinite(values[index])) {
      throw std::runtime_error(field + " contains a non-finite value");
    }
    if (values[index] < kNormalizedLower || values[index] > kNormalizedUpper) {
      throw std::runtime_error(field + " exceeds the official normalized range at index " +
                               std::to_string(index));
    }
  }
}

inline HandVector normalized_to_canonical(const HandVector& normalized) {
  validate_normalized(normalized, "normalized hand state");
  HandVector canonical{};
  for (std::size_t index = 0; index < kHandDof; ++index) {
    const float range = kCanonicalHardwareClosed[index] - kCanonicalHardwareOpen[index];
    canonical[index] =
        kCanonicalHardwareOpen[index] + (1.F - normalized[index]) * range;
  }
  return canonical;
}

inline HandVector canonical_to_normalized(const HandVector& canonical) {
  HandVector normalized{};
  for (std::size_t index = 0; index < kHandDof; ++index) {
    const float low = std::min(kCanonicalHardwareOpen[index], kCanonicalHardwareClosed[index]);
    const float high = std::max(kCanonicalHardwareOpen[index], kCanonicalHardwareClosed[index]);
    if (!std::isfinite(canonical[index]) || canonical[index] < low || canonical[index] > high) {
      throw std::runtime_error("canonical hand command exceeds its real hardware range at index " +
                               std::to_string(index));
    }
    const float range = kCanonicalHardwareClosed[index] - kCanonicalHardwareOpen[index];
    normalized[index] = 1.F - (canonical[index] - kCanonicalHardwareOpen[index]) / range;
  }
  return normalized;
}

inline HandVector normalized_velocity_to_canonical(const HandVector& normalized_velocity) {
  HandVector canonical_velocity{};
  for (std::size_t index = 0; index < kHandDof; ++index) {
    if (!std::isfinite(normalized_velocity[index])) {
      throw std::runtime_error("normalized hand velocity contains a non-finite value");
    }
    canonical_velocity[index] = -normalized_velocity[index] *
                                (kCanonicalHardwareClosed[index] -
                                 kCanonicalHardwareOpen[index]);
  }
  return canonical_velocity;
}

inline bool within_canonical_tolerance(const HandVector& target,
                                       const HandVector& measured,
                                       float tolerance) {
  if (!std::isfinite(tolerance) || tolerance < 0.F) {
    throw std::runtime_error("canonical hold tolerance must be finite and non-negative");
  }
  if (!finite(target) || !finite(measured)) return false;
  for (std::size_t index = 0; index < kHandDof; ++index) {
    if (std::abs(target[index] - measured[index]) > tolerance) return false;
  }
  return true;
}

}  // namespace inspire::real
