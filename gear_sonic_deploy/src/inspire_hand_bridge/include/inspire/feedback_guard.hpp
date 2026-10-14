#pragma once

#include <cmath>
#include <cstddef>
#include <optional>

#include "inspire/contract.hpp"

namespace inspire {

struct FeedbackViolation {
  bool left{};
  std::size_t index{};
  float delta{};
  float limit{};
};

inline std::optional<FeedbackViolation> feedback_violation(
    const HandVector& left_target,
    const HandVector& right_target,
    const HandVector& left_measured,
    const HandVector& right_measured,
    const HandVector& tolerance) {
  for (std::size_t index = 0; index < kHandDof; ++index) {
    const float delta = std::abs(left_target[index] - left_measured[index]);
    if (delta > tolerance[index]) {
      return FeedbackViolation{true, index, delta, tolerance[index]};
    }
  }
  for (std::size_t index = 0; index < kHandDof; ++index) {
    const float delta = std::abs(right_target[index] - right_measured[index]);
    if (delta > tolerance[index]) {
      return FeedbackViolation{false, index, delta, tolerance[index]};
    }
  }
  return std::nullopt;
}

}  // namespace inspire
