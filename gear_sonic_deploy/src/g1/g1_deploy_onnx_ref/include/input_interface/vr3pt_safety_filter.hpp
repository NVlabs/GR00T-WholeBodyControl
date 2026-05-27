/**
 * @file vr3pt_safety_filter.hpp
 * @brief Sanity filter for VR_3PT streams before they reach the encoder.
 */
#ifndef VR3PT_SAFETY_FILTER_HPP
#define VR3PT_SAFETY_FILTER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Vr3PtSafetyFilter {
 public:
  struct Config {
    bool enabled = true;
    double max_position_step_m = 0.03;
    double max_orientation_step_rad = M_PI / 36.0;
    int violation_streak_estop = 10;
  };

  struct Result {
    std::array<double, 9> position{};
    std::array<double, 12> orientation{
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0};
    bool used_last_good = false;
    bool estop_triggered = false;
  };

  Vr3PtSafetyFilter() = default;
  explicit Vr3PtSafetyFilter(Config cfg) : cfg_(cfg) {}

  Result Filter(const std::array<double, 9>& new_pos,
                const std::array<double, 12>& new_orn) {
    Result result;
    if (!AllFinite(new_pos) || !AllFinite(new_orn) || !QuatsValid(new_orn)) {
      return EmitLastGood(result, "NaN/Inf/invalid-quat");
    }

    if (!cfg_.enabled) {
      initialized_ = true;
      violation_streak_ = 0;
      last_pos_ = new_pos;
      last_orn_ = NormalizedOrientation(new_orn);
      result.position = new_pos;
      result.orientation = last_orn_;
      return result;
    }

    if (!initialized_) {
      initialized_ = true;
      violation_streak_ = 0;
      last_pos_ = new_pos;
      last_orn_ = NormalizedOrientation(new_orn);
      result.position = new_pos;
      result.orientation = last_orn_;
      return result;
    }

    const auto normalized_orn = NormalizedOrientation(new_orn);
    const double max_position_step = MaxPositionStep(last_pos_, new_pos);
    const double max_orientation_step = MaxQuatAngle(last_orn_, normalized_orn);
    if (max_position_step > cfg_.max_position_step_m) {
      return EmitLastGood(
          result, "position-step", max_position_step, cfg_.max_position_step_m,
          "m");
    }
    if (max_orientation_step > cfg_.max_orientation_step_rad) {
      return EmitLastGood(
          result, "orientation-step", max_orientation_step,
          cfg_.max_orientation_step_rad, "rad");
    }

    violation_streak_ = 0;
    last_pos_ = new_pos;
    last_orn_ = normalized_orn;
    result.position = new_pos;
    result.orientation = normalized_orn;
    return result;
  }

  void Reset() {
    initialized_ = false;
    violation_streak_ = 0;
    last_pos_.fill(0.0);
    last_orn_ = IdentityOrientation();
  }

  bool estop_requested() const {
    return violation_streak_ >= cfg_.violation_streak_estop;
  }

 private:
  Result EmitLastGood(Result result,
                      const char* reason,
                      double observed = 0.0,
                      double limit = 0.0,
                      const char* unit = "") {
    ++violation_streak_;
    if (violation_streak_ % 5 == 1) {
      std::cerr << "[Vr3PtSafetyFilter] reject: " << reason
                << " observed=" << std::fixed << std::setprecision(4)
                << observed << unit
                << " limit=" << limit << unit
                << " streak=" << violation_streak_ << std::endl;
    }
    result.used_last_good = true;
    result.estop_triggered = estop_requested();
    result.position = initialized_ ? last_pos_ : std::array<double, 9>{};
    result.orientation = initialized_ ? last_orn_ : IdentityOrientation();
    return result;
  }

  template <typename ArrayT>
  static bool AllFinite(const ArrayT& values) {
    for (double value : values) {
      if (!IsFiniteBits(value)) {
        return false;
      }
    }
    return true;
  }

  static bool IsFiniteBits(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "unexpected double size");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    constexpr std::uint64_t kExponentMask = 0x7ff0000000000000ULL;
    return (bits & kExponentMask) != kExponentMask;
  }

  static bool QuatsValid(const std::array<double, 12>& q12) {
    for (int i = 0; i < 3; ++i) {
      const double norm = QuatNorm(q12, i);
      if (norm < 1e-6) {
        return false;
      }
      if (norm < 0.5 || norm > 2.0) {
        return false;
      }
    }
    return true;
  }

  static std::array<double, 12> IdentityOrientation() {
    return {
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0};
  }

  static double QuatNorm(const std::array<double, 12>& q12, int idx) {
    double sum = 0.0;
    for (int j = 0; j < 4; ++j) {
      const double v = q12[idx * 4 + j];
      sum += v * v;
    }
    return std::sqrt(sum);
  }

  static std::array<double, 12> NormalizedOrientation(
      const std::array<double, 12>& q12) {
    std::array<double, 12> normalized = q12;
    for (int i = 0; i < 3; ++i) {
      const double norm = QuatNorm(q12, i);
      for (int j = 0; j < 4; ++j) {
        normalized[i * 4 + j] = q12[i * 4 + j] / norm;
      }
    }
    return normalized;
  }

  static double MaxPositionStep(const std::array<double, 9>& a,
                                const std::array<double, 9>& b) {
    double max_distance = 0.0;
    for (int i = 0; i < 3; ++i) {
      const double dx = a[i * 3 + 0] - b[i * 3 + 0];
      const double dy = a[i * 3 + 1] - b[i * 3 + 1];
      const double dz = a[i * 3 + 2] - b[i * 3 + 2];
      max_distance = std::max(max_distance, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return max_distance;
  }

  static double MaxQuatAngle(const std::array<double, 12>& a,
                             const std::array<double, 12>& b) {
    double max_angle = 0.0;
    for (int i = 0; i < 3; ++i) {
      double dot = 0.0;
      for (int j = 0; j < 4; ++j) {
        dot += a[i * 4 + j] * b[i * 4 + j];
      }
      const double abs_dot = std::min(1.0, std::abs(dot));
      max_angle = std::max(max_angle, 2.0 * std::acos(abs_dot));
    }
    return max_angle;
  }

  Config cfg_;
  std::array<double, 9> last_pos_{};
  std::array<double, 12> last_orn_ = IdentityOrientation();
  bool initialized_ = false;
  int violation_streak_ = 0;
};

#endif  // VR3PT_SAFETY_FILTER_HPP
