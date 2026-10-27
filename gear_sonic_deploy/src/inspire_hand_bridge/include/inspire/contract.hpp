#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace inspire {

inline constexpr std::size_t kHandDof = 6;
inline constexpr char kCommandSchema[] = "inspire.hand.command.v2";
inline constexpr char kStateSchema[] = "inspire.hand.state.v2";
inline constexpr std::array<const char*, kHandDof> kJointOrder = {
    "pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"};
inline constexpr std::array<float, kHandDof> kLowerLimits = {0.F, 0.F, 0.F, 0.F, 0.F, 0.F};
inline constexpr std::array<float, kHandDof> kUpperLimits = {1.7F, 1.7F, 1.7F, 1.7F, 0.6F, 1.3F};
inline constexpr std::array<float, kHandDof> kVelocityLimits = {2.F, 2.F, 2.F, 2.F, 1.F, 1.F};

using HandVector = std::array<float, kHandDof>;

enum class Status { Disarmed, Armed, Fault, Stale };

inline const char* to_string(Status status) {
  switch (status) {
    case Status::Disarmed: return "DISARMED";
    case Status::Armed: return "ARMED";
    case Status::Fault: return "FAULT";
    case Status::Stale: return "STALE";
  }
  return "FAULT";
}

struct CommandV1 {
  std::uint64_t sequence{};
  std::uint64_t source_monotonic_ns{};
  HandVector left_q{};
  HandVector right_q{};
  std::optional<std::uint64_t> frame_index;
};

struct StateV1 {
  std::uint64_t sequence{};
  std::uint64_t source_monotonic_ns{};
  std::uint64_t gateway_monotonic_ns{};
  Status status{Status::Disarmed};
  HandVector left_q{};
  HandVector right_q{};
  std::optional<HandVector> left_dq;
  std::optional<HandVector> right_dq;
  std::optional<std::uint64_t> accepted_command_sequence;
  std::optional<std::uint64_t> accepted_frame_index;
  std::optional<HandVector> left_command_q;
  std::optional<HandVector> right_command_q;
};

inline bool finite(const HandVector& values) {
  for (float value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

}  // namespace inspire
