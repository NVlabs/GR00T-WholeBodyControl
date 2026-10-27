#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

#include "inspire/contract.hpp"

namespace inspire::serial {

inline constexpr std::uint8_t kRequestHeader0 = 0xEB;
inline constexpr std::uint8_t kRequestHeader1 = 0x90;
inline constexpr std::uint8_t kResponseHeader0 = 0x90;
inline constexpr std::uint8_t kResponseHeader1 = 0xEB;
inline constexpr std::size_t kSetRequestSize = 20;
inline constexpr std::size_t kGetRequestSize = 9;
inline constexpr std::size_t kAckSize = 9;
inline constexpr std::size_t kPositionResponseSize = 20;
inline constexpr std::uint16_t kPositionAddress = 0x060A;
inline constexpr std::uint16_t kSpeedAddress = 0x05F2;
inline constexpr std::uint16_t kClearErrorAddress = 0x03EC;
inline constexpr std::uint16_t kMaximumSpeedSetting = 1000;
using SpeedVector = std::array<std::uint16_t, kHandDof>;

inline std::uint8_t checksum(std::span<const std::uint8_t> frame) {
  if (frame.size() < 4) throw std::runtime_error("Inspire frame is too short for a checksum");
  std::uint8_t sum = 0;
  for (std::size_t index = 2; index + 1 < frame.size(); ++index) {
    sum = static_cast<std::uint8_t>(sum + frame[index]);
  }
  return sum;
}

inline void validate_envelope(std::span<const std::uint8_t> frame,
                              std::size_t expected_size,
                              std::uint8_t expected_id,
                              const std::string& kind) {
  if (frame.size() != expected_size) {
    throw std::runtime_error(kind + " has the wrong frame size");
  }
  if (frame[0] != kResponseHeader0 || frame[1] != kResponseHeader1) {
    throw std::runtime_error(kind + " has the wrong header");
  }
  if (frame[2] != expected_id) {
    throw std::runtime_error(kind + " has the wrong hand id");
  }
  if (frame.back() != checksum(frame)) {
    throw std::runtime_error(kind + " checksum failed");
  }
}

inline std::array<std::uint8_t, kSetRequestSize> make_set_position_request(
    const HandVector& normalized, std::uint8_t id = 1) {
  std::array<std::uint8_t, kSetRequestSize> frame{};
  frame[0] = kRequestHeader0;
  frame[1] = kRequestHeader1;
  frame[2] = id;
  frame[3] = 0x0F;
  frame[4] = 0x12;
  frame[5] = 0xCE;
  frame[6] = 0x05;
  for (std::size_t index = 0; index < kHandDof; ++index) {
    if (!std::isfinite(normalized[index]) || normalized[index] < 0.F ||
        normalized[index] > 1.F) {
      throw std::runtime_error("normalized Inspire command is outside [0, 1]");
    }
    const auto scaled = static_cast<std::uint16_t>(std::lround(normalized[index] * 1000.F));
    frame[7 + 2 * index] = static_cast<std::uint8_t>(scaled & 0xFFU);
    frame[8 + 2 * index] = static_cast<std::uint8_t>((scaled >> 8U) & 0xFFU);
  }
  frame.back() = checksum(frame);
  return frame;
}

inline std::array<std::uint8_t, kGetRequestSize> make_get_position_request(
    std::uint8_t id = 1) {
  std::array<std::uint8_t, kGetRequestSize> frame{
      kRequestHeader0, kRequestHeader1, id, 0x04, 0x11, 0x0A, 0x06, 0x0C, 0x00};
  frame.back() = checksum(frame);
  return frame;
}

inline std::array<std::uint8_t, kSetRequestSize> make_set_speed_request(
    const SpeedVector& speed, std::uint8_t id = 1) {
  std::array<std::uint8_t, kSetRequestSize> frame{};
  frame[0] = kRequestHeader0;
  frame[1] = kRequestHeader1;
  frame[2] = id;
  frame[3] = 0x0F;
  frame[4] = 0x12;
  frame[5] = static_cast<std::uint8_t>(kSpeedAddress & 0xFFU);
  frame[6] = static_cast<std::uint8_t>((kSpeedAddress >> 8U) & 0xFFU);
  for (std::size_t index = 0; index < kHandDof; ++index) {
    if (speed[index] > kMaximumSpeedSetting) {
      throw std::runtime_error("Inspire speed setting is outside [0, 1000]");
    }
    frame[7 + 2 * index] = static_cast<std::uint8_t>(speed[index] & 0xFFU);
    frame[8 + 2 * index] = static_cast<std::uint8_t>((speed[index] >> 8U) & 0xFFU);
  }
  frame.back() = checksum(frame);
  return frame;
}

inline std::array<std::uint8_t, kGetRequestSize> make_get_speed_request(
    std::uint8_t id = 1) {
  std::array<std::uint8_t, kGetRequestSize> frame{
      kRequestHeader0,
      kRequestHeader1,
      id,
      0x04,
      0x11,
      static_cast<std::uint8_t>(kSpeedAddress & 0xFFU),
      static_cast<std::uint8_t>((kSpeedAddress >> 8U) & 0xFFU),
      0x0C,
      0x00};
  frame.back() = checksum(frame);
  return frame;
}

inline std::array<std::uint8_t, kGetRequestSize> make_clear_error_request(
    std::uint8_t id = 1) {
  std::array<std::uint8_t, kGetRequestSize> frame{
      kRequestHeader0,
      kRequestHeader1,
      id,
      0x04,
      0x12,
      static_cast<std::uint8_t>(kClearErrorAddress & 0xFFU),
      static_cast<std::uint8_t>((kClearErrorAddress >> 8U) & 0xFFU),
      0x01,
      0x00};
  frame.back() = checksum(frame);
  return frame;
}

inline void validate_ack(std::span<const std::uint8_t> frame, std::uint8_t id = 1) {
  validate_envelope(frame, kAckSize, id, "Inspire write acknowledgement");
}

inline HandVector decode_position_response(std::span<const std::uint8_t> frame,
                                           std::uint8_t id = 1) {
  validate_envelope(frame, kPositionResponseSize, id, "Inspire position response");
  HandVector normalized{};
  for (std::size_t index = 0; index < kHandDof; ++index) {
    const auto raw = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(frame[7 + 2 * index]) |
        (static_cast<std::uint16_t>(frame[8 + 2 * index]) << 8U));
    if (raw > 1000U) {
      throw std::runtime_error("Inspire position response exceeds normalized range");
    }
    normalized[index] = static_cast<float>(raw) / 1000.F;
  }
  return normalized;
}

inline SpeedVector decode_speed_response(std::span<const std::uint8_t> frame,
                                         std::uint8_t id = 1) {
  validate_envelope(frame, kPositionResponseSize, id, "Inspire speed response");
  SpeedVector speed{};
  for (std::size_t index = 0; index < kHandDof; ++index) {
    speed[index] = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(frame[7 + 2 * index]) |
        (static_cast<std::uint16_t>(frame[8 + 2 * index]) << 8U));
    if (speed[index] > kMaximumSpeedSetting) {
      throw std::runtime_error("Inspire speed response exceeds [0, 1000]");
    }
  }
  return speed;
}

}  // namespace inspire::serial
