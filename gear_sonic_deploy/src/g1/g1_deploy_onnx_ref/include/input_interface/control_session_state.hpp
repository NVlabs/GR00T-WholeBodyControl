/**
 * @file control_session_state.hpp
 * @brief Process-lifetime replay boundary for network control publishers.
 */

#ifndef CONTROL_SESSION_STATE_HPP
#define CONTROL_SESSION_STATE_HPP

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <system_error>

#include <sys/random.h>

inline constexpr std::size_t CONTROL_SESSION_TOKEN_SIZE = 16;
using ControlSessionToken =
    std::array<std::uint8_t, CONTROL_SESSION_TOKEN_SIZE>;

/**
 * Owns one receiver epoch and permanently binds the receiver to the first
 * valid publisher session that presents it.
 *
 * The owner must construct one instance for the deployment process and retain
 * it until process exit. Deliberately no reset API exists: stop, deadman, mode
 * changes, and publisher disconnects must not make an old capture admissible.
 */
class ControlSessionState final {
 public:
  static constexpr std::size_t TOKEN_SIZE = CONTROL_SESSION_TOKEN_SIZE;

  enum class ClaimResult {
    CLAIMED,
    ALREADY_CLAIMED,
    REJECTED_INVALID_TOKEN,
    REJECTED_RECEIVER_EPOCH,
    REJECTED_DIFFERENT_PUBLISHER,
  };

  ControlSessionState() : receiver_epoch_(GenerateReceiverEpoch()) {}

  ControlSessionState(const ControlSessionState&) = delete;
  ControlSessionState& operator=(const ControlSessionState&) = delete;
  ControlSessionState(ControlSessionState&&) = delete;
  ControlSessionState& operator=(ControlSessionState&&) = delete;

  /**
   * Validate the common wire representation for receiver and publisher IDs.
   * A token is exactly 16 bytes and may not be the all-zero sentinel.
   */
  static bool IsValidToken(const ControlSessionToken& token) noexcept {
    return std::any_of(
        token.begin(), token.end(), [](std::uint8_t byte) {
          return byte != 0;
        });
  }

  static std::optional<ControlSessionToken> ParseWireToken(
      std::span<const std::uint8_t> bytes) noexcept {
    if (bytes.size() != TOKEN_SIZE) {
      return std::nullopt;
    }

    ControlSessionToken token{};
    std::copy(bytes.begin(), bytes.end(), token.begin());
    if (!IsValidToken(token)) {
      return std::nullopt;
    }
    return token;
  }

  ControlSessionToken ReceiverEpoch() const noexcept {
    return receiver_epoch_;
  }

  bool IsReceiverEpoch(const ControlSessionToken& candidate) const noexcept {
    return IsValidToken(candidate) && candidate == receiver_epoch_;
  }

  /**
   * Claim control for a publisher session.
   *
   * First valid token wins. Repeating the winning token is idempotent. Any
   * different token remains rejected for this object's entire lifetime.
   */
  ClaimResult ClaimPublisher(
      const ControlSessionToken& receiver_epoch,
      const ControlSessionToken& publisher_session) noexcept {
    if (!IsValidToken(receiver_epoch) ||
        !IsValidToken(publisher_session)) {
      return ClaimResult::REJECTED_INVALID_TOKEN;
    }
    if (!IsReceiverEpoch(receiver_epoch)) {
      return ClaimResult::REJECTED_RECEIVER_EPOCH;
    }

    std::lock_guard<std::mutex> lock(claim_mutex_);
    if (!claimed_publisher_.has_value()) {
      claimed_publisher_ = publisher_session;
      return ClaimResult::CLAIMED;
    }
    if (*claimed_publisher_ == publisher_session) {
      return ClaimResult::ALREADY_CLAIMED;
    }
    return ClaimResult::REJECTED_DIFFERENT_PUBLISHER;
  }

  bool HasClaimedPublisher() const noexcept {
    std::lock_guard<std::mutex> lock(claim_mutex_);
    return claimed_publisher_.has_value();
  }

  bool IsClaimedPublisher(
      const ControlSessionToken& receiver_epoch,
      const ControlSessionToken& publisher_session) const noexcept {
    if (!IsReceiverEpoch(receiver_epoch) ||
        !IsValidToken(publisher_session)) {
      return false;
    }
    std::lock_guard<std::mutex> lock(claim_mutex_);
    return claimed_publisher_.has_value() &&
           *claimed_publisher_ == publisher_session;
  }

  std::optional<ControlSessionToken> ClaimedPublisher() const noexcept {
    std::lock_guard<std::mutex> lock(claim_mutex_);
    return claimed_publisher_;
  }

 private:
  static void FillFromGetRandom(ControlSessionToken& token) {
    std::size_t offset = 0;
    while (offset < token.size()) {
      const auto bytes_read = ::getrandom(
          token.data() + offset, token.size() - offset, 0);
      if (bytes_read < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::system_error(
            errno, std::generic_category(), "getrandom");
      }
      if (bytes_read == 0) {
        throw std::runtime_error(
            "getrandom returned zero bytes while generating receiver epoch");
      }
      offset += static_cast<std::size_t>(bytes_read);
    }
  }

  static ControlSessionToken GenerateReceiverEpoch() {
    // An all-zero value is reserved as an invalid/uninitialised wire token.
    // Refill rather than silently weakening the invariant.
    constexpr int MAX_GENERATION_ATTEMPTS = 4;
    for (int attempt = 0; attempt < MAX_GENERATION_ATTEMPTS; ++attempt) {
      ControlSessionToken epoch{};
      FillFromGetRandom(epoch);
      if (IsValidToken(epoch)) {
        return epoch;
      }
    }
    throw std::runtime_error(
        "getrandom repeatedly produced an invalid receiver epoch");
  }

  const ControlSessionToken receiver_epoch_;
  mutable std::mutex claim_mutex_;
  std::optional<ControlSessionToken> claimed_publisher_;
};

#endif  // CONTROL_SESSION_STATE_HPP
