#pragma once

#include <cstdint>
#include <string>

#include "inspire/contract.hpp"

namespace inspire {

struct SafetyConfig {
  std::uint64_t command_timeout_ns{200'000'000};
  std::uint64_t max_future_skew_ns{20'000'000};
  bool validate_source_clock_age{true};
};

class SafetyGuard {
 public:
  explicit SafetyGuard(SafetyConfig config = {}) : config_(config) {}

  Status status() const { return status_; }
  const std::string& last_error() const { return last_error_; }
  const std::optional<CommandV1>& last_command() const { return last_command_; }

  bool arm() {
    if (status_ == Status::Fault) return false;
    status_ = Status::Armed;
    last_error_.clear();
    return true;
  }

  void disarm() {
    status_ = Status::Disarmed;
    // Preserve the accepted sequence across a temporary transport interruption so
    // an old command cannot be replayed when measurements return.
  }

  void clear_fault() {
    status_ = Status::Disarmed;
    last_command_.reset();
    last_accept_monotonic_ns_ = 0;
    last_error_.clear();
  }

  void fault(std::string error) { reject(std::move(error)); }

  bool accept(const CommandV1& command, std::uint64_t now_ns) {
    if (status_ != Status::Armed) return reject("gateway is not armed");
    if (!finite(command.left_q) || !finite(command.right_q)) return reject("non-finite hand value");
    if (config_.validate_source_clock_age) {
      if (now_ns > command.source_monotonic_ns &&
          now_ns - command.source_monotonic_ns > config_.command_timeout_ns) {
        return reject("command is stale");
      }
      if (command.source_monotonic_ns > now_ns &&
          command.source_monotonic_ns - now_ns > config_.max_future_skew_ns) {
        return reject("command timestamp is too far in the future");
      }
    }
    if (!within_limits(command.left_q) || !within_limits(command.right_q)) {
      return reject("command exceeds position limits");
    }
    if (last_command_) {
      if (command.sequence <= last_command_->sequence) return reject("sequence is not increasing");
      if (command.source_monotonic_ns <= last_command_->source_monotonic_ns) {
        return reject("timestamp is not increasing");
      }
      const double elapsed =
          static_cast<double>(command.source_monotonic_ns - last_command_->source_monotonic_ns) / 1e9;
      if (!within_slew(command.left_q, last_command_->left_q, elapsed) ||
          !within_slew(command.right_q, last_command_->right_q, elapsed)) {
        return reject("command exceeds velocity limits");
      }
    }
    last_command_ = command;
    last_accept_monotonic_ns_ = now_ns;
    return true;
  }

  Status poll(std::uint64_t now_ns) {
    if (status_ == Status::Armed && last_command_ &&
        now_ns > last_accept_monotonic_ns_ &&
        now_ns - last_accept_monotonic_ns_ > config_.command_timeout_ns) {
      status_ = Status::Stale;
      last_error_ = "command watchdog expired";
    }
    return status_;
  }

 private:
  bool reject(std::string error) {
    status_ = Status::Fault;
    last_error_ = std::move(error);
    return false;
  }

  static bool within_limits(const HandVector& values) {
    for (std::size_t i = 0; i < kHandDof; ++i) {
      if (values[i] < kLowerLimits[i] || values[i] > kUpperLimits[i]) return false;
    }
    return true;
  }

  static bool within_slew(const HandVector& values, const HandVector& previous, double elapsed) {
    for (std::size_t i = 0; i < kHandDof; ++i) {
      if (std::abs(values[i] - previous[i]) / elapsed > kVelocityLimits[i]) return false;
    }
    return true;
  }

  SafetyConfig config_;
  Status status_{Status::Disarmed};
  std::optional<CommandV1> last_command_;
  std::uint64_t last_accept_monotonic_ns_{0};
  std::string last_error_;
};

}  // namespace inspire
