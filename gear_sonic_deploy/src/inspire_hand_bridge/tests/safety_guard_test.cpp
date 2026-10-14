#include <cassert>
#include <limits>

#include "inspire/safety_guard.hpp"

inspire::CommandV1 command(std::uint64_t sequence, std::uint64_t timestamp, float value) {
  inspire::CommandV1 result;
  result.sequence = sequence;
  result.source_monotonic_ns = timestamp;
  result.left_q.fill(value);
  result.right_q.fill(value);
  return result;
}

int main() {
  inspire::SafetyGuard guard({.command_timeout_ns = 100'000'000, .max_future_skew_ns = 1'000'000});
  assert(!guard.accept(command(1, 1'000'000'000, 0.F), 1'000'000'000));
  guard.clear_fault();
  assert(guard.arm());
  assert(guard.accept(command(1, 1'000'000'000, 0.F), 1'000'000'000));
  assert(!guard.accept(command(1, 1'010'000'000, 0.F), 1'010'000'000));
  guard.clear_fault();
  assert(guard.arm());
  assert(!guard.accept(command(2, 2'000'000'000, 2.F), 2'000'000'000));
  guard.clear_fault();
  assert(guard.arm());
  auto nan = command(3, 3'000'000'000, 0.F);
  nan.left_q[0] = std::numeric_limits<float>::quiet_NaN();
  assert(!guard.accept(nan, 3'000'000'000));
  guard.clear_fault();
  assert(guard.arm());
  assert(guard.accept(command(4, 4'000'000'000, 0.F), 4'000'000'000));
  assert(guard.poll(4'100'000'001) == inspire::Status::Stale);
  assert(guard.arm());
  assert(guard.accept(command(5, 4'200'000'000, 0.F), 4'200'000'000));
  assert(guard.status() == inspire::Status::Armed);
  guard.disarm();
  assert(guard.last_command()->sequence == 5);
  assert(guard.arm());
  assert(!guard.accept(command(5, 4'300'000'000, 0.F), 4'300'000'000));
  guard.clear_fault();
  assert(guard.arm());
  guard.fault("malformed command packet");
  assert(guard.status() == inspire::Status::Fault);
  assert(guard.last_error() == "malformed command packet");

  inspire::SafetyGuard cross_host_guard({
      .command_timeout_ns = 100'000'000,
      .max_future_skew_ns = 1'000'000,
      .validate_source_clock_age = false,
  });
  assert(cross_host_guard.arm());
  assert(cross_host_guard.accept(command(1, 99'000'000'000, 0.F), 5'000'000'000));
  assert(cross_host_guard.poll(5'100'000'001) == inspire::Status::Stale);
  return 0;
}
