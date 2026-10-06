#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <poll.h>
#include <span>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace inspire::serial {

using IoClock = std::chrono::steady_clock;

inline int remaining_timeout_ms(IoClock::time_point deadline) {
  const auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - IoClock::now()).count();
  return static_cast<int>(std::max<std::int64_t>(1, remaining));
}

inline void wait_for_fd(int descriptor,
                        short events,
                        IoClock::time_point deadline,
                        const std::string& label) {
  while (IoClock::now() < deadline) {
    pollfd polled{descriptor, events, 0};
    const int result = ::poll(&polled, 1, remaining_timeout_ms(deadline));
    if (result > 0) {
      if ((polled.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        throw std::runtime_error("I/O poll reported a device error: " + label);
      }
      if ((polled.revents & events) != 0) return;
    } else if (result == 0) {
      break;
    } else if (errno != EINTR) {
      throw std::runtime_error("I/O poll failed: " + label);
    }
  }
  throw std::runtime_error("I/O transaction timed out: " + label);
}

inline void write_exact_fd(int descriptor,
                           std::span<const std::uint8_t> bytes,
                           IoClock::time_point deadline,
                           const std::string& label) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    wait_for_fd(descriptor, POLLOUT, deadline, label);
    const ssize_t count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno != EINTR && errno != EAGAIN) {
      throw std::runtime_error("I/O write failed: " + label);
    }
  }
}

inline std::vector<std::uint8_t> read_exact_fd(int descriptor,
                                                std::size_t size,
                                                IoClock::time_point deadline,
                                                const std::string& label) {
  std::vector<std::uint8_t> result(size);
  std::size_t offset = 0;
  while (offset < size) {
    wait_for_fd(descriptor, POLLIN, deadline, label);
    const ssize_t count = ::read(descriptor, result.data() + offset, size - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno != EINTR && errno != EAGAIN) {
      throw std::runtime_error("I/O read failed: " + label);
    }
  }
  return result;
}

}  // namespace inspire::serial
