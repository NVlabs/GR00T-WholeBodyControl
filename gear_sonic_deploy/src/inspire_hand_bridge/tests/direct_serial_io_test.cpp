#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "inspire/direct_serial_io.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  std::array<int, 2> sockets{};
  require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, sockets.data()) == 0,
          "socketpair failed");

  std::thread fragmented_writer([&] {
    const std::array<std::uint8_t, 2> first{1, 2};
    const std::array<std::uint8_t, 3> second{3, 4, 5};
    inspire::serial::write_exact_fd(
        sockets[0], first, inspire::serial::IoClock::now() + std::chrono::milliseconds(50),
        "fragment writer");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    inspire::serial::write_exact_fd(
        sockets[0], second, inspire::serial::IoClock::now() + std::chrono::milliseconds(50),
        "fragment writer");
  });

  const auto received = inspire::serial::read_exact_fd(
      sockets[1], 5, inspire::serial::IoClock::now() + std::chrono::milliseconds(100),
      "fragment reader");
  fragmented_writer.join();
  require(received == std::vector<std::uint8_t>({1, 2, 3, 4, 5}),
          "partial reads were not accumulated exactly");

  bool timed_out = false;
  try {
    (void)inspire::serial::read_exact_fd(
        sockets[1], 1, inspire::serial::IoClock::now() + std::chrono::milliseconds(5),
        "timeout reader");
  } catch (const std::runtime_error&) {
    timed_out = true;
  }
  require(timed_out, "empty exact read did not time out");

  ::close(sockets[0]);
  ::close(sockets[1]);
  return 0;
}
