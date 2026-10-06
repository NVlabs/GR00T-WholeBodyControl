#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include <system_error>

#include <zmq.hpp>

#include "inspire/contract.hpp"
#include "inspire/direct_serial_io.hpp"
#include "inspire/direct_serial_protocol.hpp"
#include "inspire/feedback_guard.hpp"
#include "inspire/real_hardware_codec.hpp"
#include "inspire/safety_guard.hpp"
#include "inspire/zmq_endpoint.hpp"
#include "inspire/operator_session.hpp"

namespace {

using Clock = std::chrono::steady_clock;

// Per-hand period is configured in main; the deployment default is 50 Hz.
constexpr auto kTransactionTimeout = std::chrono::milliseconds(40);
constexpr auto kSafetyHoldWriteTimeout = std::chrono::milliseconds(300);
constexpr std::uint64_t kStateTimeoutNs = 300'000'000;
constexpr std::size_t kHealthyReadStreak = 3;
// Real RH56DFX feedback tests on this G1 showed that thumb rotation does not
// produce measurable motion for canonical steps up to 0.024 rad.  Keep the
// proven 0.025 gate for the other five axes, while allowing a bounded 0.050
// diagnostic/execution step on thumb rotation.  This is device-class
// calibration used by every command source (Replay, VLA, and smoke tests), not
// a trajectory-specific exception.
constexpr inspire::HandVector kHoldCurrentToleranceRad = {
    0.025F, 0.025F, 0.025F, 0.025F, 0.025F, 0.050F};
// The first native6 command must be close to measured feedback.  Subsequent
// commands are guarded by SafetyGuard's exact per-joint position and velocity
// limits against the previously accepted command.  Comparing every new target
// with lagging feedback would deadlock an actuator whose observable motion has
// a device deadband; the lab hardware layer likewise sends position targets
// without rewriting them from feedback on every cycle.
constexpr inspire::HandVector kInitialNative6ToleranceRad = {
    0.100F, 0.100F, 0.100F, 0.100F, 0.050F, 0.050F};
constexpr char kDefaultWriterLockPath[] = "/tmp/inspire-direct-serial-hand-gateway.writer.lock";

volatile std::sig_atomic_t running = 1;
void stop_signal(int) { running = 0; }

std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()).count());
}

class SingleWriterLock {
 public:
  explicit SingleWriterLock(const std::string& path) {
    descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (descriptor_ < 0 || ::flock(descriptor_, LOCK_EX | LOCK_NB) != 0) {
      if (descriptor_ >= 0) ::close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("cannot acquire direct serial Gateway writer lock: " + path);
    }
  }

  ~SingleWriterLock() {
    if (descriptor_ >= 0) {
      ::flock(descriptor_, LOCK_UN);
      ::close(descriptor_);
    }
  }

  SingleWriterLock(const SingleWriterLock&) = delete;
  SingleWriterLock& operator=(const SingleWriterLock&) = delete;

 private:
  int descriptor_{-1};
};

std::vector<int> device_owner_pids(const std::string& device) {
  struct stat target {};
  if (::stat(device.c_str(), &target) != 0 || !S_ISCHR(target.st_mode)) {
    throw std::runtime_error("serial device is missing or is not a character device: " + device);
  }

  std::vector<int> owners;
  std::error_code error;
  const auto options = std::filesystem::directory_options::skip_permission_denied;
  for (const auto& process : std::filesystem::directory_iterator("/proc", options, error)) {
    if (error) break;
    const std::string name = process.path().filename().string();
    if (name.empty() || !std::all_of(name.begin(), name.end(), ::isdigit)) continue;
    const int pid = std::stoi(name);
    if (pid == ::getpid()) continue;

    std::error_code fd_error;
    const auto fd_path = process.path() / "fd";
    if (!std::filesystem::is_directory(fd_path, fd_error)) continue;
    for (const auto& descriptor :
         std::filesystem::directory_iterator(fd_path, options, fd_error)) {
      if (fd_error) break;
      struct stat candidate {};
      if (::stat(descriptor.path().c_str(), &candidate) == 0 &&
          S_ISCHR(candidate.st_mode) && candidate.st_rdev == target.st_rdev) {
        owners.push_back(pid);
        break;
      }
    }
  }
  std::sort(owners.begin(), owners.end());
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
  return owners;
}

void require_unowned(const std::string& device) {
  const auto owners = device_owner_pids(device);
  if (owners.empty()) return;
  std::string detail;
  for (const int pid : owners) {
    if (!detail.empty()) detail += ',';
    detail += std::to_string(pid);
  }
  throw std::runtime_error(
      "serial device already has an owner; close the owning process before retrying: " + device +
      " owner_pids=" + detail);
}

class PosixSerialPort {
 public:
  PosixSerialPort(std::string path, speed_t baud, std::uint8_t id)
      : path_(std::move(path)), baud_(baud), id_(id) {
    descriptor_ = ::open(path_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
    if (descriptor_ < 0) {
      const int error = errno;
      throw std::system_error(error, std::generic_category(),
                              "failed to open serial device: " + path_);
    }
    try {
      if (::flock(descriptor_, LOCK_EX | LOCK_NB) != 0) {
        throw std::runtime_error("failed to acquire serial device lock: " + path_);
      }
      if (::ioctl(descriptor_, TIOCEXCL) != 0) {
        throw std::runtime_error("failed to make serial device exclusive: " + path_);
      }
      configure();
    } catch (...) {
      ::ioctl(descriptor_, TIOCNXCL);
      ::flock(descriptor_, LOCK_UN);
      ::close(descriptor_);
      descriptor_ = -1;
      throw;
    }
  }

  ~PosixSerialPort() {
    if (descriptor_ >= 0) {
      ::ioctl(descriptor_, TIOCNXCL);
      ::flock(descriptor_, LOCK_UN);
      ::close(descriptor_);
    }
  }

  PosixSerialPort(const PosixSerialPort&) = delete;
  PosixSerialPort& operator=(const PosixSerialPort&) = delete;

  void set_position(const inspire::HandVector& normalized) {
    const auto id = id_;
    const auto request = inspire::serial::make_set_position_request(normalized, id);
    const auto response = transact(request, inspire::serial::kAckSize);
    inspire::serial::validate_ack(response, id);
  }

  inspire::HandVector get_position() {
    const auto id = id_;
    const auto request = inspire::serial::make_get_position_request(id);
    const auto response = transact(request, inspire::serial::kPositionResponseSize);
    return inspire::serial::decode_position_response(response, id);
  }

  void set_speed(const inspire::serial::SpeedVector& speed) {
    const auto id = id_;
    const auto request = inspire::serial::make_set_speed_request(speed, id);
    const auto response = transact(request, inspire::serial::kAckSize);
    inspire::serial::validate_ack(response, id);
  }

  inspire::serial::SpeedVector get_speed() {
    const auto id = id_;
    const auto request = inspire::serial::make_get_speed_request(id);
    const auto response = transact(request, inspire::serial::kPositionResponseSize);
    return inspire::serial::decode_speed_response(response, id);
  }

  void clear_error() {
    const auto id = id_;
    const auto request = inspire::serial::make_clear_error_request(id);
    const auto response = transact(request, inspire::serial::kAckSize);
    inspire::serial::validate_ack(response, id);
  }

 private:
  void configure() {
    termios options{};
    if (::tcgetattr(descriptor_, &options) != 0) {
      throw std::runtime_error("tcgetattr failed for serial device: " + path_);
    }
    ::cfmakeraw(&options);
    ::cfsetispeed(&options, baud_);
    ::cfsetospeed(&options, baud_);
    options.c_cflag |= CLOCAL | CREAD;
    options.c_cflag &= ~CRTSCTS;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 0;
    if (::tcsetattr(descriptor_, TCSANOW, &options) != 0) {
      throw std::runtime_error("tcsetattr failed for serial device: " + path_);
    }
    if (::tcflush(descriptor_, TCIOFLUSH) != 0) {
      throw std::runtime_error("tcflush failed for serial device: " + path_);
    }
  }

  void write_exact(std::span<const std::uint8_t> bytes, Clock::time_point deadline) {
    inspire::serial::write_exact_fd(descriptor_, bytes, deadline, path_);
    if (::tcdrain(descriptor_) != 0) {
      throw std::runtime_error("serial drain failed: " + path_);
    }
  }

  std::vector<std::uint8_t> read_exact(std::size_t size, Clock::time_point deadline) {
    return inspire::serial::read_exact_fd(descriptor_, size, deadline, path_);
  }

  template <std::size_t Size>
  std::vector<std::uint8_t> transact(const std::array<std::uint8_t, Size>& request,
                                     std::size_t response_size) {
    if (::tcflush(descriptor_, TCIFLUSH) != 0) {
      throw std::runtime_error("serial input flush failed: " + path_);
    }
    const auto deadline = Clock::now() + kTransactionTimeout;
    write_exact(request, deadline);
    return read_exact(response_size, deadline);
  }

  std::string path_;
  speed_t baud_;
  std::uint8_t id_;
  int descriptor_{-1};
};

struct PendingWrite {
  std::uint64_t sequence{};
  inspire::HandVector normalized{};
  inspire::HandVector canonical{};
};

struct WorkerSnapshot {
  inspire::HandVector q{};
  inspire::HandVector dq{};
  std::uint64_t state_monotonic_ns{};
  std::uint64_t state_generation{};
  std::size_t consecutive_failures{};
  std::size_t total_failures{};
  bool received{false};
  bool healthy{false};
  std::optional<std::uint64_t> last_attempted_sequence;
  std::optional<std::uint64_t> last_written_sequence;
  std::optional<inspire::HandVector> last_written_q;
  std::uint64_t clear_error_requested_generation{};
  std::uint64_t clear_error_completed_generation{};
  std::string last_error;
};

class HandSerialWorker {
 public:
  HandSerialWorker(std::string label, std::string device, std::uint16_t runtime_speed,
                   speed_t baud, std::uint8_t id, Clock::duration period)
      : label_(std::move(label)), port_(std::move(device), baud, id), period_(period) {
    original_speed_ = port_.get_speed();
    runtime_speed_.fill(runtime_speed);
    port_.set_speed(runtime_speed_);
    const auto verified = port_.get_speed();
    if (verified != runtime_speed_) {
      throw std::runtime_error("Inspire runtime speed readback did not match requested value");
    }
    speed_configured_ = true;
    std::cout << "DIRECT_SERIAL_SPEED_CONFIGURED hand=" << label_
              << " runtime=" << runtime_speed
              << " original=[";
    for (std::size_t index = 0; index < original_speed_.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << original_speed_[index];
    }
    std::cout << "] verified=true" << std::endl;
    thread_ = std::thread(&HandSerialWorker::run, this);
  }

  ~HandSerialWorker() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    if (speed_configured_) {
      try {
        port_.set_speed(original_speed_);
        const bool verified = port_.get_speed() == original_speed_;
        std::cout << "DIRECT_SERIAL_SPEED_RESTORED hand=" << label_
                  << " verified=" << (verified ? "true" : "false") << std::endl;
      } catch (const std::exception& error) {
        std::cerr << "DIRECT_SERIAL_SPEED_RESTORE_FAILED hand=" << label_
                  << " detail='" << error.what() << "'" << std::endl;
      }
    }
  }

  HandSerialWorker(const HandSerialWorker&) = delete;
  HandSerialWorker& operator=(const HandSerialWorker&) = delete;

  void stage(PendingWrite write) {
    std::scoped_lock lock(mutex_);
    pending_ = std::move(write);
    writes_allowed_ = true;
  }

  void disable_writes() {
    std::scoped_lock lock(mutex_);
    writes_allowed_ = false;
    pending_.reset();
  }

  void reset_command_session() {
    std::scoped_lock lock(mutex_);
    writes_allowed_ = false;
    pending_.reset();
    // Command sequence numbers are scoped to one producer session.  RESET HAND
    // starts a new session, whose first legal sequence may be 1 again.
    snapshot_.last_attempted_sequence.reset();
    snapshot_.last_written_sequence.reset();
    snapshot_.last_written_q.reset();
  }

  std::uint64_t request_hardware_error_clear() {
    std::scoped_lock lock(mutex_);
    writes_allowed_ = false;
    pending_.reset();
    const auto generation = ++snapshot_.clear_error_requested_generation;
    pending_clear_error_ = generation;
    return generation;
  }

  WorkerSnapshot snapshot() const {
    std::scoped_lock lock(mutex_);
    return snapshot_;
  }

 private:
  std::size_t record_failure(const std::string& error) {
    std::scoped_lock lock(mutex_);
    ++snapshot_.consecutive_failures;
    ++snapshot_.total_failures;
    snapshot_.healthy = false;
    snapshot_.last_error = error;
    successful_read_streak_ = 0;
    return snapshot_.total_failures;
  }

  void record_state(const inspire::HandVector& normalized) {
    const auto now = monotonic_ns();
    const auto canonical = inspire::real::normalized_to_canonical(normalized);
    std::scoped_lock lock(mutex_);
    inspire::HandVector velocity{};
    if (snapshot_.received && now > snapshot_.state_monotonic_ns) {
      const double seconds = static_cast<double>(now - snapshot_.state_monotonic_ns) / 1e9;
      for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
        velocity[index] = static_cast<float>((canonical[index] - snapshot_.q[index]) / seconds);
      }
    }
    snapshot_.q = canonical;
    snapshot_.dq = velocity;
    snapshot_.state_monotonic_ns = now;
    ++snapshot_.state_generation;
    snapshot_.received = true;
    snapshot_.consecutive_failures = 0;
    snapshot_.last_error.clear();
    ++successful_read_streak_;
    snapshot_.healthy = successful_read_streak_ >= kHealthyReadStreak;
  }

  void run() {
    auto next_tick = Clock::now();
    while (!stop_.load()) {
      std::optional<PendingWrite> write;
      std::optional<std::uint64_t> clear_error_generation;
      {
        std::scoped_lock lock(mutex_);
        if (pending_clear_error_) {
          clear_error_generation = pending_clear_error_;
          pending_clear_error_.reset();
        }
        if (writes_allowed_ && pending_ &&
            (!snapshot_.last_attempted_sequence ||
             pending_->sequence > *snapshot_.last_attempted_sequence)) {
          write = pending_;
          snapshot_.last_attempted_sequence = write->sequence;
          pending_.reset();
        }
      }

      try {
        if (clear_error_generation) {
          port_.clear_error();
          std::scoped_lock lock(mutex_);
          snapshot_.clear_error_completed_generation = *clear_error_generation;
        }
        if (write) {
          port_.set_position(write->normalized);
          std::scoped_lock lock(mutex_);
          snapshot_.last_written_sequence = write->sequence;
          snapshot_.last_written_q = write->canonical;
        }
        record_state(port_.get_position());
      } catch (const std::exception& error) {
        const auto failure_count = record_failure(error.what());
        if (failure_count <= 3 || failure_count % 20 == 0) {
          std::cerr << "DIRECT_SERIAL_ERROR hand=" << label_
                    << " total_failures=" << failure_count
                    << " detail=" << error.what() << std::endl;
        }
      }

      next_tick += period_;
      if (next_tick < Clock::now()) {
        ++overruns_;
        if (overruns_ <= 3 || overruns_ % 50 == 0) {
          std::cerr << "DIRECT_SERIAL_OVERRUN hand=" << label_
                    << " count=" << overruns_ << std::endl;
        }
        // Rebase instead of issuing a burst to catch up missed deadlines.
        next_tick = Clock::now() + period_;
      }
      std::this_thread::sleep_until(next_tick);
    }
  }

  std::string label_;
  PosixSerialPort port_;
  Clock::duration period_;
  std::uint64_t overruns_{};
  mutable std::mutex mutex_;
  WorkerSnapshot snapshot_;
  std::optional<PendingWrite> pending_;
  std::optional<std::uint64_t> pending_clear_error_;
  std::size_t successful_read_streak_{};
  bool writes_allowed_{false};
  std::atomic<bool> stop_{false};
  std::thread thread_;
  inspire::serial::SpeedVector original_speed_{};
  inspire::serial::SpeedVector runtime_speed_{};
  bool speed_configured_{false};
};

bool fresh_and_healthy(const WorkerSnapshot& snapshot, std::uint64_t now) {
  return snapshot.received && snapshot.healthy && now >= snapshot.state_monotonic_ns &&
         now - snapshot.state_monotonic_ns <= kStateTimeoutNs;
}

struct SafetyHoldResult {
  bool sent{false};
  std::string reason;
};

SafetyHoldResult hold_current_and_disable(HandSerialWorker& left,
                                          HandSerialWorker& right) {
  // Cancel producer writes before taking the hold snapshot.  A write already
  // being transacted may finish, but the higher internal sequence below is
  // staged afterwards and therefore becomes the final hardware target.
  left.disable_writes();
  right.disable_writes();

  const auto left_state = left.snapshot();
  const auto right_state = right.snapshot();
  const auto now = monotonic_ns();
  if (!fresh_and_healthy(left_state, now) ||
      !fresh_and_healthy(right_state, now)) {
    return {false, "hardware_unhealthy"};
  }

  std::uint64_t highest_sequence = 0;
  for (const auto sequence : {left_state.last_attempted_sequence,
                              right_state.last_attempted_sequence}) {
    if (sequence) highest_sequence = std::max(highest_sequence, *sequence);
  }
  if (highest_sequence == std::numeric_limits<std::uint64_t>::max()) {
    return {false, "sequence_exhausted"};
  }
  const auto hold_sequence = highest_sequence + 1;
  left.stage(PendingWrite{
      hold_sequence,
      inspire::real::canonical_to_normalized(left_state.q),
      left_state.q,
  });
  right.stage(PendingWrite{
      hold_sequence,
      inspire::real::canonical_to_normalized(right_state.q),
      right_state.q,
  });

  const auto deadline = Clock::now() + kSafetyHoldWriteTimeout;
  while (Clock::now() < deadline) {
    const auto left_progress = left.snapshot();
    const auto right_progress = right.snapshot();
    if (left_progress.last_written_sequence == hold_sequence &&
        right_progress.last_written_sequence == hold_sequence) {
      left.disable_writes();
      right.disable_writes();
      return {true, "write_acknowledged"};
    }
    if (!left_progress.healthy || !right_progress.healthy ||
        left_progress.total_failures != left_state.total_failures ||
        right_progress.total_failures != right_state.total_failures) {
      left.disable_writes();
      right.disable_writes();
      return {false, "serial_became_unhealthy"};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  left.disable_writes();
  right.disable_writes();
  return {false, "write_ack_timeout"};
}

bool clear_hardware_errors(HandSerialWorker& left, HandSerialWorker& right) {
  left.disable_writes();
  right.disable_writes();
  const auto left_before = left.snapshot();
  const auto right_before = right.snapshot();
  const auto left_generation = left.request_hardware_error_clear();
  const auto right_generation = right.request_hardware_error_clear();
  const auto deadline = Clock::now() + kSafetyHoldWriteTimeout;
  while (Clock::now() < deadline) {
    const auto left_progress = left.snapshot();
    const auto right_progress = right.snapshot();
    if (left_progress.clear_error_completed_generation == left_generation &&
        right_progress.clear_error_completed_generation == right_generation) {
      return true;
    }
    if (left_progress.total_failures != left_before.total_failures ||
        right_progress.total_failures != right_before.total_failures) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

enum class CommandMode { HoldCurrent, Native6 };

const char* command_mode_name(CommandMode mode) {
  return mode == CommandMode::Native6 ? "native6-trajectory" : "hold-current";
}

std::string read_operator_line(std::string& pending) {
  std::array<char, 128> buffer{};
  const auto count = ::read(STDIN_FILENO, buffer.data(), buffer.size());
  if (count > 0) pending.append(buffer.data(), static_cast<std::size_t>(count));
  const auto newline = pending.find('\n');
  if (newline == std::string::npos) return {};
  std::string line = pending.substr(0, newline);
  pending.erase(0, newline + 1);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return line;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 11) {
      throw std::runtime_error(
          "usage: inspire_direct_serial_hand_gateway RIGHT_DEVICE LEFT_DEVICE "
          "[COMMAND_ENDPOINT=tcp://127.0.0.1:5561] "
          "[STATE_ENDPOINT=tcp://127.0.0.1:5562] "
          "[RUNTIME_SPEED=1000] [RATE_HZ=50] [LOCK_PATH=/tmp/inspire-direct-serial-hand-gateway.writer.lock] "
          "[BAUD=57600] [RIGHT_ID=1] [LEFT_ID=1]");
    }
    const std::string right_device = argv[1];
    const std::string left_device = argv[2];
    const std::string command_endpoint = argc >= 4 ? argv[3] : "tcp://127.0.0.1:5561";
    const std::string state_endpoint = argc >= 5 ? argv[4] : "tcp://127.0.0.1:5562";
    const int parsed_runtime_speed = argc >= 6 ? std::stoi(argv[5]) : 1000;
    if (parsed_runtime_speed < 1 ||
        parsed_runtime_speed > inspire::serial::kMaximumSpeedSetting) {
      throw std::runtime_error("RUNTIME_SPEED must be in [1, 1000]");
    }
    const auto runtime_speed = static_cast<std::uint16_t>(parsed_runtime_speed);
    const double rate_hz = argc >= 7 ? std::stod(argv[6]) : 50.0;
    if (!std::isfinite(rate_hz) || rate_hz < 1 || rate_hz > 100)
      throw std::runtime_error("RATE_HZ must be finite and in [1, 100]");
    const auto serial_period = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(1.0 / rate_hz));
    const std::string lock_path = argc >= 8 ? argv[7] : kDefaultWriterLockPath;
    const int baudrate = argc >= 9 ? std::stoi(argv[8]) : 57600;
    speed_t baud;
    switch (baudrate) {
      case 9600: baud = B9600; break;
      case 19200: baud = B19200; break;
      case 38400: baud = B38400; break;
      case 57600: baud = B57600; break;
      case 115200: baud = B115200; break;
      default: throw std::runtime_error("unsupported BAUD");
    }
    const int right_id = argc >= 10 ? std::stoi(argv[9]) : 1;
    const int left_id = argc >= 11 ? std::stoi(argv[10]) : 1;
    if (right_id < 1 || right_id > 254 || left_id < 1 || left_id > 254)
      throw std::runtime_error("hand IDs must be in [1, 254]");
    if (right_device == left_device)
      throw std::runtime_error("left and right serial devices must differ");


    std::signal(SIGINT, stop_signal);
    std::signal(SIGTERM, stop_signal);
    require_unowned(right_device);
    require_unowned(left_device);
    SingleWriterLock writer_lock(lock_path);

    const int original_stdin_flags = ::fcntl(STDIN_FILENO, F_GETFL, 0);
    if (original_stdin_flags >= 0) {
      ::fcntl(STDIN_FILENO, F_SETFL, original_stdin_flags | O_NONBLOCK);
    }

    HandSerialWorker right("right", right_device, runtime_speed, baud, right_id, serial_period);
    HandSerialWorker left("left", left_device, runtime_speed, baud, left_id, serial_period);

    zmq::context_t context(1);
    zmq::socket_t command_socket(context, zmq::socket_type::pull);
    zmq::socket_t state_socket(context, zmq::socket_type::pub);
    command_socket.set(zmq::sockopt::rcvtimeo, 5);
    command_socket.set(zmq::sockopt::linger, 0);
    command_socket.set(zmq::sockopt::rcvhwm, 2);
    state_socket.set(zmq::sockopt::linger, 0);
    command_socket.bind(command_endpoint);
    state_socket.bind(state_endpoint);

    inspire::SafetyGuard guard({
        // Direct-serial operator mode remains armed across command transport
        // gaps.  The operator owns shutdown through Ctrl+C/DISARM/QUIT; the
        // independent serial-health and command-contract checks remain active.
        .command_timeout_ns = 0,
        .max_future_skew_ns = 0,
        .validate_source_clock_age = false,
    });
    std::optional<inspire::CommandV1> queued;
    std::optional<inspire::CommandV1> in_flight;
    std::optional<inspire::CommandV1> accepted;
    std::uint64_t state_sequence = 0;
    std::uint64_t malformed = 0;
    std::uint64_t rejected = 0;
    std::string operator_input;
    // Bind readiness to the process launched by deploy.sh, not an existing publisher.
    const char* launch_instance = std::getenv("G1_HAND_INSTANCE_ID");
    inspire::OperatorSession operator_session(
        inspire::OperatorSession::enabled_from_environment(), launch_instance ? launch_instance : "");
    auto next_state_publish = Clock::now();
    bool state_ready_logged = false;
    CommandMode command_mode = CommandMode::HoldCurrent;

    std::cout << "DIRECT_SERIAL_HAND_GATEWAY right_device=" << right_device
              << " left_device=" << left_device
              << " baud=" << baudrate << " serial_rate_hz=" << rate_hz
              << " runtime_speed=" << runtime_speed
              << " command_endpoint=" << command_endpoint
              << " state_endpoint=" << state_endpoint
              << " transport=direct_serial"
              << " mode=operator-selected"
              << " command_watchdog=disabled"
              << " hold_tolerance_rad=[0.025,0.025,0.025,0.025,0.025,0.050]"
              << " initial_native6_tolerance_rad=[0.100,0.100,0.100,0.100,0.050,0.050]"
              << " status=DISARMED" << std::endl;
    std::cout << "LOCAL_OPERATOR_COMMANDS: ARM HAND | ARM HAND NATIVE6 | "
                 "DISARM HAND | RESET HAND | CLEAR HAND HARDWARE FAULTS | QUIT"
              << std::endl;

    while (running) {
      auto left_state = left.snapshot();
      auto right_state = right.snapshot();
      const auto now = monotonic_ns();
      const bool hardware_healthy =
          fresh_and_healthy(left_state, now) && fresh_and_healthy(right_state, now);

      if (hardware_healthy && !state_ready_logged) {
        std::cout << "DIRECT_SERIAL_STATE_READY left_generation="
                  << left_state.state_generation << " right_generation="
                  << right_state.state_generation << std::endl;
        state_ready_logged = true;
      }

      operator_session.observe(std::string(inspire::to_string(guard.status())));
      const std::string operator_line = read_operator_line(operator_input);
      if (!operator_line.empty()) {
        if (operator_line == "ARM HAND") {
          if (hardware_healthy && guard.status() == inspire::Status::Disarmed && guard.arm()) {
            command_mode = CommandMode::HoldCurrent;
            std::cout << "HAND_GATEWAY_ARMED local_operator=true backend=direct_serial "
                         "mode=hold-current"
                      << std::endl;
          } else {
            std::cout << "HAND_GATEWAY_ARM_REJECTED status="
                      << inspire::to_string(guard.status())
                      << " state_healthy=" << (hardware_healthy ? "true" : "false")
                      << std::endl;
          }
        } else if (operator_line == "ARM HAND NATIVE6") {
          if (hardware_healthy && guard.status() == inspire::Status::Disarmed && guard.arm()) {
            command_mode = CommandMode::Native6;
            std::cout << "HAND_GATEWAY_ARMED local_operator=true backend=direct_serial "
                         "mode=native6-trajectory"
                      << std::endl;
          } else {
            std::cout << "HAND_GATEWAY_ARM_REJECTED status="
                      << inspire::to_string(guard.status())
                      << " state_healthy=" << (hardware_healthy ? "true" : "false")
                      << std::endl;
          }
        } else if (operator_line == "DISARM HAND") {
          queued.reset();
          in_flight.reset();
          SafetyHoldResult hold{false, "status_not_armed"};
          if (guard.status() == inspire::Status::Armed) {
            hold = hold_current_and_disable(left, right);
          } else {
            left.disable_writes();
            right.disable_writes();
          }
          guard.disarm();
          command_mode = CommandMode::HoldCurrent;
          std::cout << "HAND_GATEWAY_DISARMED disarm_hold_current_sent="
                    << (hold.sent ? "true" : "false")
                    << " reason=" << hold.reason << std::endl;
        } else if (operator_line == "RESET HAND") {
          if (hardware_healthy && guard.status() != inspire::Status::Armed) {
            left.reset_command_session();
            right.reset_command_session();
            queued.reset();
            in_flight.reset();
            guard.clear_fault();
            command_mode = CommandMode::HoldCurrent;
            // RESET starts a new producer session.  Do not expose the prior
            // session's accepted command as the new session's sequence/pose
            // seed.  DISARM intentionally keeps it for post-run audit.
            accepted.reset();
            operator_session.manual_reset();
            std::cout << "HAND_GATEWAY_RESET status=DISARMED" << std::endl;
          } else {
            std::cout << "HAND_GATEWAY_RESET_REJECTED state_healthy="
                      << (hardware_healthy ? "true" : "false") << std::endl;
          }
        } else if (operator_line == "CLEAR HAND HARDWARE FAULTS") {
          if (hardware_healthy && guard.status() == inspire::Status::Disarmed) {
            const bool cleared = clear_hardware_errors(left, right);
            std::cout << "HAND_GATEWAY_HARDWARE_FAULT_CLEAR success="
                      << (cleared ? "true" : "false")
                      << " status=DISARMED" << std::endl;
          } else {
            std::cout << "HAND_GATEWAY_HARDWARE_FAULT_CLEAR_REJECTED status="
                      << inspire::to_string(guard.status())
                      << " state_healthy=" << (hardware_healthy ? "true" : "false")
                      << std::endl;
          }
        } else if (operator_line == "QUIT") {
          running = 0;
        } else {
          std::cout << "UNKNOWN_LOCAL_OPERATOR_COMMAND" << std::endl;
        }
      }
      if (!running) break;

      if (!hardware_healthy && guard.status() == inspire::Status::Armed) {
        left.disable_writes();
        right.disable_writes();
        queued.reset();
        in_flight.reset();
        guard.fault("direct serial state became unhealthy");
        std::cerr << "HAND_GATEWAY_FAULT reason=direct_serial_unhealthy"
                  << " left_error='" << left_state.last_error << "'"
                  << " right_error='" << right_state.last_error << "'"
                  << " fault_response_write_sent=false"
                  << " left_last_written_sequence="
                  << (left_state.last_written_sequence
                          ? std::to_string(*left_state.last_written_sequence)
                          : "none")
                  << " right_last_written_sequence="
                  << (right_state.last_written_sequence
                          ? std::to_string(*right_state.last_written_sequence)
                          : "none")
                  << std::endl;
      }

      zmq::message_t packet;
      bool received = false;
      try {
        received = command_socket.recv(packet, zmq::recv_flags::none).has_value();
      } catch (const zmq::error_t&) {
        if (!running) break;
        throw;
      }
      if (received) {
        try {
          const bool operator_packet = operator_session.consume(
              static_cast<const char*>(packet.data()), packet.size(),
              std::string(inspire::to_string(guard.status())), hardware_healthy,
              [&](const std::string& operation) {
                queued.reset();
                in_flight.reset();
                if (operation == "PREPARE") {
                  left.reset_command_session();
                  right.reset_command_session();
                  // OperatorSession already required healthy DISARMED without a fault.
                  guard.clear_fault();
                  accepted.reset();
                  command_mode = CommandMode::Native6;
                  return guard.arm();
                }
                if (guard.status() == inspire::Status::Armed) {
                  const auto hold = hold_current_and_disable(left, right);
                  if (!hold.sent) {
                    guard.fault("operator disarm hold failed: " + hold.reason);
                    return false;
                  }
                } else {
                  left.disable_writes();
                  right.disable_writes();
                }
                if (guard.status() == inspire::Status::Fault ||
                    guard.status() == inspire::Status::Stale) return false;
                guard.disarm();
                command_mode = CommandMode::HoldCurrent;
                return true;
              });
          if (!operator_packet) {
            const auto command = inspire::unpack_command(
                static_cast<const char*>(packet.data()), packet.size());
            const bool initial_native6_command =
                command_mode == CommandMode::Native6 && !guard.last_command();
            const bool feedback_proximity_required =
                command_mode != CommandMode::Native6 || initial_native6_command;
            const auto proximity_violation =
                feedback_proximity_required
                    ? inspire::feedback_violation(
                          command.left_q,
                          command.right_q,
                          left_state.q,
                          right_state.q,
                          command_mode == CommandMode::Native6
                              ? kInitialNative6ToleranceRad
                              : kHoldCurrentToleranceRad)
                    : std::nullopt;
            if (!hardware_healthy) {
              ++rejected;
            } else if (proximity_violation) {
              const auto& violation = *proximity_violation;
              ++rejected;
              left.disable_writes();
              right.disable_writes();
              queued.reset();
              in_flight.reset();
              guard.fault("feedback tracking displacement exceeded its per-joint tolerance");
              std::cerr << "HAND_GATEWAY_FAULT reason=feedback_gate_exceeded"
                        << " mode=" << command_mode_name(command_mode)
                        << " phase=" << (initial_native6_command ? "initial" : "hold-current")
                        << " command_sequence=" << command.sequence
                        << " side=" << (violation.left ? "left" : "right")
                        << " index=" << violation.index
                        << " delta_rad=" << violation.delta
                        << " limit_rad=" << violation.limit << ' '
                        << "fault_response_write_sent=false"
                        << std::endl;
            } else if (!guard.accept(command, monotonic_ns())) {
              ++rejected;
              left.disable_writes();
              right.disable_writes();
              queued.reset();
              in_flight.reset();
              std::cerr << "HAND_GATEWAY_FAULT reason='" << guard.last_error()
                        << "' fault_response_write_sent=false" << std::endl;
            } else {
              queued = command;
            }
          }
        } catch (const std::exception& error) {
          ++malformed;
          left.disable_writes();
          right.disable_writes();
          queued.reset();
          in_flight.reset();
          guard.fault("malformed command packet");
          std::cerr << "HAND_GATEWAY_FAULT reason=malformed_command detail='"
                    << error.what() << "' fault_response_write_sent=false" << std::endl;
        }
      }

      left_state = left.snapshot();
      right_state = right.snapshot();
      if (in_flight && left_state.last_written_sequence == in_flight->sequence &&
          right_state.last_written_sequence == in_flight->sequence) {
        accepted = in_flight;
        in_flight.reset();
      }

      if (guard.status() == inspire::Status::Armed && !in_flight && queued) {
        PendingWrite left_write{
            queued->sequence,
            inspire::real::canonical_to_normalized(queued->left_q),
            queued->left_q,
        };
        PendingWrite right_write{
            queued->sequence,
            inspire::real::canonical_to_normalized(queued->right_q),
            queued->right_q,
        };
        left.stage(std::move(left_write));
        right.stage(std::move(right_write));
        in_flight = queued;
        queued.reset();
      }

      operator_session.observe(std::string(inspire::to_string(guard.status())));
      if (Clock::now() >= next_state_publish && left_state.received && right_state.received) {
        inspire::StateV1 state;
        state.sequence = ++state_sequence;
        state.gateway_monotonic_ns = monotonic_ns();
        state.status = hardware_healthy ? guard.status() : inspire::Status::Fault;
        state.left_q = left_state.q;
        state.right_q = right_state.q;
        state.left_dq = left_state.dq;
        state.right_dq = right_state.dq;
        if (accepted) {
          state.source_monotonic_ns = accepted->source_monotonic_ns;
          state.accepted_command_sequence = accepted->sequence;
          state.accepted_frame_index = accepted->frame_index;
          state.left_command_q = accepted->left_q;
          state.right_command_q = accepted->right_q;
        }
        auto encoded = operator_session.state(inspire::pack_state(state));
        state_socket.send(zmq::buffer(encoded.data(), encoded.size()),
                          zmq::send_flags::dontwait);
        next_state_publish = Clock::now() + serial_period;
      }
    }

    SafetyHoldResult exit_hold{false, "status_not_armed"};
    if (guard.status() == inspire::Status::Armed) {
      queued.reset();
      in_flight.reset();
      exit_hold = hold_current_and_disable(left, right);
    } else {
      left.disable_writes();
      right.disable_writes();
    }
    if (original_stdin_flags >= 0) {
      ::fcntl(STDIN_FILENO, F_SETFL, original_stdin_flags);
    }
    const auto left_final = left.snapshot();
    const auto right_final = right.snapshot();
    std::cerr << "inspire_direct_serial_hand_gateway stopped; exit_hold_current_sent="
              << (exit_hold.sent ? "true" : "false")
              << " exit_hold_reason=" << exit_hold.reason
              << " malformed=" << malformed << " rejected=" << rejected
              << " left_total_failures=" << left_final.total_failures
              << " right_total_failures=" << right_final.total_failures << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "inspire_direct_serial_hand_gateway: " << error.what() << std::endl;
    return 1;
  }
}
