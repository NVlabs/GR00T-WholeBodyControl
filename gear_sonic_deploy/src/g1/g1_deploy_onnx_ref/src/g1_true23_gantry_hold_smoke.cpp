// Policy-free, gantry-only first-contact smoke for G1 EDU rev-1.0 True23.
// Sends damping before A, then holds the exact sampled posture for at most
// five seconds using fixed low gains.  No ONNX/PICO/action input exists.

#include "true23_active_gantry_core.hpp"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

namespace true23 = gear_sonic::true23;
namespace active = gear_sonic::true23::active;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

inline constexpr std::string_view kLowStateTopic = "rt/lowstate";
inline constexpr std::string_view kLowCmdTopic = "rt/lowcmd";
inline constexpr auto kMaximumHoldDuration = std::chrono::seconds(5);
inline constexpr auto kMaximumWaitForArm = std::chrono::seconds(60);
inline constexpr int kFaultDampingCycles = 250;
volatile std::sig_atomic_t g_stop_requested = 0;

void HandleSignal(int) { g_stop_requested = 1; }

std::int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::uint32_t Crc32(const void* bytes, std::uint32_t words) {
  const auto* input = static_cast<const std::uint8_t*>(bytes);
  std::uint32_t crc = 0xffffffffU;
  constexpr std::uint32_t polynomial = 0x04c11db7U;
  for (std::uint32_t index = 0; index < words; ++index) {
    std::uint32_t data = 0;
    std::memcpy(&data, input + index * sizeof(data), sizeof(data));
    std::uint32_t bit = 1U << 31U;
    for (std::uint32_t count = 0; count < 32U; ++count) {
      crc = (crc & 0x80000000U) != 0U
                ? (crc << 1U) ^ polynomial
                : crc << 1U;
      if ((data & bit) != 0U) {
        crc ^= polynomial;
      }
      bit >>= 1U;
    }
  }
  return crc;
}

struct Arguments {
  std::string network;
  std::string gantry_authorization;
  bool execute = false;
  bool help = false;
};

std::string Usage(std::string_view executable) {
  return
      "Usage: " + std::string(executable) +
      " --network <interface> --execute-hold-smoke --gantry-authorize " +
      std::string(active::kGantryAuthorizationPhrase) +
      "\n\n"
      "POLICY-FREE GANTRY TEST ONLY. After readiness: hold L2 and press A "
      "once. Controller holds sampled posture for maximum 5 seconds. B/R2, "
      "L2 release, SIGINT, CRC failure, stale/tick-regressed state, or "
      "mode_machine change latches zero-feedforward damping STOP.";
}

Arguments ParseArguments(int argc, char** argv) {
  Arguments result;
  const auto value = [&](int& index, std::string_view option) {
    if (++index >= argc) {
      throw std::runtime_error(std::string(option) + " requires a value");
    }
    return std::string(argv[index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help" || option == "-h") {
      result.help = true;
    } else if (option == "--network") {
      result.network = value(index, option);
    } else if (option == "--gantry-authorize") {
      result.gantry_authorization = value(index, option);
    } else if (option == "--execute-hold-smoke") {
      result.execute = true;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.help) {
    return result;
  }
  if (result.network.empty()) {
    throw std::runtime_error("--network is required");
  }
  if (!result.execute) {
    throw std::runtime_error("--execute-hold-smoke is required");
  }
  if (result.gantry_authorization != active::kGantryAuthorizationPhrase) {
    throw std::runtime_error("exact explicit gantry authorization phrase is required");
  }
  return result;
}

class HoldStateMonitor {
 public:
  ~HoldStateMonitor() {
    if (subscriber_) {
      subscriber_->CloseChannel();
    }
  }

  void Start() {
    subscriber_ =
        std::make_shared<unitree::robot::ChannelSubscriber<LowState>>(
            std::string(kLowStateTopic));
    subscriber_->InitChannel(
        [this](const void* message) { OnMessage(message); }, 1);
  }

  bool WaitForMutationGate(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    return condition_.wait_until(lock, deadline, [&] {
      return core_.mutation_surface_allowed() || core_.stopped() ||
             g_stop_requested != 0;
    }) && core_.mutation_surface_allowed();
  }

  template <typename Function>
  void WithCore(Function&& function) {
    std::lock_guard lock(mutex_);
    std::forward<Function>(function)(core_);
  }

  [[nodiscard]] active::Fault fault() const {
    std::lock_guard lock(mutex_);
    return core_.fault();
  }

 private:
  void OnMessage(const void* message) {
    if (message == nullptr) {
      return;
    }
    LowState state = *static_cast<const LowState*>(message);
    const auto now = NowNs();
    std::lock_guard lock(mutex_);
    if (state.crc() != Crc32(&state, (sizeof(LowState) >> 2U) - 1U)) {
      core_.ObserveCrcFailure();
      condition_.notify_all();
      return;
    }
    active::StateSample sample;
    sample.tick = state.tick();
    sample.mode_machine = state.mode_machine();
    sample.crc_valid = true;
    sample.received_monotonic_ns = now;
    for (std::size_t slot = 0; slot < active::kMotorSlotCount; ++slot) {
      const auto& motor = state.motor_state()[slot];
      sample.q[slot] = motor.q();
      sample.dq[slot] = motor.dq();
      sample.tau_est[slot] = motor.tau_est();
    }
    core_.ObserveState(sample, now);

    std::array<std::uint8_t, 40> remote{};
    std::copy_n(state.wireless_remote().begin(), remote.size(), remote.begin());
    const auto buttons = active::DecodeWirelessOperator(remote);
    const bool arm_edge = buttons.arm_pressed && !arm_pressed_;
    arm_pressed_ = buttons.arm_pressed;
    core_.ObserveOperator(
        {.arm_edge = arm_edge,
         .deadman_held = buttons.deadman_held,
         .stop_requested = buttons.stop_pressed},
        now);
    condition_.notify_all();
  }

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  active::HoldSmokeSafetyCore core_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  bool arm_pressed_ = false;
};

void ReleaseMotionModeAfterGate() {
  auto client =
      std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
  client->SetTimeout(3.0F);
  client->Init();
  std::string form;
  std::string name;
  if (client->CheckMode(form, name) != 0) {
    throw std::runtime_error("motion-mode pre-release CheckMode RPC failed");
  }
  if (!name.empty() && client->ReleaseMode() != 0) {
    throw std::runtime_error("motion-mode release failed");
  }
  name.clear();
  if (client->CheckMode(form, name) != 0) {
    throw std::runtime_error("motion-mode post-release CheckMode RPC failed");
  }
  if (!name.empty()) {
    throw std::runtime_error("motion mode remains active after release");
  }
}

LowCmd ToLowCmd(const active::MotorCommand& command) {
  LowCmd result;
  result.mode_pr() = 0;
  result.mode_machine() = true23::kRequiredModeMachine;
  for (std::size_t slot = 0; slot < command.size(); ++slot) {
    auto& output = result.motor_cmd().at(slot);
    output.mode() = command[slot].mode;
    output.q() = static_cast<float>(command[slot].q);
    output.dq() = static_cast<float>(command[slot].dq);
    output.kp() = static_cast<float>(command[slot].kp);
    output.kd() = static_cast<float>(command[slot].kd);
    output.tau() = static_cast<float>(command[slot].tau);
  }
  result.crc() = Crc32(&result, (sizeof(LowCmd) >> 2U) - 1U);
  return result;
}

int Run(const Arguments& arguments) {
  unitree::robot::ChannelFactory::Instance()->Init(0, arguments.network);
  HoldStateMonitor monitor;
  monitor.Start();
  if (!monitor.WaitForMutationGate(
          std::chrono::steady_clock::now() + std::chrono::seconds(8))) {
    throw std::runtime_error(
        "five advancing CRC-valid mode_machine==4 states were not obtained");
  }

  // Neither object exists before five-state gate above.
  ReleaseMotionModeAfterGate();
  monitor.WithCore([](active::HoldSmokeSafetyCore& core) {
    core.CheckWatchdog(NowNs());
  });
  if (monitor.fault() != active::Fault::None) {
    throw std::runtime_error("state fault occurred before publisher creation");
  }
  auto publisher =
      std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(
          std::string(kLowCmdTopic));
  publisher->InitChannel();
  monitor.WithCore([](active::HoldSmokeSafetyCore& core) {
    core.EnableOperatorArming();
  });

  std::atomic<bool> finished{false};
  std::jthread writer([&](std::stop_token stop_token) {
    try {
      auto wake = std::chrono::steady_clock::now();
      const auto wait_deadline = wake + kMaximumWaitForArm;
      std::optional<std::chrono::steady_clock::time_point> armed_at;
      int damping_cycles = 0;
      while (!stop_token.stop_requested() && !finished.load()) {
        wake += std::chrono::microseconds(2000);
        active::MotorCommand command;
        active::Fault fault = active::Fault::None;
        bool armed = false;
        monitor.WithCore([&](active::HoldSmokeSafetyCore& core) {
          if (g_stop_requested != 0) {
            core.Stop();
          }
          command = core.BuildCommand(NowNs());
          fault = core.fault();
          armed = core.armed();
        });
        publisher->Write(ToLowCmd(command));
        const auto now = std::chrono::steady_clock::now();
        if (armed && !armed_at.has_value()) {
          armed_at = now;
          std::cout
              << "[ARMED] L2+A accepted; holding the sampled posture for at "
                 "most 5 seconds.\n"
              << std::flush;
        }
        if (armed_at.has_value() &&
            now - *armed_at >= kMaximumHoldDuration) {
          monitor.WithCore(
              [](active::HoldSmokeSafetyCore& core) { core.Stop(); });
        } else if (!armed_at.has_value() && now >= wait_deadline) {
          std::cout << "[TIMEOUT] No L2+A arm edge received.\n" << std::flush;
          monitor.WithCore(
              [](active::HoldSmokeSafetyCore& core) { core.Stop(); });
        }
        if (fault != active::Fault::None &&
            ++damping_cycles >= kFaultDampingCycles) {
          finished.store(true);
          break;
        }
        std::this_thread::sleep_until(wake);
      }
    } catch (const std::exception&) {
      monitor.WithCore([](active::HoldSmokeSafetyCore& core) {
        core.ObserveInternalFailure();
      });
      finished.store(true);
    }
  });

  std::cout
      << "[READY] Policy-free True23 damping publisher active. Gantry secure; "
         "hold L2 then press A once for maximum 5-second sampled-posture hold. "
         "B/R2 is STOP.\n";
  while (!finished.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  writer.request_stop();
  const auto fault = monitor.fault();
  if (fault != active::Fault::OperatorStop) {
    std::cerr << "[STOPPED] latched fault: " << active::FaultName(fault) << '\n';
    return 2;
  }
  std::cout << "[COMPLETE] policy-free gantry hold smoke stopped safely.\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto arguments = ParseArguments(argc, argv);
    if (arguments.help) {
      std::cout << Usage(argv[0]) << '\n';
      return 0;
    }
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    return Run(arguments);
  } catch (const std::exception& error) {
    std::cerr << "[BLOCKED] true23 gantry hold smoke: " << error.what()
              << '\n';
    if (argc > 0) {
      std::cerr << Usage(argv[0]) << '\n';
    }
    return 1;
  }
}
