// Native G1 EDU rev-1.0 True23 first-actuation controller.
//
// Safety ordering is deliberate: immutable artifacts -> ONNX dry run -> DDS
// LowState subscriber -> five advancing CRC-valid mode_machine==4 samples ->
// motion-mode release -> LowCmd publisher.  Any fault is one-way and leaves
// only a finite, zero-feedforward damping command.

#include "true23_active_gantry_core.hpp"

#include <onnxruntime_cxx_api.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <zmq.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace true23 = gear_sonic::true23;
namespace active = gear_sonic::true23::active;
namespace live = gear_sonic::true23::live;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using nlohmann::json;

inline constexpr std::string_view kLowStateTopic = "rt/lowstate";
inline constexpr std::string_view kLowCmdTopic = "rt/lowcmd";
inline constexpr int kStateGateTimeoutSeconds = 8;
inline constexpr int kFaultDampingCycles = 250;
[[gnu::used]] const char kCompiledCausalRuntimeSurface[] =
    "TryJoinCausalReference BuildCausalEncoderInput";
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
  fs::path encoder;
  fs::path decoder;
  fs::path metadata;
  fs::path promotion;
  fs::path active_promotion;
  fs::path live_shadow_evidence;
  fs::path execution_evidence;
  std::string pico_endpoint;
  std::string authorization_id;
  std::string gantry_authorization;
  int post_arm_duration_seconds = 0;
  bool validate_only = false;
  bool execute_stage_one = false;
  bool help = false;
};

std::string Usage(std::string_view executable) {
  return
      "Usage: " + std::string(executable) +
      " --network <interface> --encoder <pair.encoder.onnx>"
      " --decoder <pair.decoder.onnx> --metadata <candidate.json>"
      " --promotion <promotion.json>"
      " --active-promotion <gantry-active-promotion.json>"
      " --live-shadow-evidence <passed-shadow.jsonl>"
      " --authorization-id <session-id>"
      " [--validate-only | --network <interface>"
      " --pico-endpoint <tcp://host:port> --execute-stage-one"
      " --evidence <new.jsonl> --post-arm-duration-seconds <20..30>"
      " --gantry-authorize " +
      std::string(active::kGantryAuthorizationPhrase) +
      "]"
      "\n\n"
      "Wireless operator contract: hold L2 deadman, press A once to ARM; "
      "B or R2 is STOP. Stage one is gantry-only. Diagnostic ONNX, generic "
      "29-output policy, non-mode-4 robot, stale input, and CRC bypass are "
      "unconditionally rejected. Causal artifacts require "
      "g1_true23_causal_history_reference_terms; future-command packets are "
      "rejected.";
}

Arguments ParseArguments(int argc, char** argv) {
  Arguments result;
  std::unordered_set<std::string> seen_options;
  const auto value = [&](int& index, std::string_view option) {
    if (++index >= argc) {
      throw std::runtime_error(std::string(option) + " requires a value");
    }
    return std::string(argv[index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (!seen_options.insert(option).second) {
      throw std::runtime_error("duplicate option rejected: " + option);
    }
    if (option == "--help" || option == "-h") {
      result.help = true;
    } else if (option == "--network") {
      result.network = value(index, option);
    } else if (option == "--encoder") {
      result.encoder = value(index, option);
    } else if (option == "--decoder") {
      result.decoder = value(index, option);
    } else if (option == "--metadata") {
      result.metadata = value(index, option);
    } else if (option == "--promotion") {
      result.promotion = value(index, option);
    } else if (option == "--active-promotion") {
      result.active_promotion = value(index, option);
    } else if (option == "--live-shadow-evidence") {
      result.live_shadow_evidence = value(index, option);
    } else if (option == "--evidence") {
      result.execution_evidence = value(index, option);
    } else if (option == "--pico-endpoint") {
      result.pico_endpoint = value(index, option);
    } else if (option == "--authorization-id") {
      result.authorization_id = value(index, option);
    } else if (option == "--gantry-authorize") {
      result.gantry_authorization = value(index, option);
    } else if (option == "--post-arm-duration-seconds") {
      const auto duration = value(index, option);
      std::size_t parsed = 0;
      try {
        result.post_arm_duration_seconds = std::stoi(duration, &parsed);
      } catch (const std::exception&) {
        throw std::runtime_error(
            "--post-arm-duration-seconds must be an integer");
      }
      if (parsed != duration.size()) {
        throw std::runtime_error(
            "--post-arm-duration-seconds must be an integer");
      }
    } else if (option == "--validate-only") {
      result.validate_only = true;
    } else if (option == "--execute-stage-one") {
      result.execute_stage_one = true;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.help) {
    return result;
  }
  if (result.encoder.empty() || result.decoder.empty() ||
      result.metadata.empty() ||
      result.promotion.empty() || result.active_promotion.empty() ||
      result.live_shadow_evidence.empty() ||
      !active::IsSafeAuthorizationId(result.authorization_id)) {
    throw std::runtime_error("all artifact and evidence arguments are required");
  }
  if (result.validate_only) {
    if (result.execute_stage_one || !result.gantry_authorization.empty() ||
        !result.execution_evidence.empty() ||
        result.post_arm_duration_seconds != 0) {
      throw std::runtime_error(
          "--validate-only rejects execution, evidence, duration, and "
          "authorization arguments");
    }
  }
  if (result.network.empty() || result.pico_endpoint.empty()) {
    throw std::runtime_error(
        "network and PICO bindings are required for validation/execution");
  }
  if (result.validate_only) {
    if (!result.pico_endpoint.starts_with("tcp://") &&
        !result.pico_endpoint.starts_with("ipc://")) {
      throw std::runtime_error(
          "PICO endpoint must use explicit tcp:// or ipc:// URI");
    }
    return result;
  }
  if (!result.execute_stage_one) {
    throw std::runtime_error("--execute-stage-one is required");
  }
  if (result.gantry_authorization != active::kGantryAuthorizationPhrase) {
    throw std::runtime_error("exact explicit gantry authorization phrase is required");
  }
  if (result.execution_evidence.empty()) {
    throw std::runtime_error("--evidence is required for stage-one execution");
  }
  if (result.post_arm_duration_seconds <
          active::kMinimumStageOnePostArmSeconds ||
      result.post_arm_duration_seconds >
          active::kMaximumStageOnePostArmSeconds) {
    throw std::runtime_error(
        "--post-arm-duration-seconds must be within reviewed 20..30 range");
  }
  if (!result.pico_endpoint.starts_with("tcp://") &&
      !result.pico_endpoint.starts_with("ipc://")) {
    throw std::runtime_error("PICO endpoint must use explicit tcp:// or ipc:// URI");
  }
  return result;
}

fs::path ResolveFile(const fs::path& argument, std::string_view role) {
  std::error_code error;
  const auto path = fs::canonical(argument, error);
  if (error || !fs::is_regular_file(path, error)) {
    throw std::runtime_error(
        std::string(role) + " is not a resolvable regular file: " +
        argument.string());
  }
  return path;
}

json ParseStrictJson(std::string_view bytes, std::string_view role) {
  std::vector<std::unordered_set<std::string>> keys;
  const json::parser_callback_t callback =
      [&](int, json::parse_event_t event, json& parsed) {
        if (event == json::parse_event_t::object_start) {
          keys.emplace_back();
        } else if (event == json::parse_event_t::key) {
          if (keys.empty()) {
            throw std::runtime_error(std::string(role) + " parser scope error");
          }
          const auto key = parsed.get<std::string>();
          if (!keys.back().insert(key).second) {
            throw std::runtime_error(
                std::string(role) + " contains duplicate key: " + key);
          }
        } else if (event == json::parse_event_t::object_end) {
          if (keys.empty()) {
            throw std::runtime_error(std::string(role) + " parser underflow");
          }
          keys.pop_back();
        }
        return true;
      };
  auto result = json::parse(bytes, callback, true, false);
  if (!result.is_object()) {
    throw std::runtime_error(std::string(role) + " root must be object");
  }
  return result;
}

json LoadJson(const fs::path& path, std::string_view role) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + std::string(role));
  }
  const std::string bytes{
      std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>()};
  return ParseStrictJson(bytes, role);
}

std::string LoadBoundedBytes(
    const fs::path& path,
    std::size_t maximum_bytes,
    std::string_view role) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  if (error || size == 0 || size > maximum_bytes) {
    throw std::runtime_error(
        std::string(role) + " is empty, unreadable, or oversized");
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + std::string(role));
  }
  std::string bytes(static_cast<std::size_t>(size), '\0');
  stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error(
        std::string(role) + " changed or failed during bounded read");
  }
  return bytes;
}

Ort::SessionOptions SessionOptions() {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  return options;
}

class OnnxModel {
 public:
  OnnxModel(Ort::Env& environment, fs::path path)
      : path_(std::move(path)),
        options_(SessionOptions()),
        session_(environment, path_.c_str(), options_) {
    Inspect();
  }

  [[nodiscard]] const true23::ModelSignature& signature() const {
    return signature_;
  }
  [[nodiscard]] const json& embedded() const { return embedded_; }

  template <std::size_t Input, std::size_t Output>
  std::array<float, Output> Run(const std::array<float, Input>& input) {
    std::array<float, Output> result{};
    auto memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    auto shape = signature_.input_shape;
    auto tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(input.data()), input.size(),
        shape.data(), shape.size());
    const char* inputs[] = {signature_.input_name.c_str()};
    const char* outputs[] = {signature_.output_name.c_str()};
    auto values = session_.Run(
        Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    if (values.size() != 1 || !values.front().IsTensor()) {
      throw std::runtime_error("ONNX inference returned invalid output count/type");
    }
    const auto info = values.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != Output) {
      throw std::runtime_error("ONNX inference output shape/type changed");
    }
    const auto* data = values.front().template GetTensorData<float>();
    std::copy_n(data, Output, result.begin());
    if (!live::IsFinite(result)) {
      throw std::runtime_error("ONNX inference produced non-finite output");
    }
    return result;
  }

 private:
  void Inspect() {
    Ort::AllocatorWithDefaultOptions allocator;
    signature_.input_count = session_.GetInputCount();
    signature_.output_count = session_.GetOutputCount();
    if (signature_.input_count == 1) {
      auto name = session_.GetInputNameAllocated(0, allocator);
      signature_.input_name = name ? name.get() : "";
      const auto type_info = session_.GetInputTypeInfo(0);
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      signature_.input_shape = info.GetShape();
      signature_.input_float32 =
          info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    if (signature_.output_count == 1) {
      auto name = session_.GetOutputNameAllocated(0, allocator);
      signature_.output_name = name ? name.get() : "";
      const auto type_info = session_.GetOutputTypeInfo(0);
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      signature_.output_shape = info.GetShape();
      signature_.output_float32 =
          info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    signature_.default_opsets =
        true23::InspectDefaultOnnxOpsetsFile(path_.string());
    auto metadata = session_.GetModelMetadata();
    auto metadata_keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    if (metadata_keys.size() != 1 || !metadata_keys.front() ||
        std::string_view(metadata_keys.front().get()) !=
            true23::kOnnxMetadataKey) {
      throw std::runtime_error("ONNX embedded metadata key set is not exact");
    }
    auto value = metadata.LookupCustomMetadataMapAllocated(
        std::string(true23::kOnnxMetadataKey).c_str(), allocator);
    if (!value) {
      throw std::runtime_error("ONNX embedded metadata is missing");
    }
    embedded_ = ParseStrictJson(value.get(), "embedded ONNX metadata");
  }

  fs::path path_;
  Ort::SessionOptions options_;
  Ort::Session session_;
  true23::ModelSignature signature_;
  json embedded_;
};

template <std::size_t Size>
std::array<float, Size> JsonFloatArray(
    const json& root, std::string_view key) {
  const auto iterator = root.find(std::string(key));
  if (iterator == root.end() || !iterator->is_array() ||
      iterator->size() != Size) {
    throw std::runtime_error(
        "PICO semantic field has wrong size/type: " + std::string(key));
  }
  std::array<float, Size> result{};
  for (std::size_t index = 0; index < Size; ++index) {
    result[index] = iterator->at(index).get<float>();
  }
  return result;
}

live::PicoEncoderTerms ParsePicoTerms(std::string_view message) {
  const auto root = ParseStrictJson(message, "PICO semantic terms");
  live::PicoEncoderTerms result;
  result.schema_version = root.at("schema_version").get<int>();
  result.kind = root.at("kind").get<std::string>();
  result.source_frame_index = root.at("source_frame_index").get<std::uint64_t>();
  result.source_monotonic_ns = root.at("source_monotonic_ns").get<std::int64_t>();
  const auto offsets = root.at("future_frame_offsets_s");
  if (!offsets.is_array() || offsets.size() != result.future_frame_offsets_s.size()) {
    throw std::runtime_error("PICO future_frame_offsets_s shape mismatch");
  }
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    result.future_frame_offsets_s[index] = offsets[index].get<double>();
  }
  result.command_multi_future_lower_body =
      JsonFloatArray<live::kCommandDim>(root, "command_multi_future_lower_body");
  result.vr_3point_local_target =
      JsonFloatArray<live::kVrPositionDim>(root, "vr_3point_local_target");
  result.vr_3point_local_orn_target =
      JsonFloatArray<live::kVrOrientationDim>(root, "vr_3point_local_orn_target");
  result.motion_anchor_ori_b =
      JsonFloatArray<live::kAnchorOrientationDim>(root, "motion_anchor_ori_b");
  const auto errors = live::ValidatePicoEncoderTerms(result);
  if (!errors.empty()) {
    throw std::runtime_error("PICO semantic terms rejected: " + errors.front());
  }
  return result;
}

live::CausalPicoReferenceTerms ParseCausalPicoReferenceTerms(
    std::string_view message) {
  const auto root = ParseStrictJson(message, "causal PICO reference terms");
  return live::ParseCausalPicoReferenceTermsDocument(root);
}

enum class CausalJoinStatus {
  Ready,
  AwaitingCoverage,
  InvalidCoveredRange,
};

struct CausalJoinAttempt {
  CausalJoinStatus status = CausalJoinStatus::AwaitingCoverage;
  std::optional<live::CausalEncoderJoin> joined;
};

class StateMonitor {
 public:
  explicit StateMonitor(active::GantrySafetyCore& core) : core_(core) {}
  ~StateMonitor() {
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

  std::optional<live::TimedProprioSample> LatestProprio() const {
    std::lock_guard lock(mutex_);
    return proprio_;
  }

  CausalJoinAttempt TryJoinCausalReference(
      const live::CausalPicoReferenceTerms& reference) const {
    std::lock_guard lock(mutex_);
    if (!causal_history_.Covers(
            reference.pico_anchor_monotonic_ns,
            reference.control_monotonic_ns)) {
      return {};
    }
    try {
      return {
          .status = CausalJoinStatus::Ready,
          .joined = live::JoinCausalReferenceWithLowState(
              reference, causal_history_),
      };
    } catch (const std::invalid_argument&) {
      return {.status = CausalJoinStatus::InvalidCoveredRange};
    }
  }

  void WithCore(const auto& function) {
    std::lock_guard lock(mutex_);
    function(core_);
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
    const auto received = NowNs();
    const auto valid_crc =
        state.crc() == Crc32(&state, (sizeof(LowState) >> 2U) - 1U);
    std::lock_guard lock(mutex_);
    if (!valid_crc) {
      core_.ObserveCrcFailure();
      condition_.notify_all();
      return;
    }

    active::StateSample sample;
    sample.tick = state.tick();
    sample.mode_machine = state.mode_machine();
    sample.crc_valid = true;
    sample.received_monotonic_ns = received;
    for (std::size_t slot = 0; slot < active::kMotorSlotCount; ++slot) {
      const auto& motor = state.motor_state()[slot];
      sample.q[slot] = motor.q();
      sample.dq[slot] = motor.dq();
      sample.tau_est[slot] = motor.tau_est();
    }
    core_.ObserveState(sample, received);

    live::TimedProprioSample proprio;
    proprio.received_monotonic_ns = received;
    for (std::size_t compact = 0; compact < 23; ++compact) {
      const auto slot = static_cast<std::size_t>(
          true23::kHardwareJointIds[compact]);
      proprio.hardware_q[compact] = state.motor_state()[slot].q();
      proprio.hardware_dq[compact] = state.motor_state()[slot].dq();
    }
    const auto& imu = state.imu_state();
    std::copy_n(imu.gyroscope().begin(), 3, proprio.gyroscope.begin());
    std::copy_n(imu.quaternion().begin(), 4, proprio.quaternion_wxyz.begin());
    const bool advancing_history_tick =
        !history_tick_.has_value() ||
        static_cast<std::int32_t>(state.tick() - *history_tick_) > 0;
    if (state.mode_machine() == true23::kRequiredModeMachine &&
        !core_.stopped() && advancing_history_tick) {
      if (!causal_history_.Push(proprio)) {
        core_.ObserveInternalFailure();
      } else {
        history_tick_ = state.tick();
        proprio_ = proprio;
      }
    }

    std::array<std::uint8_t, 40> remote{};
    std::copy_n(state.wireless_remote().begin(), remote.size(), remote.begin());
    const auto buttons = active::DecodeWirelessOperator(remote);
    const bool arm_edge = buttons.arm_pressed && !arm_pressed_;
    arm_pressed_ = buttons.arm_pressed;
    core_.ObserveOperator(
        {.arm_edge = arm_edge,
         .deadman_held = buttons.deadman_held,
         .stop_requested = buttons.stop_pressed},
        received);
    condition_.notify_all();
  }

  active::GantrySafetyCore& core_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<live::TimedProprioSample> proprio_;
  live::CausalLowStateHistory causal_history_;
  std::optional<std::uint32_t> history_tick_;
  bool arm_pressed_ = false;
};

class ZmqSocket {
 public:
  explicit ZmqSocket(const std::string& endpoint) {
    context_ = zmq_ctx_new();
    if (context_ == nullptr) {
      throw std::runtime_error("cannot create PICO ZMQ context");
    }
    socket_ = zmq_socket(context_, ZMQ_SUB);
    if (socket_ == nullptr) {
      throw std::runtime_error("cannot create PICO ZMQ subscriber");
    }
    constexpr int timeout_ms = 10;
    constexpr int conflate = 1;
    if (zmq_setsockopt(socket_, ZMQ_SUBSCRIBE, "", 0) != 0 ||
        zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &timeout_ms,
                       sizeof(timeout_ms)) != 0 ||
        zmq_setsockopt(socket_, ZMQ_CONFLATE, &conflate,
                       sizeof(conflate)) != 0 ||
        zmq_connect(socket_, endpoint.c_str()) != 0) {
      throw std::runtime_error("cannot configure/connect PICO ZMQ subscriber");
    }
  }
  ~ZmqSocket() {
    if (socket_ != nullptr) {
      zmq_close(socket_);
    }
    if (context_ != nullptr) {
      zmq_ctx_term(context_);
    }
  }

  std::optional<std::string> Receive() {
    std::array<char, 1 << 20> buffer{};
    const auto size = zmq_recv(socket_, buffer.data(), buffer.size(), 0);
    if (size < 0) {
      if (zmq_errno() == EAGAIN) {
        return std::nullopt;
      }
      throw std::runtime_error("PICO ZMQ receive failed");
    }
    if (static_cast<std::size_t>(size) >= buffer.size()) {
      throw std::runtime_error("PICO semantic packet exceeds 1 MiB limit");
    }
    return std::string(buffer.data(), static_cast<std::size_t>(size));
  }

 private:
  void* context_ = nullptr;
  void* socket_ = nullptr;
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

struct LoadedArtifacts {
  fs::path encoder;
  fs::path decoder;
  fs::path metadata;
  fs::path promotion;
  fs::path active_promotion;
  fs::path live_shadow_evidence;
  json candidate_json;
  json promotion_json;
  json active_json;
  std::string encoder_sha;
  std::string decoder_sha;
  std::string metadata_sha;
  std::string promotion_sha;
  std::string active_sha;
  std::string live_shadow_evidence_sha;
  std::string live_shadow_evidence_bytes;
};

fs::path ResolveNewEvidencePath(const fs::path& argument) {
  if (argument.empty() || argument.filename().empty() ||
      argument.filename() == "." || argument.filename() == "..") {
    throw std::runtime_error("execution evidence path is invalid");
  }
  std::error_code error;
  const auto parent = fs::canonical(
      argument.has_parent_path() ? argument.parent_path() : fs::current_path(),
      error);
  if (error || !fs::is_directory(parent, error)) {
    throw std::runtime_error(
        "execution evidence parent is not a resolvable directory");
  }
  const auto path = parent / argument.filename();
  error.clear();
  const auto status = fs::symlink_status(path, error);
  if (!error && status.type() != fs::file_type::not_found) {
    throw std::runtime_error(
        "execution evidence destination already exists");
  }
  if (error && error != std::errc::no_such_file_or_directory) {
    throw std::runtime_error(
        "cannot inspect execution evidence destination");
  }
  return path;
}

class ExecutionEvidenceLog {
 public:
  ExecutionEvidenceLog(
      fs::path path,
      std::string authorization_id,
      json session_start)
      : path_(ResolveNewEvidencePath(path)),
        authorization_id_(std::move(authorization_id)) {
    fd_ = ::open(
        path_.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC | O_NOFOLLOW,
        0600);
    if (fd_ < 0) {
      throw std::runtime_error(
          "cannot exclusively create execution evidence: " +
          std::string(std::strerror(errno)));
    }
    try {
      AppendEvent("session_start", std::move(session_start));
    } catch (...) {
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }

  ExecutionEvidenceLog(const ExecutionEvidenceLog&) = delete;
  ExecutionEvidenceLog& operator=(const ExecutionEvidenceLog&) = delete;

  ~ExecutionEvidenceLog() {
    if (fd_ < 0) {
      return;
    }
    if (!terminal_) {
      try {
        AppendEvent(
            "session_failed",
            {{"passed", false},
             {"reason", "controller_scope_exited_without_terminal_record"}});
      } catch (...) {
      }
    }
    (void)::fsync(fd_);
    (void)::close(fd_);
  }

  [[nodiscard]] const fs::path& path() const { return path_; }

  void AppendEvent(std::string_view event, json payload = json::object()) {
    std::lock_guard lock(mutex_);
    AppendEventLocked(event, std::move(payload));
  }

  void Finalize(std::string_view event, json payload) {
    std::lock_guard lock(mutex_);
    VerifyPathIdentityLocked();
    AppendEventLocked(event, std::move(payload));
    VerifyPathIdentityLocked();
    terminal_ = true;
  }

 private:
  void AppendEventLocked(std::string_view event, json payload) {
    if (!payload.is_object()) {
      throw std::runtime_error("execution evidence payload must be object");
    }
    if (terminal_) {
      throw std::runtime_error("execution evidence is already terminal");
    }
    for (const auto key : {
             "schema_version", "kind", "event", "authorization_id",
             "monotonic_ns"}) {
      if (payload.contains(key)) {
        throw std::runtime_error(
            "execution evidence payload contains reserved field");
      }
    }
    json record = {
        {"schema_version", 1},
        {"kind", "g1_true23_stage1_gantry_execution_evidence"},
        {"event", event},
        {"authorization_id", authorization_id_},
        {"monotonic_ns", NowNs()},
    };
    record.update(payload);
    WriteLocked(record.dump() + "\n");
    if (::fsync(fd_) != 0) {
      throw std::runtime_error("cannot fsync execution evidence");
    }
  }

  void VerifyPathIdentityLocked() const {
    struct stat descriptor_status {};
    struct stat path_status {};
    if (::fstat(fd_, &descriptor_status) != 0 ||
        ::lstat(path_.c_str(), &path_status) != 0 ||
        !S_ISREG(descriptor_status.st_mode) ||
        !S_ISREG(path_status.st_mode) ||
        descriptor_status.st_dev != path_status.st_dev ||
        descriptor_status.st_ino != path_status.st_ino ||
        descriptor_status.st_nlink != 1) {
      throw std::runtime_error(
          "execution evidence path identity changed during session");
    }
  }
  void WriteLocked(std::string_view bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      const auto written = ::write(
          fd_, bytes.data() + offset, bytes.size() - offset);
      if (written < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("cannot append execution evidence");
      }
      if (written == 0) {
        throw std::runtime_error("zero-byte execution evidence write");
      }
      offset += static_cast<std::size_t>(written);
    }
  }

  fs::path path_;
  std::string authorization_id_;
  int fd_ = -1;
  std::mutex mutex_;
  bool terminal_ = false;
};

LoadedArtifacts LoadArtifacts(const Arguments& arguments) {
  LoadedArtifacts result;
  result.encoder = ResolveFile(arguments.encoder, "encoder");
  result.decoder = ResolveFile(arguments.decoder, "decoder");
  result.metadata = ResolveFile(arguments.metadata, "candidate metadata");
  result.promotion = ResolveFile(arguments.promotion, "MuJoCo promotion");
  result.active_promotion =
      ResolveFile(arguments.active_promotion, "active promotion");
  result.live_shadow_evidence = ResolveFile(
      arguments.live_shadow_evidence, "live shadow evidence");
  std::error_code evidence_time_error;
  const auto evidence_write_time = fs::last_write_time(
      result.live_shadow_evidence, evidence_time_error);
  if (evidence_time_error) {
    throw std::runtime_error("cannot inspect live shadow evidence freshness");
  }
  const auto evidence_age_seconds =
      std::chrono::duration<double>(
          fs::file_time_type::clock::now() - evidence_write_time)
          .count();
  if (!std::isfinite(evidence_age_seconds) || evidence_age_seconds < -5.0 ||
      evidence_age_seconds > active::kMaximumShadowEvidenceAgeSeconds) {
    throw std::runtime_error("live shadow evidence is not fresh");
  }
  const std::array<fs::path, 6> paths = {
      result.encoder, result.decoder, result.metadata, result.promotion,
      result.active_promotion, result.live_shadow_evidence};
  for (std::size_t left = 0; left < paths.size(); ++left) {
    for (std::size_t right = left + 1; right < paths.size(); ++right) {
      if (paths[left] == paths[right]) {
        throw std::runtime_error("all artifact paths must resolve distinctly");
      }
    }
  }
  result.encoder_sha = true23::Sha256File(result.encoder.string());
  result.decoder_sha = true23::Sha256File(result.decoder.string());
  result.metadata_sha = true23::Sha256File(result.metadata.string());
  result.promotion_sha = true23::Sha256File(result.promotion.string());
  result.active_sha = true23::Sha256File(result.active_promotion.string());
  result.live_shadow_evidence_sha =
      true23::Sha256File(result.live_shadow_evidence.string());
  result.live_shadow_evidence_bytes = LoadBoundedBytes(
      result.live_shadow_evidence,
      active::kMaximumShadowEvidenceBytes,
      "live shadow evidence");
  if (true23::Sha256Bytes(result.live_shadow_evidence_bytes) !=
      result.live_shadow_evidence_sha) {
    throw std::runtime_error(
        "live shadow evidence byte/file hash changed during load");
  }
  result.candidate_json = LoadJson(result.metadata, "candidate metadata");
  result.promotion_json = LoadJson(result.promotion, "MuJoCo promotion");
  result.active_json = LoadJson(result.active_promotion, "active promotion");
  return result;
}

void VerifyFilesUnchanged(const LoadedArtifacts& files) {
  if (true23::Sha256File(files.encoder.string()) != files.encoder_sha ||
      true23::Sha256File(files.decoder.string()) != files.decoder_sha ||
      true23::Sha256File(files.metadata.string()) != files.metadata_sha ||
      true23::Sha256File(files.promotion.string()) != files.promotion_sha ||
      true23::Sha256File(files.active_promotion.string()) != files.active_sha ||
      true23::Sha256File(files.live_shadow_evidence.string()) !=
          files.live_shadow_evidence_sha) {
    throw std::runtime_error("artifact changed during validation/load");
  }
}

void RequireExactKeys(
    const json& value,
    std::initializer_list<std::string_view> keys,
    std::string_view context) {
  if (!value.is_object() || value.size() != keys.size()) {
    throw std::runtime_error(
        std::string(context) + " field set is not exact");
  }
  for (const auto key : keys) {
    if (!value.contains(std::string(key))) {
      throw std::runtime_error(
          std::string(context) + " missing field: " + std::string(key));
    }
  }
}

std::string RequireSha256(const json& value, std::string_view context) {
  if (!value.is_string()) {
    throw std::runtime_error(std::string(context) + " must be SHA-256");
  }
  const auto result = value.get<std::string>();
  if (!active::IsLowerSha256(result)) {
    throw std::runtime_error(
        std::string(context) + " must be lowercase SHA-256");
  }
  return result;
}

void ValidateCausalDiagnosticPair(
    const LoadedArtifacts& files,
    const OnnxModel& encoder,
    const OnnxModel& decoder) {
  const auto& root = files.candidate_json;
  RequireExactKeys(
      root,
      {"schema_version", "kind", "diagnostic_only", "deployment_ready",
       "promotion_eligible", "active_motor_control_authorized",
       "checkpoint_role", "allowed_uses", "forbidden_uses",
       "no_robot_or_network_commands_performed", "source", "contract",
       "artifacts", "hashes", "validation", "metadata_payload_sha256"},
      "causal diagnostic bundle");
  if (root.at("schema_version") != live::kSafeTargetDiagnosticSchemaVersion ||
      root.at("kind") != "g1_true23_mjlab_diagnostic_onnx_pair" ||
      root.at("diagnostic_only") != true ||
      root.at("deployment_ready") != false ||
      root.at("promotion_eligible") != false ||
      root.at("active_motor_control_authorized") != false ||
      root.at("checkpoint_role") != "training_resume_only" ||
      root.at("allowed_uses") != json::array(
          {"mujoco_sim2sim_diagnostic", "pico_shadow_diagnostic"}) ||
      root.at("forbidden_uses") != json::array(
          {"active_motor_control", "deployment", "promotion"}) ||
      root.at("no_robot_or_network_commands_performed") != true) {
    throw std::runtime_error("causal diagnostic safety contract mismatch");
  }

  const auto& source = root.at("source");
  RequireExactKeys(
      source,
      {"checkpoint_filename", "checkpoint_update_count", "reference_profile",
       "simulation_candidate_review_allowed"},
      "causal diagnostic source");
  if (!source.at("checkpoint_filename").is_string() ||
      source.at("checkpoint_filename").get<std::string>().empty() ||
      !source.at("checkpoint_update_count").is_number_integer() ||
      source.at("checkpoint_update_count").get<int>() <= 0 ||
      source.at("reference_profile") != live::kCausalReferenceProfile ||
      source.at("simulation_candidate_review_allowed") != true) {
    throw std::runtime_error("causal diagnostic source mismatch");
  }

  const auto& contract = root.at("contract");
  RequireExactKeys(
      contract,
      {"robot_model", "required_mode_machine", "native_action_dof",
       "history_length", "observation_layout", "onnx_opset", "encoder",
       "decoder", "decoder_output_semantics",
       "external_safe_target_transform_allowed",
       "previous_action_semantics", "safe_target_transform"},
      "causal diagnostic contract");
  if (contract.at("robot_model") != true23::kRobotModel ||
      contract.at("required_mode_machine") != true23::kRequiredModeMachine ||
      contract.at("native_action_dof") != true23::kDecoderOutputDim ||
      contract.at("history_length") != true23::kHistoryLength ||
      contract.at("observation_layout") !=
          "canonical_il29_fixed_slots_v1" ||
      contract.at("onnx_opset") != true23::kOnnxOpset) {
    throw std::runtime_error("causal diagnostic embodiment mismatch");
  }
  std::vector<std::string> signature_errors;
  true23::ValidatePairModelSignatures(
      encoder.signature(), decoder.signature(), signature_errors);
  if (!signature_errors.empty()) {
    throw std::runtime_error(signature_errors.front());
  }

  const auto& artifacts = root.at("artifacts");
  RequireExactKeys(
      artifacts,
      {"encoder_onnx_filename", "decoder_onnx_filename", "metadata_filename"},
      "causal diagnostic artifacts");
  if (artifacts.at("encoder_onnx_filename") !=
          files.encoder.filename().string() ||
      artifacts.at("decoder_onnx_filename") !=
          files.decoder.filename().string() ||
      artifacts.at("metadata_filename") !=
          files.metadata.filename().string()) {
    throw std::runtime_error("causal diagnostic filename binding mismatch");
  }

  const auto& hashes = root.at("hashes");
  RequireExactKeys(
      hashes,
      {"checkpoint_sha256", "lineage_sha256", "policy_state_sha256",
       "encoder_state_sha256", "decoder_state_sha256",
       "encoder_onnx_sha256", "decoder_onnx_sha256",
       "encoder_embedded_metadata_sha256",
       "decoder_embedded_metadata_sha256", "safe_target_transform_sha256"},
      "causal diagnostic hashes");
  for (const auto& [key, value] : hashes.items()) {
    (void)RequireSha256(value, key);
  }
  if (hashes.at("encoder_onnx_sha256") != files.encoder_sha ||
      hashes.at("decoder_onnx_sha256") != files.decoder_sha ||
      true23::Sha256CanonicalJson(encoder.embedded()) !=
          hashes.at("encoder_embedded_metadata_sha256") ||
      true23::Sha256CanonicalJson(decoder.embedded()) !=
          hashes.at("decoder_embedded_metadata_sha256")) {
    throw std::runtime_error("causal diagnostic ONNX/hash binding mismatch");
  }
  if (encoder.embedded().at("schema_version") !=
          live::kSafeTargetDiagnosticSchemaVersion ||
      decoder.embedded().at("schema_version") !=
          live::kSafeTargetDiagnosticSchemaVersion) {
    throw std::runtime_error("causal diagnostic embedded schema mismatch");
  }
  live::ValidateAppliedSafeDecoderContract(
      contract, hashes, encoder.embedded(), decoder.embedded());

  const auto expected_payload = RequireSha256(
      root.at("metadata_payload_sha256"), "metadata_payload_sha256");
  auto unhashed = root;
  unhashed.erase("metadata_payload_sha256");
  if (true23::Sha256CanonicalJson(unhashed) != expected_payload) {
    throw std::runtime_error("causal diagnostic payload hash mismatch");
  }
}

void ValidateCausalMujocoPromotion(
    const LoadedArtifacts& files) {
  const auto& root = files.promotion_json;
  RequireExactKeys(
      root,
      {"schema_version", "kind", "robot_model", "promotion_stage",
       "deployment_bytes_authorized", "active_motor_control_authorized",
       "gantry_or_rated_support_required", "free_standing_authorized",
       "required_mode_machine", "decoder_output_semantics",
       "previous_action_semantics", "external_safe_target_transform_allowed",
       "safe_target_transform_sha256", "source_artifact",
       "full_campaign_evidence", "promotion_payload_sha256"},
      "causal MuJoCo promotion");
  if (root.at("schema_version") != 2 ||
      root.at("kind") != "g1_true23_causal_mujoco_promotion" ||
      root.at("robot_model") != true23::kRobotModel ||
      root.at("promotion_stage") != "causal_mujoco_full" ||
      root.at("deployment_bytes_authorized") != true ||
      root.at("active_motor_control_authorized") != false ||
      root.at("gantry_or_rated_support_required") != true ||
      root.at("free_standing_authorized") != false ||
      root.at("required_mode_machine") != true23::kRequiredModeMachine ||
      root.at("decoder_output_semantics") !=
          live::kAppliedSafeNativeActionSemantics ||
      root.at("previous_action_semantics") !=
          live::kAppliedSafeNativeActionSemantics ||
      root.at("external_safe_target_transform_allowed") != false ||
      root.at("safe_target_transform_sha256") !=
          live::kSafeTargetTransformSha256) {
    throw std::runtime_error("causal MuJoCo promotion contract mismatch");
  }

  const auto& candidate = files.candidate_json;
  const auto& source = root.at("source_artifact");
  RequireExactKeys(
      source,
      {"checkpoint_filename", "checkpoint_sha256", "checkpoint_update_count",
       "lineage_sha256", "policy_state_sha256", "reference_profile",
       "causal_reference_contract_sha256", "encoder_onnx_filename",
       "encoder_onnx_sha256", "decoder_onnx_filename",
       "decoder_onnx_sha256", "metadata_filename", "metadata_sha256",
       "metadata_payload_sha256"},
      "causal promotion source artifact");
  for (const auto key : {
           "checkpoint_sha256", "lineage_sha256", "policy_state_sha256",
           "causal_reference_contract_sha256", "encoder_onnx_sha256",
           "decoder_onnx_sha256", "metadata_sha256",
           "metadata_payload_sha256"}) {
    (void)RequireSha256(source.at(key), key);
  }
  if (source.at("checkpoint_filename") !=
          candidate.at("source").at("checkpoint_filename") ||
      source.at("checkpoint_update_count") !=
          candidate.at("source").at("checkpoint_update_count") ||
      source.at("checkpoint_sha256") !=
          candidate.at("hashes").at("checkpoint_sha256") ||
      source.at("lineage_sha256") !=
          candidate.at("hashes").at("lineage_sha256") ||
      source.at("policy_state_sha256") !=
          candidate.at("hashes").at("policy_state_sha256") ||
      source.at("reference_profile") != live::kCausalReferenceProfile ||
      source.at("causal_reference_contract_sha256") !=
          live::kCausalReferenceContractSha256 ||
      source.at("encoder_onnx_filename") != files.encoder.filename().string() ||
      source.at("decoder_onnx_filename") != files.decoder.filename().string() ||
      source.at("metadata_filename") != files.metadata.filename().string() ||
      source.at("encoder_onnx_sha256") != files.encoder_sha ||
      source.at("decoder_onnx_sha256") != files.decoder_sha ||
      source.at("metadata_sha256") != files.metadata_sha ||
      source.at("metadata_payload_sha256") !=
          candidate.at("metadata_payload_sha256")) {
    throw std::runtime_error("causal promotion source binding mismatch");
  }

  const auto& campaign = root.at("full_campaign_evidence");
  RequireExactKeys(
      campaign,
      {"aggregate_report_filename", "aggregate_report_sha256",
       "aggregate_report_payload_sha256", "campaign_layout",
       "shard_manifest_sha256", "trace_manifest_sha256", "shards",
       "deterministic_seeds", "scenarios", "run_count",
       "episodes_per_scenario", "total_episodes", "total_records",
       "all_strict_gates_pass", "summary_metrics", "provenance"},
      "causal full campaign evidence");
  for (const auto key : {
           "aggregate_report_sha256", "aggregate_report_payload_sha256",
           "shard_manifest_sha256", "trace_manifest_sha256"}) {
    (void)RequireSha256(campaign.at(key), key);
  }
  if (campaign.at("campaign_layout") != "monolithic" ||
      campaign.at("shards") != json::array() ||
      campaign.at("shard_manifest_sha256") !=
          true23::Sha256CanonicalJson(json::array()) ||
      campaign.at("deterministic_seeds") != json::array({1729, 2718, 3141}) ||
      campaign.at("scenarios") != json::array(
          {"nominal", "push_50", "push_100", "domain_push_100"}) ||
      campaign.at("run_count") != 12 ||
      campaign.at("episodes_per_scenario") != 66 ||
      campaign.at("total_episodes") != 264 ||
      campaign.at("total_records") != 66000 ||
      campaign.at("all_strict_gates_pass") != true) {
    throw std::runtime_error("causal full campaign fixed contract mismatch");
  }

  const auto& provenance = campaign.at("provenance");
  RequireExactKeys(
      provenance,
      {"base_sim2sim_config_relpath", "base_sim2sim_config_sha256",
       "mjcf_relpath", "mjcf_sha256", "domain_source_relpath",
       "domain_source_sha256", "mjlab_env_relpath", "mjlab_env_sha256",
       "executed_producer_archive_manifest_relpath",
       "executed_producer_archive_manifest_sha256",
       "executed_runtime_filename", "executed_runtime_sha256",
       "executed_runner_filename", "executed_runner_sha256"},
      "causal campaign provenance");
  for (const auto key : {
           "base_sim2sim_config_sha256", "mjcf_sha256",
           "domain_source_sha256", "mjlab_env_sha256",
           "executed_producer_archive_manifest_sha256",
           "executed_runtime_sha256", "executed_runner_sha256"}) {
    (void)RequireSha256(provenance.at(key), key);
  }
  for (const auto key : {
           "base_sim2sim_config_relpath", "mjcf_relpath",
           "domain_source_relpath", "mjlab_env_relpath",
           "executed_producer_archive_manifest_relpath"}) {
    if (!provenance.at(key).is_string()) {
      throw std::runtime_error("causal provenance path must be a string");
    }
    const fs::path relative(provenance.at(key).get<std::string>());
    if (relative.empty() || relative.is_absolute() ||
        relative.lexically_normal() != relative) {
      throw std::runtime_error("causal provenance path is not canonical relative");
    }
    for (const auto& component : relative) {
      if (component == "..") {
        throw std::runtime_error("causal provenance path escapes root");
      }
    }
  }
  for (const auto key : {
           "executed_runtime_filename", "executed_runner_filename"}) {
    if (!provenance.at(key).is_string()) {
      throw std::runtime_error("causal provenance filename must be a string");
    }
    const auto filename = provenance.at(key).get<std::string>();
    if (filename.empty() || fs::path(filename).filename() != filename) {
      throw std::runtime_error("causal provenance filename is not a basename");
    }
  }

  const auto& metrics = campaign.at("summary_metrics");
  RequireExactKeys(
      metrics,
      {"run_count", "episode_count", "record_count", "termination_count",
       "fall_count", "policy_output_nonfinite_count",
       "joint_limit_violation_count", "joint_velocity_bound_violation_count",
       "effort_bound_violation_count", "minimum_recovery_fraction",
       "maximum_recovery_time_s", "max_action_saturation_fraction",
       "max_joint_velocity_ratio", "max_effort_ratio",
       "max_abs_native_action_raw", "minimum_base_height_m",
       "maximum_tilt_rad"},
      "causal campaign summary metrics");
  if (metrics.at("run_count") != 12 || metrics.at("episode_count") != 264 ||
      metrics.at("record_count") != 66000 ||
      metrics.at("termination_count") != 0 || metrics.at("fall_count") != 0 ||
      metrics.at("policy_output_nonfinite_count") != 0 ||
      metrics.at("joint_limit_violation_count") != 0 ||
      metrics.at("joint_velocity_bound_violation_count") != 0 ||
      metrics.at("effort_bound_violation_count") != 0 ||
      metrics.at("minimum_recovery_fraction") != 1.0 ||
      metrics.at("maximum_recovery_time_s").get<double>() > 2.0 ||
      metrics.at("max_action_saturation_fraction") != 0.0 ||
      metrics.at("max_joint_velocity_ratio").get<double>() > 1.0 ||
      metrics.at("max_effort_ratio").get<double>() > 1.0) {
    throw std::runtime_error("causal campaign strict metrics failed");
  }
  for (const auto key : {
           "maximum_recovery_time_s", "max_joint_velocity_ratio",
           "max_effort_ratio", "max_abs_native_action_raw",
           "minimum_base_height_m", "maximum_tilt_rad"}) {
    if (!metrics.at(key).is_number() ||
        !std::isfinite(metrics.at(key).get<double>())) {
      throw std::runtime_error("causal campaign metric is non-finite");
    }
  }

  const auto expected_payload = RequireSha256(
      root.at("promotion_payload_sha256"), "promotion_payload_sha256");
  auto unhashed = root;
  unhashed.erase("promotion_payload_sha256");
  if (true23::Sha256CanonicalJson(unhashed) != expected_payload) {
    throw std::runtime_error("causal promotion payload hash mismatch");
  }
}

int Run(const Arguments& arguments) {
  auto files = LoadArtifacts(arguments);
  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "g1_true23_active_gantry");
  OnnxModel encoder(environment, files.encoder);
  OnnxModel decoder(environment, files.decoder);

  ValidateCausalDiagnosticPair(files, encoder, decoder);
  ValidateCausalMujocoPromotion(files);
  const auto shadow_summary = active::ValidateLiveShadowEvidenceJsonl(
      files.live_shadow_evidence_bytes,
      {
          .encoder_sha256 = files.encoder_sha,
          .decoder_sha256 = files.decoder_sha,
          .metadata_sha256 = files.metadata_sha,
          .promotion_sha256 = files.promotion_sha,
          .network = arguments.network,
          .pico_endpoint = arguments.pico_endpoint,
      });
  const bool causal_reference_artifact = true;
  auto artifact = active::ParseActivePromotion(
      files.active_json, true, files.promotion_sha,
      files.encoder_sha, files.decoder_sha, files.metadata_sha,
      files.live_shadow_evidence_sha);
  if (artifact.authorization_id != arguments.authorization_id) {
    throw std::runtime_error(
        "operator authorization-id does not match active promotion");
  }
  const auto& candidate_hashes = files.candidate_json.at("hashes");
  const auto& promotion_source =
      files.promotion_json.at("source_artifact");
  const auto& campaign =
      files.promotion_json.at("full_campaign_evidence");
  if (artifact.checkpoint_sha256 !=
          candidate_hashes.at("checkpoint_sha256") ||
      artifact.lineage_sha256 != candidate_hashes.at("lineage_sha256") ||
      artifact.policy_state_sha256 !=
          candidate_hashes.at("policy_state_sha256") ||
      artifact.safe_target_transform_sha256 !=
          candidate_hashes.at("safe_target_transform_sha256") ||
      artifact.checkpoint_sha256 !=
          promotion_source.at("checkpoint_sha256") ||
      artifact.lineage_sha256 != promotion_source.at("lineage_sha256") ||
      artifact.policy_state_sha256 !=
          promotion_source.at("policy_state_sha256") ||
      artifact.full_campaign_aggregate_sha256 !=
          campaign.at("aggregate_report_sha256") ||
      artifact.full_campaign_shard_manifest_sha256 !=
          campaign.at("shard_manifest_sha256")) {
    throw std::runtime_error(
        "active promotion material/campaign binding mismatch");
  }
  const auto active_errors = active::ValidateActiveArtifactBinding(artifact);
  if (!active_errors.empty()) {
    throw std::runtime_error("active promotion rejected: " + active_errors.front());
  }

  const std::array<float, true23::kEncoderInputDim> zero_encoder{};
  const auto token = encoder.Run<true23::kEncoderInputDim,
                                 true23::kEncoderOutputDim>(zero_encoder);
  std::array<float, true23::kDecoderInputDim> zero_decoder{};
  std::copy(token.begin(), token.end(), zero_decoder.begin());
  (void)decoder.Run<true23::kDecoderInputDim,
                    true23::kDecoderOutputDim>(zero_decoder);
  VerifyFilesUnchanged(files);
  if (arguments.validate_only) {
    std::cout
        << "[PASS] exact True23 causal gantry promotion and "
        << shadow_summary.action_frames
        << "-frame promoted shadow PASS validated; no DDS, motion-mode "
           "client, or LowCmd publisher was created.\n";
    return 0;
  }

  std::error_code executable_error;
  const auto executable_path = fs::canonical("/proc/self/exe", executable_error);
  if (executable_error || !fs::is_regular_file(executable_path)) {
    throw std::runtime_error("cannot resolve running controller binary");
  }
  const auto executable_sha = true23::Sha256File(executable_path.string());
  ExecutionEvidenceLog execution_evidence(
      arguments.execution_evidence,
      arguments.authorization_id,
      {
          {"controller_binary_path", executable_path.string()},
          {"controller_binary_sha256", executable_sha},
          {"encoder_sha256", files.encoder_sha},
          {"decoder_sha256", files.decoder_sha},
          {"metadata_sha256", files.metadata_sha},
          {"promotion_sha256", files.promotion_sha},
          {"active_promotion_sha256", files.active_sha},
          {"live_shadow_evidence_sha256", files.live_shadow_evidence_sha},
          {"live_shadow_action_frames", shadow_summary.action_frames},
          {"network", arguments.network},
          {"pico_endpoint", arguments.pico_endpoint},
          {"post_arm_duration_seconds",
           arguments.post_arm_duration_seconds},
          {"minimum_policy_command_frames",
           active::kMinimumPromotedShadowActionFrames},
          {"mutation_gate_open", false},
          {"motion_mode_released", false},
          {"lowcmd_publisher_created", false},
      });
  execution_evidence.AppendEvent(
      "artifact_gate_passed",
      {{"onnx_dry_run_passed", true},
       {"artifact_bytes_reverified", true},
       {"active_promotion_authorized", true},
       {"live_shadow_evidence_validated", true}});

  active::GantrySafetyCore core(std::move(artifact));
  if (core.stopped()) {
    throw std::runtime_error("active artifact safety core refused authorization");
  }

  // Mutation boundary: DDS starts subscriber-only. Publisher and motion-mode
  // client do not exist until five exact advancing samples have passed.
  unitree::robot::ChannelFactory::Instance()->Init(0, arguments.network);
  StateMonitor monitor(core);
  monitor.Start();
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(kStateGateTimeoutSeconds);
  if (!monitor.WaitForMutationGate(deadline)) {
    throw std::runtime_error(
        "five advancing CRC-valid mode_machine==4 states were not obtained");
  }
  execution_evidence.AppendEvent(
      "mutation_gate_open",
      {{"stable_advancing_lowstate_samples", 5},
       {"required_mode_machine", true23::kRequiredModeMachine},
       {"crc_valid", true}});
  if (g_stop_requested != 0) {
    throw std::runtime_error(
        "signal stopped controller before motion-mode release");
  }
  ReleaseMotionModeAfterGate();
  execution_evidence.AppendEvent(
      "motion_mode_released",
      {{"post_release_mode_name_empty", true}});
  if (g_stop_requested != 0) {
    throw std::runtime_error(
        "signal stopped controller after motion-mode release and before publisher");
  }
  monitor.WithCore([](active::GantrySafetyCore& value) {
    value.CheckWatchdogs(NowNs());
  });
  if (monitor.fault() != active::Fault::None) {
    throw std::runtime_error("state fault occurred before publisher creation");
  }

  auto publisher =
      std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(
          std::string(kLowCmdTopic));
  publisher->InitChannel();
  execution_evidence.AppendEvent(
      "lowcmd_publisher_created",
      {{"topic", std::string(kLowCmdTopic)},
       {"writes_before_event", 0}});
  if (g_stop_requested != 0) {
    throw std::runtime_error(
        "signal stopped controller before command-writer startup");
  }
  std::atomic<bool> stop_threads{false};
  std::atomic<bool> first_policy_ready_for_arm{false};
  std::atomic<std::int64_t> first_policy_write_ns{0};
  std::uint64_t policy_command_frames = 0;
  int damping_frames_after_stop = 0;
  double maximum_target_delta_from_state_rad = 0.0;
  double maximum_target_slew_rad = 0.0;
  double maximum_abs_predicted_effort_nm = 0.0;
  double maximum_abs_feedforward_tau_nm = 0.0;
  std::array<double, 23> previous_active_targets{};
  bool have_previous_active_targets = false;
  std::string inference_error;
  std::string writer_error;
  bool publisher_write_failed = false;

  std::jthread inference_thread([&](std::stop_token stop_token) {
    try {
      ZmqSocket socket(arguments.pico_endpoint);
      live::ProprioHistory history;
      active::RealProprioWarmupGate real_history_gate;
      std::array<float, 23> previous_action{};
      std::optional<std::uint64_t> last_frame;
      std::optional<std::int64_t> last_source_monotonic_ns;
      while (!stop_token.stop_requested() && !stop_threads.load()) {
        const auto message = socket.Receive();
        if (!message.has_value()) {
          continue;
        }
        std::optional<live::PicoEncoderTerms> released_terms;
        std::optional<live::CausalPicoReferenceTerms> causal_reference;
        std::uint64_t frame_index = 0;
        std::int64_t source_monotonic_ns = 0;
        if (causal_reference_artifact) {
          causal_reference = ParseCausalPicoReferenceTerms(*message);
          frame_index = causal_reference->control_source_frame_index;
          source_monotonic_ns = causal_reference->control_monotonic_ns;
        } else {
          released_terms = ParsePicoTerms(*message);
          frame_index = released_terms->source_frame_index;
          source_monotonic_ns = released_terms->source_monotonic_ns;
        }
        const auto now = NowNs();
        const auto age = now - source_monotonic_ns;
        if (age < -active::kFutureClockToleranceNs ||
            age > active::kPolicyFreshnessNs) {
          monitor.WithCore([](active::GantrySafetyCore& value) {
            value.ObservePicoTermsFailure();
          });
          break;
        }
        if (last_frame.has_value() && frame_index != *last_frame + 1U) {
          monitor.WithCore([](active::GantrySafetyCore& value) {
            value.ObservePicoFrameRegression();
          });
          break;
        }
        if (last_source_monotonic_ns.has_value() &&
            source_monotonic_ns !=
                *last_source_monotonic_ns + active::kShadowControlPeriodNs) {
          monitor.WithCore([](active::GantrySafetyCore& value) {
            value.ObservePicoTermsFailure();
          });
          break;
        }
        last_frame = frame_index;
        last_source_monotonic_ns = source_monotonic_ns;

        live::TimedProprioSample proprio;
        std::array<float, true23::kEncoderInputDim> encoder_input{};
        if (causal_reference_artifact) {
          const auto attempt = monitor.TryJoinCausalReference(
              *causal_reference);
          if (attempt.status == CausalJoinStatus::AwaitingCoverage) {
            continue;
          }
          if (attempt.status != CausalJoinStatus::Ready ||
              !attempt.joined.has_value()) {
            monitor.WithCore([](active::GantrySafetyCore& value) {
              value.ObservePicoTermsFailure();
            });
            break;
          }
          proprio = attempt.joined->control_proprio_q10;
          encoder_input = live::BuildCausalEncoderInput(
              attempt.joined->encoder_terms);
        } else {
          const auto latest = monitor.LatestProprio();
          if (!latest.has_value() ||
              now - latest->received_monotonic_ns >
                  active::kStateFreshnessNs) {
            continue;
          }
          proprio = *latest;
          encoder_input = live::BuildEncoderInput(*released_terms);
        }
        live::ProprioSource source{
            .hardware_q = proprio.hardware_q,
            .hardware_dq = proprio.hardware_dq,
            .imu_gyroscope = proprio.gyroscope,
            .imu_quaternion_wxyz = proprio.quaternion_wxyz,
            .previous_action_native = previous_action,
        };
        history.Push(live::BuildProprioFrame(source));
        if (!real_history_gate.Observe(frame_index)) {
          if (real_history_gate.rejected()) {
            monitor.WithCore([](active::GantrySafetyCore& value) {
              value.ObservePicoFrameRegression();
            });
            break;
          }
          continue;
        }
        if (!history.ready()) {
          monitor.WithCore([](active::GantrySafetyCore& value) {
            value.ObservePicoTermsFailure();
          });
          break;
        }
        const auto live_token =
            encoder.Run<true23::kEncoderInputDim,
                        true23::kEncoderOutputDim>(encoder_input);
        const auto decoder_input =
            live::BuildDecoderInput(live_token, history.Flatten());
        const auto action =
            decoder.Run<true23::kDecoderInputDim,
                        true23::kDecoderOutputDim>(decoder_input);
        bool policy_ready = false;
        monitor.WithCore([&](active::GantrySafetyCore& value) {
          value.SubmitPolicy(
              {.native_action = action, .produced_monotonic_ns = now}, now);
          policy_ready = value.policy_ready_for_arm(now);
        });
        if (policy_ready &&
            !first_policy_ready_for_arm.load(std::memory_order_relaxed)) {
          execution_evidence.AppendEvent(
              "first_policy_ready_for_arm",
              {{"real_history_frames", true23::kHistoryLength},
               {"policy_freshness_limit_ns", active::kPolicyFreshnessNs}});
          first_policy_ready_for_arm.store(true, std::memory_order_release);
        }
        previous_action = action;
      }
    } catch (const std::exception& error) {
      std::cerr << "PICO/inference fault: " << error.what() << '\n';
      inference_error = error.what();
      monitor.WithCore([](active::GantrySafetyCore& value) {
        value.ObservePicoTermsFailure();
      });
    }
  });

  std::jthread writer_thread([&](std::stop_token stop_token) {
    const auto enter_fail_safe = [&](std::string failure) {
      writer_error = std::move(failure);
      try {
        monitor.WithCore([](active::GantrySafetyCore& value) {
          value.ObserveInternalFailure();
        });
      } catch (...) {
        writer_error += "; failed to latch internal safety fault";
      }
      auto damping_deadline_ns = NowNs();
      constexpr int kMaximumDampingWriteAttempts =
          kFaultDampingCycles + 25;
      int successful_damping_writes = 0;
      for (int attempt = 0;
           attempt < kMaximumDampingWriteAttempts &&
           successful_damping_writes < kFaultDampingCycles;
           ++attempt) {
        std::this_thread::sleep_until(
            std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(damping_deadline_ns)));
        try {
          monitor.WithCore([&](active::GantrySafetyCore& value) {
            publisher->Write(ToLowCmd(value.BuildDampingCommand()));
          });
          ++successful_damping_writes;
          damping_frames_after_stop = successful_damping_writes;
        } catch (const std::exception& damping_error) {
          publisher_write_failed = true;
          writer_error += "; fail-safe damping write failed: ";
          writer_error += damping_error.what();
          // Continue bounded retries: a transient failed write does not
          // prove later damping writes will fail.
        } catch (...) {
          publisher_write_failed = true;
          writer_error += "; fail-safe damping write failed: unknown exception";
        }
        damping_deadline_ns =
            active::NextNoCatchUpWriterDeadlineNs(NowNs());
      }
      stop_threads.store(true);
    };
    try {
      auto next_write_ns = NowNs();
      int damping_cycles = 0;
      while (!stop_token.stop_requested() && !stop_threads.load()) {
        std::this_thread::sleep_until(
            std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(next_write_ns)));
        active::MotorCommand command;
        active::Fault fault = active::Fault::None;
        std::optional<active::StateSample> command_state;
        bool policy_command = false;
        monitor.WithCore([&](active::GantrySafetyCore& value) {
          if (g_stop_requested != 0) {
            value.Stop();
          }
          const bool armed_before_build = value.armed();
          command = value.BuildCommand(NowNs());
          fault = value.fault();
          command_state = value.latest_state();
          policy_command =
              armed_before_build && value.armed() &&
              fault == active::Fault::None;
          try {
            publisher->Write(ToLowCmd(command));
          } catch (...) {
            publisher_write_failed = true;
            throw;
          }
        });
        const auto completed_write_ns = NowNs();
        next_write_ns =
            active::NextNoCatchUpWriterDeadlineNs(completed_write_ns);
        if (policy_command) {
          if (!command_state.has_value()) {
            throw std::runtime_error(
                "policy command lacks bound LowState sample");
          }
          for (std::size_t compact = 0; compact < 23; ++compact) {
            const auto slot = static_cast<std::size_t>(
                true23::kHardwareJointIds[compact]);
            const auto target = command[slot].q;
            maximum_target_delta_from_state_rad = std::max(
                maximum_target_delta_from_state_rad,
                std::abs(target - command_state->q[slot]));
            const auto previous_target = have_previous_active_targets
                                             ? previous_active_targets[compact]
                                             : command_state->q[slot];
            maximum_target_slew_rad = std::max(
                maximum_target_slew_rad,
                std::abs(target - previous_target));
            const auto predicted_effort =
                command[slot].kp * (target - command_state->q[slot]) -
                command[slot].kd * command_state->dq[slot] +
                command[slot].tau;
            maximum_abs_predicted_effort_nm = std::max(
                maximum_abs_predicted_effort_nm,
                std::abs(predicted_effort));
            maximum_abs_feedforward_tau_nm = std::max(
                maximum_abs_feedforward_tau_nm,
                std::abs(command[slot].tau));
            previous_active_targets[compact] = target;
          }
          have_previous_active_targets = true;
          ++policy_command_frames;
          if (policy_command_frames == 1) {
            first_policy_write_ns.store(completed_write_ns);
            execution_evidence.AppendEvent(
                "first_armed_policy_command_written",
                {{"policy_command_frame", 1},
                 {"native_action_dof", 23},
                 {"feedforward_tau_zero", true}});
          }
        }
        if (fault != active::Fault::None &&
            ++damping_cycles > 0) {
          damping_frames_after_stop = damping_cycles;
          if (damping_cycles >= kFaultDampingCycles) {
            stop_threads.store(true);
            break;
          }
        }
      }
    } catch (const std::exception& error) {
      enter_fail_safe(error.what());
    } catch (...) {
      enter_fail_safe("unknown command-writer exception");
    }
  });

  std::cout
      << "[WAIT] True23 publisher is damping-only; waiting for first fresh "
         "10-frame causal policy. Do not press A yet.\n";
  while (!first_policy_ready_for_arm.load(std::memory_order_acquire) &&
         !stop_threads.load() &&
         g_stop_requested == 0 && monitor.fault() == active::Fault::None) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (first_policy_ready_for_arm.load(std::memory_order_acquire) &&
      !stop_threads.load() &&
      g_stop_requested == 0 && monitor.fault() == active::Fault::None) {
    std::cout
        << "[READY] Fresh policy ready. Gantry secure; release A if already "
           "held, hold L2, then press A once. B/R2 or L2 release stops.\n";
  }
  std::string stop_reason;
  std::int64_t stop_requested_ns = 0;
  while (!stop_threads.load()) {
    const auto now_ns = NowNs();
    if (g_stop_requested != 0 && stop_reason.empty()) {
      stop_reason = "process_signal";
      stop_requested_ns = now_ns;
      monitor.WithCore([](active::GantrySafetyCore& value) { value.Stop(); });
    }
    const auto first_write = first_policy_write_ns.load();
    if (first_write > 0 && stop_reason.empty() &&
        now_ns - first_write >=
            static_cast<std::int64_t>(arguments.post_arm_duration_seconds) *
                1'000'000'000LL) {
      stop_reason = "reviewed_post_arm_duration_complete";
      stop_requested_ns = now_ns;
      monitor.WithCore([](active::GantrySafetyCore& value) { value.Stop(); });
    }
    const auto observed_fault = monitor.fault();
    if (observed_fault != active::Fault::None && stop_reason.empty()) {
      stop_reason = observed_fault == active::Fault::OperatorStop
                        ? "wireless_operator_stop"
                        : std::string(active::FaultName(observed_fault));
      stop_requested_ns = now_ns;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  inference_thread.request_stop();
  writer_thread.request_stop();
  if (inference_thread.joinable()) {
    inference_thread.join();
  }
  if (writer_thread.joinable()) {
    writer_thread.join();
  }
  const auto fault = monitor.fault();
  if (stop_reason.empty()) {
    stop_reason = fault == active::Fault::None
                      ? "writer_stopped_without_latched_fault"
                      : std::string(active::FaultName(fault));
  }
  const bool armed_transition_observed = first_policy_write_ns.load() > 0;
  const bool enough_policy_commands =
      policy_command_frames >= static_cast<std::uint64_t>(
                                   active::kMinimumPromotedShadowActionFrames);
  const bool full_damping_stop =
      damping_frames_after_stop >= kFaultDampingCycles;
  const bool safe_terminal_fault = fault == active::Fault::OperatorStop;
  const auto first_write_ns = first_policy_write_ns.load();
  const auto post_arm_elapsed_ns =
      first_write_ns > 0 && stop_requested_ns >= first_write_ns
          ? stop_requested_ns - first_write_ns
          : 0;
  const auto required_post_arm_duration_ns =
      static_cast<std::int64_t>(arguments.post_arm_duration_seconds) *
      1'000'000'000LL;
  const bool reviewed_duration_completed =
      stop_reason == "reviewed_post_arm_duration_complete" &&
      post_arm_elapsed_ns >= required_post_arm_duration_ns;
  const bool passed =
      armed_transition_observed && enough_policy_commands &&
      full_damping_stop && safe_terminal_fault && reviewed_duration_completed &&
      inference_error.empty() && writer_error.empty();
  json terminal = {
      {"passed", passed},
      {"armed_transition_observed", armed_transition_observed},
      {"policy_command_frames", policy_command_frames},
      {"minimum_policy_command_frames",
       active::kMinimumPromotedShadowActionFrames},
      {"damping_frames_after_stop", damping_frames_after_stop},
      {"required_damping_frames_after_stop", kFaultDampingCycles},
      {"maximum_target_delta_from_state_rad",
       maximum_target_delta_from_state_rad},
      {"maximum_target_slew_rad", maximum_target_slew_rad},
      {"maximum_abs_predicted_effort_nm",
       maximum_abs_predicted_effort_nm},
      {"maximum_abs_feedforward_tau_nm",
       maximum_abs_feedforward_tau_nm},
      {"final_fault", std::string(active::FaultName(fault))},
      {"stop_reason", stop_reason},
      {"post_arm_elapsed_ns", post_arm_elapsed_ns},
      {"required_post_arm_duration_ns", required_post_arm_duration_ns},
      {"inference_error", inference_error},
      {"writer_error", writer_error},
      {"publisher_write_failed", publisher_write_failed},
  };
  if (!armed_transition_observed) {
    execution_evidence.Finalize("session_no_actuation", terminal);
    std::cerr << "[NO ACTUATION] no armed policy command was written; evidence: "
              << execution_evidence.path() << '\n';
    return 3;
  }
  if (!passed) {
    execution_evidence.Finalize("session_failed", terminal);
    std::cerr << "[STOPPED] stage-one evidence failed: fault="
              << active::FaultName(fault)
              << " policy_command_frames=" << policy_command_frames
              << " damping_frames=" << damping_frames_after_stop << '\n';
    return 2;
  }
  execution_evidence.Finalize("session_complete", terminal);
  std::cout << "[COMPLETE] policy_command_frames=" << policy_command_frames
            << " damping_frames=" << damping_frames_after_stop
            << " evidence=" << execution_evidence.path() << '\n';
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
    std::cerr << "[BLOCKED] true23 active gantry controller: "
              << error.what() << '\n';
    if (argc > 0) {
      std::cerr << Usage(argv[0]) << '\n';
    }
    return 1;
  }
}
