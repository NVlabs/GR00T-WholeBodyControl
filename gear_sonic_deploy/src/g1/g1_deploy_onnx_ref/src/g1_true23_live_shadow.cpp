// Native G1 rev-1.0 True23 causal live-shadow inference.
//
// This process is intentionally read-only: immutable ONNX validation, a
// LowState subscriber, and a PICO ZMQ subscriber are its complete external
// surface. It has no robot command type, command topic, mode switcher, or
// writer. Any contract, freshness, continuity, numeric, or limit failure
// terminates inference after recording fail-closed evidence. The explicit
// native124 rejected-frame diagnostic can continue only per-action rejection;
// its terminal record is permanently failed and active-runtime-ineligible.

#include "true23_live_shadow_core.hpp"

#include <onnxruntime_cxx_api.h>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <zmq.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace true23 = gear_sonic::true23;
namespace live = gear_sonic::true23::live;
using LowState = unitree_hg::msg::dds_::LowState_;
using nlohmann::json;

inline constexpr std::string_view kLowStateTopic = "rt/lowstate";
inline constexpr std::string_view kSelectedNative124PolicySha256 =
    "cc644839807b6ef522e47b3bcb69845843aa345b4fb895847c76642830b5d2b9";
inline constexpr std::int64_t kStateFreshnessNs = 40'000'000;
// High-level XR24/reference latency is bounded independently from the local
// 40 ms LowState gate.  Forty milliseconds remains the publisher performance
// target; the read-only integration gate rejects only above 100 ms.
inline constexpr std::int64_t kPicoFreshnessNs = 100'000'000;
inline constexpr std::int64_t kFutureClockToleranceNs = 5'000'000;
inline constexpr std::int64_t kInferenceDeadlineNs = 20'000'000;
inline constexpr std::int64_t kHistoryWarmupSpanNs = 40'000'000;
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
  std::string mode = "shadow";
  std::string network;
  fs::path encoder;
  fs::path decoder;
  fs::path metadata;
  fs::path native124_policy;
  std::optional<fs::path> promotion;
  std::string pico_endpoint;
  fs::path evidence;
  int frames = 100;
  int timeout_seconds = 20;
  bool validate_only = false;
  bool continue_rejected_diagnostic = false;
  bool describe_contract = false;
  bool help = false;
};

std::string Usage(std::string_view executable) {
  return
      "Usage: " + std::string(executable) +
      " --mode shadow --encoder <pair.encoder.onnx>"
      " --decoder <pair.decoder.onnx> --metadata <candidate.json>"
      " [--promotion <promotion.json>]"
      " --network <interface> --pico-endpoint <tcp://host:port>"
      " --evidence <new.jsonl> [--frames 100] [--timeout-seconds 20]\n"
      "       " + std::string(executable) +
      " --mode shadow --native124-policy <OldTownRoad_v1.onnx>"
      " --network <interface> --pico-endpoint <tcp://host:port>"
      " --evidence <new.jsonl> [--frames 100] [--timeout-seconds 20]"
      " [--continue-rejected-diagnostic]\n"
      "       " + std::string(executable) +
      " --validate-only --encoder <pair.encoder.onnx>"
      " --decoder <pair.decoder.onnx> --metadata <candidate.json>"
      " [--promotion <promotion.json>]\n"
      "       " + std::string(executable) + " --describe-contract\n\n"
      "Causal read-only inference. Diagnostic ONNX is accepted only in "
      "shadow mode. Requires g1_true23_causal_history_reference_terms, "
      "exact consecutive 20 ms frames, and q9/q10 LowState brackets.";
}

int ParsePositiveInt(std::string_view text, std::string_view option) {
  std::size_t consumed = 0;
  const auto value = std::stoll(std::string(text), &consumed, 10);
  if (consumed != text.size() || value <= 0 || value > 100'000) {
    throw std::runtime_error(
        std::string(option) + " must be a positive bounded integer");
  }
  return static_cast<int>(value);
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
    } else if (option == "--describe-contract") {
      result.describe_contract = true;
    } else if (option == "--validate-only") {
      result.validate_only = true;
    } else if (option == "--continue-rejected-diagnostic") {
      result.continue_rejected_diagnostic = true;
    } else if (option == "--mode") {
      result.mode = value(index, option);
    } else if (option == "--network") {
      result.network = value(index, option);
    } else if (option == "--encoder") {
      result.encoder = value(index, option);
    } else if (option == "--decoder") {
      result.decoder = value(index, option);
    } else if (option == "--metadata") {
      result.metadata = value(index, option);
    } else if (option == "--native124-policy") {
      result.native124_policy = value(index, option);
    } else if (option == "--promotion") {
      result.promotion = value(index, option);
    } else if (option == "--pico-endpoint") {
      result.pico_endpoint = value(index, option);
    } else if (option == "--evidence") {
      result.evidence = value(index, option);
    } else if (option == "--frames") {
      result.frames = ParsePositiveInt(value(index, option), option);
    } else if (option == "--timeout-seconds") {
      result.timeout_seconds =
          ParsePositiveInt(value(index, option), option);
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.help || result.describe_contract) {
    return result;
  }
  if (result.mode != "shadow") {
    throw std::runtime_error("--mode must be shadow");
  }
  const bool native124 = !result.native124_policy.empty();
  const bool paired = !result.encoder.empty() || !result.decoder.empty() ||
                      !result.metadata.empty();
  if (native124 == paired) {
    throw std::runtime_error(
        "select exactly one backend: --native124-policy or encoder/decoder/metadata");
  }
  if (paired && (result.encoder.empty() || result.decoder.empty() ||
                 result.metadata.empty())) {
    throw std::runtime_error(
        "--encoder, --decoder, and --metadata are required");
  }
  if (!result.validate_only &&
      (result.network.empty() || result.pico_endpoint.empty() ||
       result.evidence.empty())) {
    throw std::runtime_error(
        "live shadow requires --network, --pico-endpoint, and --evidence");
  }
  if (!result.validate_only &&
      !result.pico_endpoint.starts_with("tcp://") &&
      !result.pico_endpoint.starts_with("ipc://")) {
    throw std::runtime_error(
        "PICO endpoint must use explicit tcp:// or ipc:// URI");
  }
  if (result.continue_rejected_diagnostic &&
      (!native124 || result.validate_only ||
       result.frames != live::kNative124RejectedDiagnosticFrames)) {
    throw std::runtime_error(
        "--continue-rejected-diagnostic requires live native124 and exactly 100 frames");
  }
  return result;
}

json DescribeContract() {
  return {
      {"schema_version", live::kCausalPicoTermsSchemaVersion},
      {"kind", live::kCausalReferenceTermsKind},
      {"reference_profile", live::kCausalReferenceProfile},
      {"reference_contract_sha256",
       live::kCausalReferenceContractSha256},
      {"control_derivative_contract", live::kCausalDerivativeContract},
      {"encoder_input_dim", true23::kEncoderInputDim},
      {"decoder_input_dim", true23::kDecoderInputDim},
      {"decoder_output_dim", true23::kDecoderOutputDim},
      {"q9_robot_anchor", true},
      {"q10_control_proprio", true},
      {"sdk_derivatives_consumed", false},
      {"robot_mutation_authorized", false},
  };
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
            throw std::runtime_error(
                std::string(role) + " parser underflow");
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
      throw std::runtime_error("ONNX inference returned invalid output");
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
    std::cout << "[LOAD] Inspecting ONNX signature: " << path_ << '\n';
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
    std::cout << "[LOAD] Inspecting ONNX opset bytes.\n";
    signature_.default_opsets =
        true23::InspectDefaultOnnxOpsetsFile(path_.string());
    std::cout << "[LOAD] Inspecting ONNX custom metadata.\n";
    auto metadata = session_.GetModelMetadata();
    auto keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    if (keys.size() != 1 || !keys.front() ||
        std::string_view(keys.front().get()) != true23::kOnnxMetadataKey) {
      throw std::runtime_error("ONNX embedded metadata key set is not exact");
    }
    auto value = metadata.LookupCustomMetadataMapAllocated(
        std::string(true23::kOnnxMetadataKey).c_str(), allocator);
    if (!value) {
      throw std::runtime_error("ONNX embedded metadata is missing");
    }
    embedded_ = ParseStrictJson(value.get(), "embedded ONNX metadata");
    std::cout << "[LOAD] ONNX inspection complete.\n";
  }

  fs::path path_;
  Ort::SessionOptions options_;
  Ort::Session session_;
  true23::ModelSignature signature_;
  json embedded_;
};

class Native124OnnxModel {
 public:
  Native124OnnxModel(Ort::Env& environment, fs::path path)
      : path_(std::move(path)),
        options_(SessionOptions()),
        session_(environment, path_.c_str(), options_) {
    Ort::AllocatorWithDefaultOptions allocator;
    if (session_.GetInputCount() != 2 || session_.GetOutputCount() < 1) {
      throw std::runtime_error("native124 ONNX input/output count mismatch");
    }
    const auto require_tensor = [](const Ort::TypeInfo& type,
                                   std::span<const std::int64_t> shape,
                                   std::string_view role) {
      const auto info = type.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          info.GetShape() != std::vector<std::int64_t>(shape.begin(), shape.end())) {
        throw std::runtime_error(
            "native124 ONNX " + std::string(role) + " type/shape mismatch");
      }
    };
    auto obs_name = session_.GetInputNameAllocated(0, allocator);
    auto time_name = session_.GetInputNameAllocated(1, allocator);
    if (!obs_name || !time_name || std::string_view(obs_name.get()) != "obs" ||
        std::string_view(time_name.get()) != "time_step") {
      throw std::runtime_error("native124 ONNX input names mismatch");
    }
    const std::array<std::int64_t, 2> obs_shape = {1, 124};
    const std::array<std::int64_t, 2> time_shape = {1, 1};
    require_tensor(session_.GetInputTypeInfo(0), obs_shape, "obs");
    require_tensor(session_.GetInputTypeInfo(1), time_shape, "time_step");
    bool found_actions = false;
    for (std::size_t index = 0; index < session_.GetOutputCount(); ++index) {
      auto name = session_.GetOutputNameAllocated(index, allocator);
      if (name && std::string_view(name.get()) == "actions") {
        const std::array<std::int64_t, 2> action_shape = {1, 23};
        require_tensor(
            session_.GetOutputTypeInfo(index), action_shape, "actions");
        found_actions = true;
        break;
      }
    }
    if (!found_actions) {
      throw std::runtime_error("native124 ONNX actions output missing");
    }
  }

  std::array<float, 23> Run(
      const std::array<float, live::kNative124ObservationDim>& observation) {
    if (!live::IsFinite(observation)) {
      throw std::invalid_argument("native124 observation is non-finite");
    }
    auto memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::array<std::int64_t, 2> obs_shape = {1, 124};
    std::array<std::int64_t, 2> time_shape = {1, 1};
    std::array<float, 1> time_step = {0.0F};
    auto obs_tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(observation.data()), observation.size(),
        obs_shape.data(), obs_shape.size());
    auto time_tensor = Ort::Value::CreateTensor<float>(
        memory, time_step.data(), time_step.size(),
        time_shape.data(), time_shape.size());
    std::array<Ort::Value, 2> inputs = {
        std::move(obs_tensor), std::move(time_tensor)};
    constexpr std::array<const char*, 2> input_names = {"obs", "time_step"};
    constexpr std::array<const char*, 1> output_names = {"actions"};
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());
    if (outputs.size() != 1 || !outputs.front().IsTensor()) {
      throw std::runtime_error("native124 ONNX inference output mismatch");
    }
    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetElementCount() != 23) {
      throw std::runtime_error("native124 ONNX action type/shape changed");
    }
    std::array<float, 23> result{};
    std::copy_n(outputs.front().GetTensorData<float>(), 23, result.begin());
    if (!live::IsFinite(result)) {
      throw std::runtime_error("native124 ONNX produced non-finite action");
    }
    return result;
  }

 private:
  fs::path path_;
  Ort::SessionOptions options_;
  Ort::Session session_;
};

struct LoadedArtifacts {
  fs::path encoder;
  fs::path decoder;
  fs::path metadata;
  std::optional<fs::path> promotion;
  json candidate;
  std::optional<json> promotion_json;
  std::string encoder_sha;
  std::string decoder_sha;
  std::string metadata_sha;
  std::optional<std::string> promotion_sha;
};

LoadedArtifacts LoadArtifacts(const Arguments& arguments) {
  LoadedArtifacts result;
  result.encoder = ResolveFile(arguments.encoder, "encoder");
  result.decoder = ResolveFile(arguments.decoder, "decoder");
  result.metadata = ResolveFile(arguments.metadata, "candidate metadata");
  if (arguments.promotion.has_value()) {
    result.promotion = ResolveFile(*arguments.promotion, "promotion");
  }
  std::vector<fs::path> paths = {
      result.encoder, result.decoder, result.metadata};
  if (result.promotion.has_value()) {
    paths.push_back(*result.promotion);
  }
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
  result.candidate = LoadJson(result.metadata, "candidate metadata");
  if (result.promotion.has_value()) {
    result.promotion_sha = true23::Sha256File(result.promotion->string());
    result.promotion_json = LoadJson(*result.promotion, "promotion");
  }
  return result;
}

void VerifyUnchanged(const LoadedArtifacts& files) {
  if (true23::Sha256File(files.encoder.string()) != files.encoder_sha ||
      true23::Sha256File(files.decoder.string()) != files.decoder_sha ||
      true23::Sha256File(files.metadata.string()) != files.metadata_sha ||
      (files.promotion.has_value() &&
       true23::Sha256File(files.promotion->string()) !=
           files.promotion_sha.value())) {
    throw std::runtime_error("artifact changed during validation/inference");
  }
}

class EvidenceLog {
 public:
  explicit EvidenceLog(const fs::path& requested) {
    std::error_code error;
    auto absolute = fs::absolute(requested, error);
    if (error || absolute.filename().empty()) {
      throw std::runtime_error("evidence path is invalid");
    }
    auto parent = absolute.parent_path();
    fs::create_directories(parent, error);
    if (error) {
      throw std::runtime_error("cannot create evidence directory");
    }
    parent = fs::canonical(parent, error);
    if (error) {
      throw std::runtime_error("cannot resolve evidence directory");
    }
    path_ = parent / absolute.filename();
    if (fs::exists(path_, error) || error) {
      throw std::runtime_error("evidence output must not already exist");
    }
    stream_.open(path_, std::ios::binary | std::ios::out);
    if (!stream_) {
      throw std::runtime_error("cannot create evidence output");
    }
  }

  void Write(const json& record) {
    stream_ << record.dump() << '\n';
    stream_.flush();
    if (!stream_) {
      throw std::runtime_error("evidence write failed");
    }
  }

  [[nodiscard]] const fs::path& path() const { return path_; }

 private:
  fs::path path_;
  std::ofstream stream_;
};

class ReadOnlyStateMonitor {
 public:
  ~ReadOnlyStateMonitor() {
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

  void WaitUntilReady(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    condition_.wait_until(lock, deadline, [&] {
      return !fault_.empty() ||
             (mode_gate_.ready() && first_history_ns_ > 0 &&
              last_history_ns_ - first_history_ns_ >=
                  kHistoryWarmupSpanNs);
    });
    ThrowFaultLocked();
    if (!mode_gate_.ready() || first_history_ns_ <= 0 ||
        last_history_ns_ - first_history_ns_ < kHistoryWarmupSpanNs) {
      throw std::runtime_error(
          "LowState did not provide mode_machine==4 plus causal history "
          "warmup before timeout");
    }
  }

  live::CausalEncoderJoin WaitForJoin(
      const live::CausalPicoReferenceTerms& reference,
      std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    while (true) {
      ThrowFaultLocked();
      const auto now = NowNs();
      if (last_history_ns_ <= 0 ||
          now - last_history_ns_ > kStateFreshnessNs) {
        throw std::runtime_error("LowState became stale before causal join");
      }
      if (history_.Covers(
              reference.pico_anchor_monotonic_ns,
              reference.control_monotonic_ns)) {
        return live::JoinCausalReferenceWithLowState(reference, history_);
      }
      if (condition_.wait_until(lock, deadline) ==
          std::cv_status::timeout) {
        throw std::runtime_error(
            "LowState history did not bracket exact causal q9/q10");
      }
    }
  }

  std::int64_t LatestAgeNs(std::int64_t now_ns) const {
    std::lock_guard lock(mutex_);
    ThrowFaultLocked();
    if (last_history_ns_ <= 0) {
      throw std::runtime_error("LowState history is empty");
    }
    const auto age = now_ns - last_history_ns_;
    if (age < -kFutureClockToleranceNs || age > kStateFreshnessNs) {
      throw std::runtime_error("LowState freshness contract failed");
    }
    return age;
  }

  [[nodiscard]] std::uint64_t crc_rejects() const {
    return crc_rejects_.load(std::memory_order_relaxed);
  }

 private:
  void ThrowFaultLocked() const {
    if (!fault_.empty()) {
      throw std::runtime_error(fault_);
    }
  }

  void LatchFault(std::string message) {
    if (fault_.empty()) {
      fault_ = std::move(message);
    }
  }

  void OnMessage(const void* message) {
    if (message == nullptr) {
      return;
    }
    LowState state = *static_cast<const LowState*>(message);
    const auto received = NowNs();
    if (state.crc() != Crc32(&state, (sizeof(LowState) >> 2U) - 1U)) {
      crc_rejects_.fetch_add(1, std::memory_order_relaxed);
      std::lock_guard lock(mutex_);
      LatchFault("LowState CRC failure");
      condition_.notify_all();
      return;
    }

    std::lock_guard lock(mutex_);
    if (!fault_.empty()) {
      return;
    }
    const auto observation =
        mode_gate_.Observe(state.tick(), state.mode_machine());
    if (observation == true23::ModeObservation::TickRegression) {
      LatchFault("LowState tick regression");
      condition_.notify_all();
      return;
    }
    if (observation == true23::ModeObservation::LatchedFailure) {
      LatchFault("mode_machine changed away from 4");
      condition_.notify_all();
      return;
    }
    if (observation == true23::ModeObservation::WrongMode ||
        observation == true23::ModeObservation::DuplicateTick) {
      condition_.notify_all();
      return;
    }

    live::TimedProprioSample sample;
    sample.received_monotonic_ns = received;
    for (std::size_t compact = 0; compact < 23; ++compact) {
      const auto slot = static_cast<std::size_t>(
          true23::kHardwareJointIds[compact]);
      const auto q = state.motor_state()[slot].q();
      const auto dq = state.motor_state()[slot].dq();
      if (!std::isfinite(q) || !std::isfinite(dq) ||
          q < live::kHardwareLowerLimit[compact] ||
          q > live::kHardwareUpperLimit[compact] ||
          std::abs(dq) > live::kHardwareVelocityLimit[compact]) {
        LatchFault("LowState required-joint telemetry failed finite/limit gate");
        condition_.notify_all();
        return;
      }
      sample.hardware_q[compact] = q;
      sample.hardware_dq[compact] = dq;
    }
    const auto& imu = state.imu_state();
    std::copy_n(imu.gyroscope().begin(), 3, sample.gyroscope.begin());
    std::copy_n(imu.quaternion().begin(), 4, sample.quaternion_wxyz.begin());
    if (!history_.Push(sample)) {
      LatchFault("LowState causal history rejected timestamp/IMU/telemetry");
      condition_.notify_all();
      return;
    }
    if (first_history_ns_ == 0) {
      first_history_ns_ = received;
    }
    last_history_ns_ = received;
    condition_.notify_all();
  }

  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  true23::ModeMachineShadowGate mode_gate_;
  live::CausalLowStateHistory history_;
  std::string fault_;
  std::int64_t first_history_ns_ = 0;
  std::int64_t last_history_ns_ = 0;
  std::atomic<std::uint64_t> crc_rejects_{0};
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
    constexpr int receive_hwm = 1000;
    constexpr int linger_ms = 0;
    if (zmq_setsockopt(socket_, ZMQ_SUBSCRIBE, "", 0) != 0 ||
        zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &timeout_ms,
                       sizeof(timeout_ms)) != 0 ||
        zmq_setsockopt(socket_, ZMQ_RCVHWM, &receive_hwm,
                       sizeof(receive_hwm)) != 0 ||
        zmq_setsockopt(socket_, ZMQ_LINGER, &linger_ms,
                       sizeof(linger_ms)) != 0 ||
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
      throw std::runtime_error("PICO packet exceeds 1 MiB limit");
    }
    return std::string(buffer.data(), static_cast<std::size_t>(size));
  }

 private:
  void* context_ = nullptr;
  void* socket_ = nullptr;
};

live::CausalPicoReferenceTerms ParseCausalPacket(
    std::string_view message) {
  return live::ParseCausalPicoReferenceTermsDocument(
      ParseStrictJson(message, "causal PICO reference terms"));
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
          std::string(context) + " is missing field: " +
          std::string(key));
    }
  }
}

void RequireSha256(const json& value, std::string_view context) {
  if (!value.is_string()) {
    throw std::runtime_error(std::string(context) + " must be SHA-256");
  }
  const auto text = value.get<std::string>();
  if (text.size() != 64 ||
      !std::all_of(text.begin(), text.end(), [](char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
      })) {
    throw std::runtime_error(
        std::string(context) + " must be lowercase SHA-256");
  }
}

void ValidateDiagnosticGraph(
    const json& graph,
    const true23::ModelSignature& signature,
    std::string_view input_name,
    std::int64_t input_dim,
    std::string_view output_name,
    std::int64_t output_dim,
    std::string_view context) {
  RequireExactKeys(
      graph,
      {"input_name", "input_shape", "input_dtype", "output_name",
       "output_shape", "output_dtype", "dynamic_axes"},
      context);
  if (graph.at("input_name") != input_name ||
      graph.at("input_shape") != json::array({1, input_dim}) ||
      graph.at("input_dtype") != "float32" ||
      graph.at("output_name") != output_name ||
      graph.at("output_shape") != json::array({1, output_dim}) ||
      graph.at("output_dtype") != "float32" ||
      graph.at("dynamic_axes") != false ||
      signature.input_count != 1 || signature.output_count != 1 ||
      signature.input_name != input_name ||
      signature.output_name != output_name ||
      signature.input_shape != std::vector<std::int64_t>{1, input_dim} ||
      signature.output_shape != std::vector<std::int64_t>{1, output_dim} ||
      !signature.input_float32 || !signature.output_float32 ||
      signature.default_opsets != std::vector<std::int64_t>{13}) {
    throw std::runtime_error(
        std::string(context) + " graph/signature contract mismatch");
  }
}

void ValidateDiagnosticParity(const json& value, std::string_view context) {
  RequireExactKeys(
      value,
      {"onnx_checker_full_check", "shape_inference", "ort_provider",
       "parity_case_count", "parity_atol", "parity_rtol",
       "parity_max_abs_error", "parity_max_rel_error",
       "parity_inputs_sha256", "parity_outputs_sha256"},
      context);
  if (value.at("onnx_checker_full_check") != true ||
      value.at("shape_inference") != true ||
      value.at("ort_provider") != "CPUExecutionProvider" ||
      value.at("parity_case_count") != 3 ||
      value.at("parity_atol") != 1e-5 ||
      value.at("parity_rtol") != 1e-5) {
    throw std::runtime_error(
        std::string(context) + " validation contract mismatch");
  }
  for (const auto key : {"parity_max_abs_error", "parity_max_rel_error"}) {
    if (!value.at(key).is_number() ||
        !std::isfinite(value.at(key).get<double>()) ||
        value.at(key).get<double>() < 0.0) {
      throw std::runtime_error(
          std::string(context) + " contains invalid parity metric");
    }
  }
  RequireSha256(value.at("parity_inputs_sha256"), context);
  RequireSha256(value.at("parity_outputs_sha256"), context);
}

void ValidateDiagnosticEmbedded(
    const json& embedded,
    const json& hashes,
    std::string_view role,
    std::string_view input_name,
    std::int64_t input_dim,
    std::string_view output_name,
    std::int64_t output_dim,
    int update_count,
    std::string_view expected_hash) {
  RequireExactKeys(
      embedded,
      {"schema_version", "kind", "artifact_role", "diagnostic_only",
       "deployment_ready", "promotion_eligible",
       "active_motor_control_authorized", "checkpoint_role",
       "robot_model", "required_mode_machine", "checkpoint_update_count",
       "reference_profile", "input_name", "input_shape", "output_name",
       "output_shape", "input_dtype", "output_dtype", "onnx_opset",
       "checkpoint_sha256", "lineage_sha256", "policy_state_sha256",
       "encoder_state_sha256", "decoder_state_sha256",
       "decoder_output_semantics",
       "external_safe_target_transform_allowed",
       "safe_target_transform_sha256"},
      "diagnostic embedded metadata");
  if (embedded.at("schema_version") !=
          live::kSafeTargetDiagnosticSchemaVersion ||
      embedded.at("kind") != "g1_true23_mjlab_diagnostic_onnx" ||
      embedded.at("artifact_role") != role ||
      embedded.at("diagnostic_only") != true ||
      embedded.at("deployment_ready") != false ||
      embedded.at("promotion_eligible") != false ||
      embedded.at("active_motor_control_authorized") != false ||
      embedded.at("checkpoint_role") != "training_resume_only" ||
      embedded.at("robot_model") != true23::kRobotModel ||
      embedded.at("required_mode_machine") !=
          true23::kRequiredModeMachine ||
      embedded.at("checkpoint_update_count") != update_count ||
      embedded.at("reference_profile") != live::kCausalReferenceProfile ||
      embedded.at("input_name") != input_name ||
      embedded.at("input_shape") != json::array({1, input_dim}) ||
      embedded.at("output_name") != output_name ||
      embedded.at("output_shape") != json::array({1, output_dim}) ||
      embedded.at("input_dtype") != "float32" ||
      embedded.at("output_dtype") != "float32" ||
      embedded.at("onnx_opset") != 13) {
    throw std::runtime_error("diagnostic embedded metadata contract mismatch");
  }
  for (const auto key : {
           "checkpoint_sha256", "lineage_sha256", "policy_state_sha256",
           "encoder_state_sha256", "decoder_state_sha256"}) {
    RequireSha256(embedded.at(key), key);
    if (embedded.at(key) != hashes.at(key)) {
      throw std::runtime_error(
          "diagnostic embedded/checkpoint material binding mismatch");
    }
  }
  if (true23::Sha256CanonicalJson(embedded) != expected_hash) {
    throw std::runtime_error("diagnostic embedded metadata hash mismatch");
  }
}

void ValidateDiagnosticPair(
    const LoadedArtifacts& files,
    const OnnxModel& encoder,
    const OnnxModel& decoder) {
  const auto& root = files.candidate;
  RequireExactKeys(
      root,
      {"schema_version", "kind", "diagnostic_only", "deployment_ready",
       "promotion_eligible", "active_motor_control_authorized",
       "checkpoint_role", "allowed_uses", "forbidden_uses",
       "no_robot_or_network_commands_performed", "source", "contract",
       "artifacts", "hashes", "validation",
       "metadata_payload_sha256"},
      "diagnostic bundle");
  if (root.at("schema_version") !=
          live::kSafeTargetDiagnosticSchemaVersion ||
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
    throw std::runtime_error("diagnostic bundle safety flags mismatch");
  }

  const auto& source = root.at("source");
  RequireExactKeys(
      source,
      {"checkpoint_filename", "checkpoint_update_count",
       "reference_profile", "simulation_candidate_review_allowed"},
      "diagnostic source");
  if (!source.at("checkpoint_update_count").is_number_integer() ||
      source.at("checkpoint_update_count").get<int>() <= 0 ||
      source.at("reference_profile") != live::kCausalReferenceProfile ||
      source.at("simulation_candidate_review_allowed") != true) {
    throw std::runtime_error("diagnostic source contract mismatch");
  }
  const auto update_count = source.at("checkpoint_update_count").get<int>();

  const auto& contract = root.at("contract");
  RequireExactKeys(
      contract,
      {"robot_model", "required_mode_machine", "native_action_dof",
       "history_length", "observation_layout", "onnx_opset", "encoder",
       "decoder", "decoder_output_semantics",
       "external_safe_target_transform_allowed",
       "previous_action_semantics", "safe_target_transform"},
      "diagnostic contract");
  if (contract.at("robot_model") != true23::kRobotModel ||
      contract.at("required_mode_machine") != true23::kRequiredModeMachine ||
      contract.at("native_action_dof") != true23::kDecoderOutputDim ||
      contract.at("history_length") != true23::kHistoryLength ||
      contract.at("observation_layout") !=
          "canonical_il29_fixed_slots_v1" ||
      contract.at("onnx_opset") != 13) {
    throw std::runtime_error("diagnostic pair contract mismatch");
  }
  ValidateDiagnosticGraph(
      contract.at("encoder"), encoder.signature(), "teleop_obs", 267,
      "token", 64, "diagnostic encoder");
  ValidateDiagnosticGraph(
      contract.at("decoder"), decoder.signature(), "obs_dict", 994,
      "action", 23, "diagnostic decoder");

  const auto& artifacts = root.at("artifacts");
  RequireExactKeys(
      artifacts,
      {"encoder_onnx_filename", "decoder_onnx_filename",
       "metadata_filename"},
      "diagnostic artifacts");
  if (artifacts.at("encoder_onnx_filename") !=
          files.encoder.filename().string() ||
      artifacts.at("decoder_onnx_filename") !=
          files.decoder.filename().string() ||
      artifacts.at("metadata_filename") !=
          files.metadata.filename().string()) {
    throw std::runtime_error("diagnostic artifact filename binding mismatch");
  }

  const auto& hashes = root.at("hashes");
  RequireExactKeys(
      hashes,
      {"checkpoint_sha256", "lineage_sha256", "policy_state_sha256",
       "encoder_state_sha256", "decoder_state_sha256",
       "encoder_onnx_sha256", "decoder_onnx_sha256",
       "encoder_embedded_metadata_sha256",
       "decoder_embedded_metadata_sha256",
       "safe_target_transform_sha256"},
      "diagnostic hashes");
  for (const auto& [key, value] : hashes.items()) {
    RequireSha256(value, key);
  }
  if (hashes.at("encoder_onnx_sha256") != files.encoder_sha ||
      hashes.at("decoder_onnx_sha256") != files.decoder_sha) {
    throw std::runtime_error("diagnostic ONNX file hash mismatch");
  }
  ValidateDiagnosticEmbedded(
      encoder.embedded(), hashes, "teleop_encoder", "teleop_obs", 267,
      "token", 64, update_count,
      hashes.at("encoder_embedded_metadata_sha256").get<std::string>());
  ValidateDiagnosticEmbedded(
      decoder.embedded(), hashes, "true23_decoder", "obs_dict", 994,
      "action", 23, update_count,
      hashes.at("decoder_embedded_metadata_sha256").get<std::string>());
  live::ValidateAppliedSafeDecoderContract(
      contract, hashes, encoder.embedded(), decoder.embedded());

  const auto& validation = root.at("validation");
  RequireExactKeys(
      validation,
      {"weights_only_checkpoint_validated", "exact_policy_reconstructed",
       "simulation_candidate_review_gate_validated", "teleop_encoder",
       "true23_decoder", "paired_inference"},
      "diagnostic validation");
  if (validation.at("weights_only_checkpoint_validated") != true ||
      validation.at("exact_policy_reconstructed") != true ||
      validation.at("simulation_candidate_review_gate_validated") != true) {
    throw std::runtime_error("diagnostic reconstruction validation mismatch");
  }
  ValidateDiagnosticParity(
      validation.at("teleop_encoder"), "diagnostic teleop encoder");
  ValidateDiagnosticParity(
      validation.at("true23_decoder"), "diagnostic decoder");
  const auto& paired = validation.at("paired_inference");
  RequireExactKeys(
      paired,
      {"performed", "provider", "dtype", "case_count",
       "all_outputs_finite", "parity_atol", "parity_rtol",
       "max_token_abs_error", "max_action_abs_error", "outputs_sha256"},
      "diagnostic paired inference");
  if (paired.at("performed") != true ||
      paired.at("provider") != "CPUExecutionProvider" ||
      paired.at("dtype") != "float32" || paired.at("case_count") != 3 ||
      paired.at("all_outputs_finite") != true ||
      paired.at("parity_atol") != 1e-5 ||
      paired.at("parity_rtol") != 1e-5) {
    throw std::runtime_error("diagnostic paired inference mismatch");
  }
  for (const auto key : {"max_token_abs_error", "max_action_abs_error"}) {
    if (!paired.at(key).is_number() ||
        !std::isfinite(paired.at(key).get<double>()) ||
        paired.at(key).get<double>() < 0.0) {
      throw std::runtime_error("diagnostic paired metric invalid");
    }
  }
  RequireSha256(paired.at("outputs_sha256"), "paired outputs_sha256");

  RequireSha256(
      root.at("metadata_payload_sha256"), "metadata_payload_sha256");
  auto payload = root;
  const auto expected_payload_hash =
      payload.at("metadata_payload_sha256").get<std::string>();
  payload.erase("metadata_payload_sha256");
  if (true23::Sha256CanonicalJson(payload) != expected_payload_hash) {
    throw std::runtime_error("diagnostic metadata payload hash mismatch");
  }
}

void ValidateCausalMujocoPromotion(const LoadedArtifacts& files) {
  if (!files.promotion_json.has_value()) {
    throw std::runtime_error("causal promotion sidecar is missing");
  }
  const auto& root = *files.promotion_json;
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

  const auto& candidate = files.candidate;
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
    RequireSha256(source.at(key), key);
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
    RequireSha256(campaign.at(key), key);
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
    RequireSha256(provenance.at(key), key);
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
      !metrics.at("maximum_recovery_time_s").is_number() ||
      metrics.at("maximum_recovery_time_s").get<double>() > 2.0 ||
      metrics.at("max_action_saturation_fraction") != 0.0 ||
      !metrics.at("max_joint_velocity_ratio").is_number() ||
      metrics.at("max_joint_velocity_ratio").get<double>() > 1.0 ||
      !metrics.at("max_effort_ratio").is_number() ||
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

  RequireSha256(root.at("promotion_payload_sha256"),
                "promotion_payload_sha256");
  const auto expected_payload =
      root.at("promotion_payload_sha256").get<std::string>();
  auto unhashed = root;
  unhashed.erase("promotion_payload_sha256");
  if (true23::Sha256CanonicalJson(unhashed) != expected_payload) {
    throw std::runtime_error("causal promotion payload hash mismatch");
  }
}

true23::ValidationResult ValidatePair(
    const LoadedArtifacts& files,
    const OnnxModel& encoder,
    const OnnxModel& decoder) {
  const true23::PairBinding binding{
      .encoder_filename = files.encoder.filename().string(),
      .decoder_filename = files.decoder.filename().string(),
      .metadata_filename = files.metadata.filename().string(),
      .encoder_onnx_sha256 = files.encoder_sha,
      .decoder_onnx_sha256 = files.decoder_sha,
      .encoder_path = files.encoder.string(),
      .decoder_path = files.decoder.string(),
      .metadata_path = files.metadata.string(),
      .metadata_sha256 = files.metadata_sha,
  };
  if (files.promotion_json.has_value()) {
    return true23::ValidateMujocoShadowPromotion(
        *files.promotion_json, files.candidate, encoder.embedded(),
        decoder.embedded(), encoder.signature(), decoder.signature(),
        binding, true23::RequestedMode::Shadow);
  }
  return true23::ValidateShadowArtifact(
      files.candidate, encoder.embedded(), decoder.embedded(),
      encoder.signature(), decoder.signature(), binding,
      true23::RequestedMode::Shadow);
}

void RequireCausalProfile(const LoadedArtifacts& files) {
  const auto diagnostic =
      files.candidate.value("kind", std::string{}) ==
      "g1_true23_mjlab_diagnostic_onnx_pair";
  const auto* profile = diagnostic
      ? &files.candidate.at("source").at("reference_profile")
      : (files.candidate.contains("reference_profile")
             ? &files.candidate.at("reference_profile")
             : nullptr);
  if (profile == nullptr || !profile->is_string() ||
      profile->get<std::string>() != live::kCausalReferenceProfile) {
    throw std::runtime_error(
        "live shadow requires exact causal reference profile");
  }
}

int RunNative124(const Arguments& arguments) {
  const auto policy_path = ResolveFile(
      arguments.native124_policy, "native124 public policy");
  const auto policy_sha = true23::Sha256File(policy_path.string());
  if (policy_sha != kSelectedNative124PolicySha256) {
    throw std::runtime_error("native124 selected policy SHA256 mismatch");
  }
  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "g1_true23_native124_shadow");
  Native124OnnxModel policy(environment, policy_path);
  const std::array<float, live::kNative124ObservationDim> zero_observation{};
  const auto dry_action = policy.Run(zero_observation);
  (void)live::Native124RawActionToClampedTargets(dry_action);
  if (true23::Sha256File(policy_path.string()) != policy_sha) {
    throw std::runtime_error("native124 public policy changed during dry run");
  }
  std::cout << "[PASS] Pinned native124 public ONNX validated and dry-ran. "
               "Backend remains clip-specific shadow-only.\n";
  if (arguments.validate_only) {
    std::cout << "[PASS] Validation-only completed; robot APIs unopened.\n";
    return 0;
  }

  EvidenceLog evidence(arguments.evidence);
  evidence.Write({
      {"schema_version", live::kEvidenceSchemaVersion},
      {"kind", live::kEvidenceKind},
      {"event", "session_start"},
      {"started_monotonic_ns", NowNs()},
      {"backend", "public_native124_clip_shadow_only"},
      {"policy_sha256", policy_sha},
      {"observation_dim", live::kNative124ObservationDim},
      {"action_dim", 23},
      {"reference_profile", live::kCausalReferenceProfile},
      {"pico_soft_latency_target_ms", 40},
      {"pico_hard_freshness_limit_ms", 100},
      {"lowstate_hard_freshness_limit_ms", 40},
      {"maximum_raw_action_abs", live::kNative124ShadowMaximumRawAction},
      {"maximum_raw_target_clamps_per_frame",
       live::kNative124ShadowMaximumRawTargetClamps},
      {"maximum_raw_target_excess_rad",
       live::kNative124ShadowMaximumRawTargetExcessRad},
      {"network", arguments.network},
      {"pico_endpoint", arguments.pico_endpoint},
      {"requested_action_frames", arguments.frames},
      {"clip_specific_policy", true},
      {"task_general_pico_policy", false},
      {"continue_rejected_diagnostic",
       arguments.continue_rejected_diagnostic},
      {"active_runtime_eligible", false},
      {"robot_mutation_authorized", false},
  });

  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, arguments.network);
    ReadOnlyStateMonitor monitor;
    monitor.Start();
    monitor.WaitUntilReady(
        std::chrono::steady_clock::now() + std::chrono::seconds(5));
    ZmqSocket socket(arguments.pico_endpoint);
    std::array<float, 23> previous_raw_action{};
    std::optional<std::array<double, 23>> previous_targets;
    std::optional<std::uint64_t> previous_frame;
    std::optional<std::int64_t> previous_control_ns;
    int action_frames = 0;
    int accepted_frames = 0;
    int rejected_frames = 0;
    std::optional<int> first_rejected_frame;
    int total_raw_target_clamps = 0;
    double maximum_raw_action_abs = 0.0;
    double maximum_raw_target_excess_rad = 0.0;
    double maximum_target_slew_rad = 0.0;
    const auto session_deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(arguments.timeout_seconds);

    while (action_frames < arguments.frames) {
      if (g_stop_requested != 0) {
        throw std::runtime_error("signal stopped native124 shadow");
      }
      if (std::chrono::steady_clock::now() >= session_deadline) {
        throw std::runtime_error("native124 PICO shadow session timed out");
      }
      const auto message = socket.Receive();
      if (!message.has_value()) {
        (void)monitor.LatestAgeNs(NowNs());
        continue;
      }
      const auto received_ns = NowNs();
      const auto reference = ParseCausalPacket(*message);
      const auto packet_age_ns = received_ns - reference.control_monotonic_ns;
      if (packet_age_ns < -kFutureClockToleranceNs ||
          packet_age_ns > kPicoFreshnessNs) {
        throw std::runtime_error("native124 PICO q10 hard freshness failed");
      }
      if (previous_frame.has_value() &&
          reference.control_source_frame_index != *previous_frame + 1) {
        throw std::runtime_error("native124 PICO frame continuity failed");
      }
      if (previous_control_ns.has_value() &&
          reference.control_monotonic_ns != *previous_control_ns + 20'000'000) {
        throw std::runtime_error("native124 PICO timestamp continuity failed");
      }
      previous_frame = reference.control_source_frame_index;
      previous_control_ns = reference.control_monotonic_ns;

      const auto joined = monitor.WaitForJoin(
          reference,
          std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
      const auto inference_started_ns = NowNs();
      const auto observation = live::BuildNative124Observation(
          reference, joined, previous_raw_action);
      const auto action = policy.Run(observation);
      const auto transformed =
          live::Native124RawActionToClampedTargets(action);
      const auto produced_ns = NowNs();
      const auto inference_ns = produced_ns - inference_started_ns;
      const auto end_to_end_age_ns =
          produced_ns - reference.control_monotonic_ns;
      const auto lowstate_age_ns = monitor.LatestAgeNs(produced_ns);
      double raw_action_abs = 0.0;
      for (const auto value : action) {
        raw_action_abs = std::max(raw_action_abs, std::abs(static_cast<double>(value)));
      }
      double target_slew_rad = 0.0;
      for (std::size_t index = 0; index < 23; ++index) {
        const auto prior_target = previous_targets.has_value()
                                      ? previous_targets->at(index)
                                      : static_cast<double>(
                                            joined.control_proprio_q10
                                                .hardware_q[index]);
        target_slew_rad = std::max(
            target_slew_rad,
            std::abs(transformed.hardware_targets[index] - prior_target));
      }
      const auto frame_assessment = live::AssessNative124ShadowFrame(
          raw_action_abs, inference_ns, lowstate_age_ns, end_to_end_age_ns,
          transformed.raw_target_limit_clamps,
          transformed.maximum_raw_target_excess_rad);
      const bool accepted = frame_assessment.accepted;
      evidence.Write({
          {"schema_version", live::kEvidenceSchemaVersion},
          {"kind", live::kEvidenceKind},
          {"event", "native124_action_frame"},
          {"action_frame_index", action_frames},
          {"control_source_frame_index", reference.control_source_frame_index},
          {"packet_age_ns", packet_age_ns},
          {"end_to_end_age_ns", end_to_end_age_ns},
          {"lowstate_age_ns", lowstate_age_ns},
          {"inference_ns", inference_ns},
          {"raw_action_native", action},
          {"clamped_hardware_targets", transformed.hardware_targets},
          {"raw_target_limit_clamps", transformed.raw_target_limit_clamps},
          {"maximum_raw_target_excess_rad",
           transformed.maximum_raw_target_excess_rad},
          {"target_slew_rad", target_slew_rad},
          {"target_slew_is_report_only", true},
          {"target_slew_source",
           previous_targets.has_value() ? "previous_policy_target"
                                        : "joined_q10_hardware_position"},
          {"external_urdf_target_clamp_applied", true},
          {"sdk_derivatives_consumed", false},
          {"robot_mutation_authorized", false},
          {"diagnostic_only", arguments.continue_rejected_diagnostic},
          {"active_runtime_eligible", false},
          {"accepted", accepted},
          {"rejection_reasons", frame_assessment.rejection_reasons},
      });
      if (!accepted) {
        ++rejected_frames;
        if (!first_rejected_frame.has_value()) {
          first_rejected_frame = action_frames;
        }
        if (!arguments.continue_rejected_diagnostic) {
          throw std::runtime_error("native124 action failed shadow gate");
        }
      } else {
        ++accepted_frames;
      }
      previous_raw_action = action;
      previous_targets = transformed.hardware_targets;
      ++action_frames;
      total_raw_target_clamps += transformed.raw_target_limit_clamps;
      maximum_raw_action_abs = std::max(maximum_raw_action_abs, raw_action_abs);
      maximum_raw_target_excess_rad = std::max(
          maximum_raw_target_excess_rad,
          transformed.maximum_raw_target_excess_rad);
      maximum_target_slew_rad =
          std::max(maximum_target_slew_rad, target_slew_rad);
    }
    if (true23::Sha256File(policy_path.string()) != policy_sha) {
      throw std::runtime_error("native124 public policy changed during shadow");
    }
    if (arguments.continue_rejected_diagnostic) {
      evidence.Write({
          {"schema_version", live::kEvidenceSchemaVersion},
          {"kind", live::kEvidenceKind},
          {"event", "diagnostic_session_complete"},
          {"passed", false},
          {"diagnostic_only", true},
          {"active_runtime_eligible", false},
          {"action_frames", action_frames},
          {"accepted_frames", accepted_frames},
          {"rejected_frames", rejected_frames},
          {"first_rejected_frame",
           first_rejected_frame.has_value() ? json(*first_rejected_frame)
                                            : json(nullptr)},
          {"total_raw_target_limit_clamps", total_raw_target_clamps},
          {"maximum_raw_action_abs", maximum_raw_action_abs},
          {"maximum_raw_target_excess_rad", maximum_raw_target_excess_rad},
          {"maximum_target_slew_rad", maximum_target_slew_rad},
          {"robot_mutation_authorized", false},
      });
      std::cout << "[DIAGNOSTIC] Native124 rejected-frame capture completed; "
                   "passed=false and active-runtime-ineligible.\n";
      return 2;
    }
    evidence.Write({
        {"schema_version", live::kEvidenceSchemaVersion},
        {"kind", live::kEvidenceKind},
        {"event", "session_complete"},
        {"passed", true},
        {"action_frames", action_frames},
        {"total_raw_target_limit_clamps", total_raw_target_clamps},
        {"maximum_raw_action_abs", maximum_raw_action_abs},
        {"maximum_raw_target_excess_rad", maximum_raw_target_excess_rad},
        {"maximum_target_slew_rad", maximum_target_slew_rad},
        {"clip_specific_policy", true},
        {"task_general_pico_policy", false},
        {"robot_mutation_authorized", false},
    });
    std::cout << "[PASS] Native124 live shadow completed; no command-publisher surface.\n";
    return 0;
  } catch (const std::exception& error) {
    evidence.Write({
        {"schema_version", live::kEvidenceSchemaVersion},
        {"kind", live::kEvidenceKind},
        {"event", "session_failed"},
        {"failure", error.what()},
        {"failed_monotonic_ns", NowNs()},
        {"robot_mutation_authorized", false},
    });
    throw;
  }
}

int Run(const Arguments& arguments) {
  if (!arguments.native124_policy.empty()) {
    return RunNative124(arguments);
  }
  std::cout << "[LOAD] Resolving and hashing artifacts.\n";
  auto files = LoadArtifacts(arguments);
  std::cout << "[LOAD] Creating encoder session.\n";
  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "g1_true23_live_shadow");
  OnnxModel encoder(environment, files.encoder);
  std::cout << "[LOAD] Creating decoder session.\n";
  OnnxModel decoder(environment, files.decoder);
  std::cout << "[LOAD] Validating artifact contracts.\n";
  const bool diagnostic_artifact =
      files.candidate.value("kind", std::string{}) ==
      "g1_true23_mjlab_diagnostic_onnx_pair";
  const bool causal_promoted_artifact =
      diagnostic_artifact && files.promotion.has_value();
  if (diagnostic_artifact) {
    ValidateDiagnosticPair(files, encoder, decoder);
    if (causal_promoted_artifact) {
      ValidateCausalMujocoPromotion(files);
    }
  } else {
    const auto validation = ValidatePair(files, encoder, decoder);
    validation.Require();
    if (validation.authorization.lowcmd_publisher_allowed ||
        validation.authorization.motion_mode_release_allowed ||
        validation.authorization.command_writer_allowed) {
      throw std::logic_error("shadow validator granted mutation authority");
    }
  }
  RequireCausalProfile(files);

  const std::array<float, true23::kEncoderInputDim> zero_encoder{};
  const auto token = encoder.Run<true23::kEncoderInputDim,
                                 true23::kEncoderOutputDim>(zero_encoder);
  std::array<float, true23::kDecoderInputDim> zero_decoder{};
  std::copy(token.begin(), token.end(), zero_decoder.begin());
  (void)decoder.Run<true23::kDecoderInputDim,
                    true23::kDecoderOutputDim>(zero_decoder);
  VerifyUnchanged(files);

  std::cout
      << "[PASS] Exact causal "
      << (diagnostic_artifact && !causal_promoted_artifact
              ? "diagnostic" : "promoted")
      << " ONNX pair validated and dry-ran for shadow only.\n";
  if (arguments.validate_only) {
    std::cout << "[PASS] Validation-only completed; robot APIs unopened.\n";
    return 0;
  }

  EvidenceLog evidence(arguments.evidence);
  evidence.Write({
      {"schema_version", live::kEvidenceSchemaVersion},
      {"kind", live::kEvidenceKind},
      {"event", "session_start"},
      {"started_monotonic_ns", NowNs()},
      {"reference_profile", live::kCausalReferenceProfile},
      {"reference_contract_sha256",
       live::kCausalReferenceContractSha256},
      {"artifact_class",
       diagnostic_artifact && !causal_promoted_artifact
           ? "diagnostic_shadow_only" : "promoted_shadow"},
      {"decoder_output_semantics",
       std::string(live::kAppliedSafeNativeActionSemantics)},
      {"external_safe_target_transform_applied", false},
      {"encoder_sha256", files.encoder_sha},
      {"decoder_sha256", files.decoder_sha},
      {"metadata_sha256", files.metadata_sha},
      {"promotion_sha256",
       files.promotion_sha.has_value() ? json(*files.promotion_sha) :
                                        json(nullptr)},
      {"network", arguments.network},
      {"pico_endpoint", arguments.pico_endpoint},
      {"requested_action_frames", arguments.frames},
      {"robot_mutation_authorized", false},
  });
  std::cout << "[EVIDENCE] " << evidence.path() << '\n';

  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, arguments.network);
    ReadOnlyStateMonitor monitor;
    monitor.Start();
    monitor.WaitUntilReady(
        std::chrono::steady_clock::now() + std::chrono::seconds(5));
    evidence.Write({
        {"schema_version", live::kEvidenceSchemaVersion},
        {"kind", live::kEvidenceKind},
        {"event", "lowstate_gate_open"},
        {"mode_machine", true23::kRequiredModeMachine},
        {"crc_rejects", monitor.crc_rejects()},
        {"history_warmup_span_ns", kHistoryWarmupSpanNs},
    });

    ZmqSocket socket(arguments.pico_endpoint);
    live::ProprioHistory proprio_history;
    std::optional<std::array<float, 23>> previous_action;
    std::optional<std::uint64_t> previous_frame;
    std::optional<std::int64_t> previous_control_ns;
    int real_history_samples = 0;
    int action_frames = 0;
    double max_abs = 0.0;
    double minimum_margin = std::numeric_limits<double>::infinity();
    double maximum_slew_ratio = 0.0;
    const auto session_deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(arguments.timeout_seconds);

    while (action_frames < arguments.frames) {
      if (g_stop_requested != 0) {
        throw std::runtime_error("signal stopped shadow before requested frames");
      }
      if (std::chrono::steady_clock::now() >= session_deadline) {
        throw std::runtime_error("PICO live-shadow session timed out");
      }
      const auto message = socket.Receive();
      if (!message.has_value()) {
        (void)monitor.LatestAgeNs(NowNs());
        continue;
      }
      const auto received_ns = NowNs();
      const auto reference = ParseCausalPacket(*message);
      const auto packet_age_ns =
          received_ns - reference.control_monotonic_ns;
      if (packet_age_ns < -kFutureClockToleranceNs ||
          packet_age_ns > kPicoFreshnessNs) {
        throw std::runtime_error("PICO q10 freshness contract failed");
      }
      if (previous_frame.has_value() &&
          reference.control_source_frame_index != *previous_frame + 1) {
        throw std::runtime_error(
            "PICO control frame repeated, regressed, or skipped");
      }
      if (previous_control_ns.has_value() &&
          reference.control_monotonic_ns !=
              *previous_control_ns + 20'000'000) {
        throw std::runtime_error(
            "PICO control timestamp is not exact consecutive 20 ms");
      }
      previous_frame = reference.control_source_frame_index;
      previous_control_ns = reference.control_monotonic_ns;

      const auto joined = monitor.WaitForJoin(
          reference,
          std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
      live::ProprioSource source{
          .hardware_q = joined.control_proprio_q10.hardware_q,
          .hardware_dq = joined.control_proprio_q10.hardware_dq,
          .imu_gyroscope = joined.control_proprio_q10.gyroscope,
          .imu_quaternion_wxyz =
              joined.control_proprio_q10.quaternion_wxyz,
          .previous_action_native =
              previous_action.value_or(std::array<float, 23>{}),
      };
      proprio_history.Push(live::BuildProprioFrame(source));
      ++real_history_samples;
      if (real_history_samples < true23::kHistoryLength) {
        evidence.Write({
            {"schema_version", live::kEvidenceSchemaVersion},
            {"kind", live::kEvidenceKind},
            {"event", "causal_warmup_frame"},
            {"control_source_frame_index",
             reference.control_source_frame_index},
            {"control_monotonic_ns", reference.control_monotonic_ns},
            {"history_samples", real_history_samples},
            {"packet_age_ns", packet_age_ns},
            {"sdk_derivatives_consumed", false},
        });
        continue;
      }

      const auto inference_start_ns = NowNs();
      const auto encoder_input =
          live::BuildCausalEncoderInput(joined.encoder_terms);
      const auto live_token =
          encoder.Run<true23::kEncoderInputDim,
                      true23::kEncoderOutputDim>(encoder_input);
      const auto decoder_input = live::BuildDecoderInput(
          live_token, proprio_history.Flatten());
      const auto action =
          decoder.Run<true23::kDecoderInputDim,
                      true23::kDecoderOutputDim>(decoder_input);
      const auto produced_ns = NowNs();
      const auto inference_ns = produced_ns - inference_start_ns;
      const auto end_to_end_age_ns =
          produced_ns - reference.control_monotonic_ns;
      const auto assessment = live::AssessOutput(
          action, previous_action, live::kControlPeriodSeconds);
      const auto lowstate_age_ns = monitor.LatestAgeNs(produced_ns);
      const bool accepted =
          assessment.finite && assessment.normalized_max_abs <= 20.0 &&
          assessment.target_limit_violations == 0 &&
          assessment.target_slew_violations == 0 &&
          inference_ns <= kInferenceDeadlineNs &&
          end_to_end_age_ns >= -kFutureClockToleranceNs &&
          end_to_end_age_ns <= kPicoFreshnessNs;
      evidence.Write({
          {"schema_version", live::kEvidenceSchemaVersion},
          {"kind", live::kEvidenceKind},
          {"event", "action_frame"},
          {"action_frame_index", action_frames},
          {"control_source_frame_index",
           reference.control_source_frame_index},
          {"pico_anchor_source_frame_index",
           reference.pico_anchor_source_frame_index},
          {"pico_anchor_monotonic_ns",
           reference.pico_anchor_monotonic_ns},
          {"control_monotonic_ns", reference.control_monotonic_ns},
          {"received_monotonic_ns", received_ns},
          {"produced_monotonic_ns", produced_ns},
          {"packet_age_ns", packet_age_ns},
          {"end_to_end_age_ns", end_to_end_age_ns},
          {"lowstate_age_ns", lowstate_age_ns},
          {"inference_ns", inference_ns},
          {"native_action", action},
          {"decoder_output_semantics",
           std::string(live::kAppliedSafeNativeActionSemantics)},
          {"external_safe_target_transform_applied", false},
          {"normalized_max_abs", assessment.normalized_max_abs},
          {"target_position_min_margin_rad",
           assessment.target_position_min_margin_rad},
          {"target_limit_violations",
           assessment.target_limit_violations},
          {"slew_checked", assessment.slew_checked},
          {"target_slew_ratio_max",
           assessment.target_slew_ratio_max},
          {"target_slew_violations",
           assessment.target_slew_violations},
          {"sdk_derivatives_consumed", false},
          {"accepted", accepted},
      });
      if (!accepted) {
        throw std::runtime_error(
            "action frame failed numeric/limit/freshness/deadline gate");
      }
      previous_action = action;
      ++action_frames;
      max_abs = std::max(max_abs, assessment.normalized_max_abs);
      minimum_margin = std::min(
          minimum_margin, assessment.target_position_min_margin_rad);
      maximum_slew_ratio = std::max(
          maximum_slew_ratio, assessment.target_slew_ratio_max);
      if (action_frames == 1 || action_frames % 25 == 0 ||
          action_frames == arguments.frames) {
        std::cout << "[SHADOW] accepted action frame " << action_frames
                  << '/' << arguments.frames << '\n';
      }
    }

    VerifyUnchanged(files);
    evidence.Write({
        {"schema_version", live::kEvidenceSchemaVersion},
        {"kind", live::kEvidenceKind},
        {"event", "session_complete"},
        {"passed", true},
        {"action_frames", action_frames},
        {"causal_warmup_frames", true23::kHistoryLength - 1},
        {"maximum_normalized_abs", max_abs},
        {"minimum_target_position_margin_rad", minimum_margin},
        {"maximum_target_slew_ratio", maximum_slew_ratio},
        {"crc_rejects", monitor.crc_rejects()},
        {"robot_mutation_authorized", false},
    });
    std::cout
        << "[PASS] Causal live shadow completed. Actions stayed within "
           "finite, position, slew, and freshness gates.\n";
    return 0;
  } catch (const std::exception& error) {
    evidence.Write({
        {"schema_version", live::kEvidenceSchemaVersion},
        {"kind", live::kEvidenceKind},
        {"event", "session_failed"},
        {"passed", false},
        {"failure", error.what()},
        {"failed_monotonic_ns", NowNs()},
        {"robot_mutation_authorized", false},
    });
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto arguments = ParseArguments(argc, argv);
    if (arguments.help) {
      std::cout << Usage(argc > 0 ? argv[0] : "g1_true23_live_shadow")
                << '\n';
      return 0;
    }
    if (arguments.describe_contract) {
      std::cout << DescribeContract().dump(2) << '\n';
      return 0;
    }
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    return Run(arguments);
  } catch (const std::exception& error) {
    std::cerr << "[BLOCKED] true23 causal live shadow: "
              << error.what() << '\n';
    if (argc > 0) {
      std::cerr << Usage(argv[0]) << '\n';
    }
    return 1;
  }
}
