// Native G1 rev-1.0 23-DoF artifact and embodiment probe.
//
// This executable is deliberately shadow-only. It loads and dry-runs the
// validated teleop-encoder/decoder pair, then observes LowState. It contains
// no command message, command channel, motion-switcher client, or writer.

#include "true23_shadow_gate.hpp"

#include <onnxruntime_cxx_api.h>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

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
using LowState = unitree_hg::msg::dds_::LowState_;
using nlohmann::json;

inline constexpr std::string_view kLowStateTopic = "rt/lowstate";
volatile std::sig_atomic_t g_stop_requested = 0;

void HandleSignal(int) {
  g_stop_requested = 1;
}

std::uint32_t LowStateCrc32(
    const void* data_bytes,
    std::uint32_t word_count) {
  const auto* bytes =
      static_cast<const std::uint8_t*>(data_bytes);
  std::uint32_t crc = 0xffffffffU;
  constexpr std::uint32_t polynomial = 0x04c11db7U;
  for (std::uint32_t index = 0; index < word_count; ++index) {
    std::uint32_t data = 0;
    std::memcpy(&data, bytes + index * sizeof(data), sizeof(data));
    std::uint32_t bit = 1U << 31U;
    for (std::uint32_t count = 0; count < 32; ++count) {
      crc = (crc & 0x80000000U) != 0
                ? (crc << 1U) ^ polynomial
                : crc << 1U;
      if ((data & bit) != 0) {
        crc ^= polynomial;
      }
      bit >>= 1U;
    }
  }
  return crc;
}

struct Arguments {
  std::string mode;
  std::string network;
  fs::path encoder_path;
  fs::path decoder_path;
  fs::path metadata_path;
  fs::path promotion_path;
  bool once = false;
  bool help = false;
};

std::string Usage(std::string_view executable) {
  return
      "Usage: " + std::string(executable) +
      " --mode shadow --network <interface>"
      " --encoder <X_encoder.onnx>"
      " --decoder <X_decoder.onnx>"
      " --metadata <X.metadata.json>"
      " [--promotion <X.promotion.json>] [--once]\n"
      "\n"
      "Read-only true23 probe. Active control is unsupported. This checks "
      "legacy promoted artifacts when --promotion is omitted. With "
      "--promotion, it validates a MuJoCo candidate and promotion sidecar "
      "for shadow use only; active motor control remains unauthorized.";
}

Arguments ParseArguments(int argc, char** argv) {
  Arguments result;
  const auto require_value =
      [&](int& index, std::string_view option) -> std::string {
    if (index + 1 >= argc) {
      throw std::runtime_error(
          std::string(option) + " requires a value");
    }
    ++index;
    return argv[index];
  };

  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help" || option == "-h") {
      result.help = true;
    } else if (option == "--mode") {
      result.mode = require_value(index, option);
    } else if (option == "--network") {
      result.network = require_value(index, option);
    } else if (option == "--encoder") {
      result.encoder_path = require_value(index, option);
    } else if (option == "--decoder") {
      result.decoder_path = require_value(index, option);
    } else if (option == "--metadata") {
      result.metadata_path = require_value(index, option);
    } else if (option == "--promotion") {
      result.promotion_path = require_value(index, option);
    } else if (option == "--once") {
      result.once = true;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (result.help) {
    return result;
  }
  if (result.mode.empty()) {
    throw std::runtime_error("--mode shadow is required");
  }
  if (result.mode != "shadow") {
    throw std::runtime_error(
        "native true23 control is unsupported; --mode must be shadow");
  }
  if (result.network.empty()) {
    throw std::runtime_error("--network is required");
  }
  if (result.encoder_path.empty() || result.decoder_path.empty() ||
      result.metadata_path.empty()) {
    throw std::runtime_error(
        "--encoder, --decoder, and --metadata are all required");
  }
  return result;
}

fs::path ResolveRegularFile(const fs::path& path, std::string_view role) {
  std::error_code error;
  const auto canonical = fs::canonical(path, error);
  if (error) {
    throw std::runtime_error(
        std::string(role) + " path cannot be resolved: " +
        path.string() + ": " + error.message());
  }
  if (!fs::is_regular_file(canonical, error) || error) {
    throw std::runtime_error(
        std::string(role) + " path is not a regular file: " +
        canonical.string());
  }
  return canonical;
}

struct ResolvedFiles {
  fs::path encoder_argument;
  fs::path decoder_argument;
  fs::path metadata_argument;
  std::optional<fs::path> promotion_argument;
  fs::path encoder_identity;
  fs::path decoder_identity;
  fs::path metadata_identity;
  std::optional<fs::path> promotion_identity;
};

ResolvedFiles ResolveDistinctFiles(const Arguments& arguments) {
  const auto absolute_argument =
      [](const fs::path& value) {
        return fs::absolute(value).lexically_normal();
      };
  ResolvedFiles result{
      .encoder_argument = absolute_argument(arguments.encoder_path),
      .decoder_argument = absolute_argument(arguments.decoder_path),
      .metadata_argument = absolute_argument(arguments.metadata_path),
      .promotion_argument = std::nullopt,
      .encoder_identity =
          ResolveRegularFile(arguments.encoder_path, "encoder"),
      .decoder_identity =
          ResolveRegularFile(arguments.decoder_path, "decoder"),
      .metadata_identity =
          ResolveRegularFile(arguments.metadata_path, "metadata"),
      .promotion_identity = std::nullopt,
  };
  if (!arguments.promotion_path.empty()) {
    result.promotion_argument =
        absolute_argument(arguments.promotion_path);
    result.promotion_identity =
        ResolveRegularFile(arguments.promotion_path, "promotion");
  }
  if (result.encoder_identity == result.decoder_identity ||
      result.encoder_identity == result.metadata_identity ||
      result.decoder_identity == result.metadata_identity ||
      (result.promotion_identity.has_value() &&
       (result.promotion_identity.value() == result.encoder_identity ||
        result.promotion_identity.value() == result.decoder_identity ||
        result.promotion_identity.value() == result.metadata_identity))) {
    throw std::runtime_error(
        "encoder, decoder, metadata, and optional promotion paths must "
        "resolve to distinct files");
  }
  return result;
}

json ParseStrictJson(std::istream& stream, std::string_view role) {
  std::vector<std::unordered_set<std::string>> object_keys;
  const json::parser_callback_t callback =
      [&](int, json::parse_event_t event, json& parsed) {
        if (event == json::parse_event_t::object_start) {
          object_keys.emplace_back();
        } else if (event == json::parse_event_t::key) {
          if (object_keys.empty()) {
            throw std::runtime_error(
                std::string(role) + " JSON parser lost object scope");
          }
          const auto key = parsed.get<std::string>();
          if (!object_keys.back().insert(key).second) {
            throw std::runtime_error(
                std::string(role) + " JSON contains duplicate key: " +
                key);
          }
        } else if (event == json::parse_event_t::object_end) {
          if (object_keys.empty()) {
            throw std::runtime_error(
                std::string(role) + " JSON parser underflow");
          }
          object_keys.pop_back();
        }
        return true;
      };
  auto value = json::parse(stream, callback, true, false);
  if (!value.is_object()) {
    throw std::runtime_error(
        std::string(role) + " JSON root must be an object");
  }
  return value;
}

json LoadSidecar(const fs::path& path, std::string_view role) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(
        "cannot open " + std::string(role) + ": " + path.string());
  }
  return ParseStrictJson(stream, role);
}

Ort::SessionOptions MakeSessionOptions() {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  return options;
}

class ReadOnlyOnnxModel {
 public:
  ReadOnlyOnnxModel(Ort::Env& environment, fs::path path)
      : path_(std::move(path)),
        options_(MakeSessionOptions()),
        session_(environment, path_.c_str(), options_) {
    Inspect();
  }

  [[nodiscard]] const true23::ModelSignature& signature() const {
    return signature_;
  }

  [[nodiscard]] const json& embedded_metadata() const {
    return embedded_metadata_;
  }

  std::vector<float> Run(
      const std::vector<float>& input,
      std::size_t expected_output_size) {
    if (signature_.input_shape.size() != 2 ||
        signature_.output_shape.size() != 2 ||
        input.size() !=
            static_cast<std::size_t>(signature_.input_shape[1])) {
      throw std::runtime_error(
          "internal dry-run input does not match inspected ONNX shape");
    }
    auto memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    auto input_shape = signature_.input_shape;
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(input.data()), input.size(),
        input_shape.data(), input_shape.size());
    const char* input_names[] = {signature_.input_name.c_str()};
    const char* output_names[] = {signature_.output_name.c_str()};
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
        output_names, 1);
    if (outputs.size() != 1 || !outputs.front().IsTensor()) {
      throw std::runtime_error(
          "ONNX dry-run did not return exactly one tensor");
    }
    const auto output_info =
        outputs.front().GetTensorTypeAndShapeInfo();
    if (output_info.GetElementCount() != expected_output_size ||
        output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error(
          "ONNX dry-run output type or element count changed");
    }
    const auto* data = outputs.front().GetTensorData<float>();
    std::vector<float> result(data, data + expected_output_size);
    if (!std::all_of(
            result.begin(), result.end(),
            [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error(
          "ONNX dry-run produced non-finite output");
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
      if (name) {
        signature_.input_name = name.get();
      }
      const auto type_info = session_.GetInputTypeInfo(0);
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      signature_.input_shape = info.GetShape();
      signature_.input_float32 =
          info.GetElementType() ==
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    if (signature_.output_count == 1) {
      auto name = session_.GetOutputNameAllocated(0, allocator);
      if (name) {
        signature_.output_name = name.get();
      }
      const auto type_info = session_.GetOutputTypeInfo(0);
      const auto info = type_info.GetTensorTypeAndShapeInfo();
      signature_.output_shape = info.GetShape();
      signature_.output_float32 =
          info.GetElementType() ==
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    signature_.default_opsets =
        true23::InspectDefaultOnnxOpsetsFile(path_.string());

    auto metadata = session_.GetModelMetadata();
    auto keys =
        metadata.GetCustomMetadataMapKeysAllocated(allocator);
    if (keys.size() != 1 || !keys.front() ||
        std::string_view(keys.front().get()) !=
            true23::kOnnxMetadataKey) {
      throw std::runtime_error(
          "ONNX metadata must contain exactly g1_23dof_artifact");
    }
    auto value = metadata.LookupCustomMetadataMapAllocated(
        std::string(true23::kOnnxMetadataKey).c_str(), allocator);
    if (!value) {
      throw std::runtime_error(
          "ONNX g1_23dof_artifact metadata is missing");
    }
    std::istringstream stream(value.get());
    embedded_metadata_ =
        ParseStrictJson(stream, "embedded ONNX metadata");
  }

  fs::path path_;
  Ort::SessionOptions options_;
  Ort::Session session_;
  true23::ModelSignature signature_;
  json embedded_metadata_;
};

void DryRunPair(
    ReadOnlyOnnxModel& encoder,
    ReadOnlyOnnxModel& decoder) {
  const std::vector<float> encoder_input(
      true23::kEncoderInputDim, 0.0F);
  const auto token =
      encoder.Run(encoder_input, true23::kEncoderOutputDim);
  std::vector<float> decoder_input(
      true23::kDecoderInputDim, 0.0F);
  std::copy(token.begin(), token.end(), decoder_input.begin());
  const auto native_action =
      decoder.Run(decoder_input, true23::kDecoderOutputDim);

  std::array<float, 23> native{};
  std::copy(native_action.begin(), native_action.end(), native.begin());
  const auto hardware = true23::NativeToHardwareCompact(native);
  const auto max_abs =
      [](const auto& values) {
        float result = 0.0F;
        for (const auto value : values) {
          result = std::max(result, std::abs(value));
        }
        return result;
      };
  std::cout
      << "[PASS] Paired CPU dry-run: encoder [1,267]->[1,64], "
         "decoder [1,994]->[1,23], finite native/hardware outputs "
      << "(max_abs=" << max_abs(native)
      << ", mapped_max_abs=" << max_abs(hardware) << ").\n";
}

class ReadOnlyLowStateMonitor {
 public:
  struct Snapshot {
    LowState value;
    std::uint64_t generation = 0;
    std::chrono::steady_clock::time_point received_at;
  };

  ~ReadOnlyLowStateMonitor() {
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

  std::optional<Snapshot> WaitAfter(
      std::uint64_t generation,
      std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    if (!condition_.wait_until(
            lock, deadline,
            [&] { return generation_ > generation; })) {
      return std::nullopt;
    }
    return Snapshot{
        .value = *latest_,
        .generation = generation_,
        .received_at = received_at_,
    };
  }

  [[nodiscard]] std::uint64_t crc_errors() const {
    return crc_errors_.load(std::memory_order_relaxed);
  }

 private:
  void OnMessage(const void* message) {
    if (message == nullptr) {
      return;
    }
    LowState state = *static_cast<const LowState*>(message);
    const auto received_crc = state.crc();
    const auto calculated_crc = LowStateCrc32(
        &state, (sizeof(LowState) >> 2U) - 1U);
    if (received_crc != calculated_crc) {
      crc_errors_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    {
      std::lock_guard lock(mutex_);
      latest_ = std::move(state);
      ++generation_;
      received_at_ = std::chrono::steady_clock::now();
    }
    condition_.notify_one();
  }

  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>>
      subscriber_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<LowState> latest_;
  std::uint64_t generation_ = 0;
  std::chrono::steady_clock::time_point received_at_;
  std::atomic<std::uint64_t> crc_errors_{0};
};

void ValidateNativeTelemetry(const LowState& state) {
  std::array<float, 29> q_slots{};
  std::array<float, 29> dq_slots{};
  for (const int motor_id : true23::kHardwareJointIds) {
    const auto& motor =
        state.motor_state()[static_cast<std::size_t>(motor_id)];
    if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq())) {
      throw std::runtime_error(
          "non-finite telemetry in required true23 motor slot " +
          std::to_string(motor_id));
    }
    q_slots[static_cast<std::size_t>(motor_id)] = motor.q();
    dq_slots[static_cast<std::size_t>(motor_id)] = motor.dq();
  }
  const auto native_q = true23::HardwareSlotsToNative(q_slots);
  const auto native_dq = true23::HardwareSlotsToNative(dq_slots);
  if (!std::all_of(
          native_q.begin(), native_q.end(),
          [](float value) { return std::isfinite(value); }) ||
      !std::all_of(
          native_dq.begin(), native_dq.end(),
          [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error(
        "native true23 telemetry permutation produced non-finite data");
  }
}

struct OpenShadowGate {
  true23::ModeMachineShadowGate gate;
  std::uint64_t generation = 0;
};

OpenShadowGate WaitForModeFour(ReadOnlyLowStateMonitor& monitor) {
  true23::ModeMachineShadowGate gate;
  std::uint64_t generation = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto snapshot = monitor.WaitAfter(generation, deadline);
    if (!snapshot.has_value()) {
      break;
    }
    generation = snapshot->generation;
    const auto observation = gate.Observe(
        snapshot->value.tick(), snapshot->value.mode_machine());
    if (observation == true23::ModeObservation::TickRegression ||
        observation == true23::ModeObservation::LatchedFailure) {
      throw std::runtime_error(
          "LowState tick regressed while establishing mode_machine==4");
    }
    if (observation == true23::ModeObservation::Ready) {
      ValidateNativeTelemetry(snapshot->value);
      return {.gate = gate, .generation = generation};
    }
  }
  throw std::runtime_error(
      "no five advancing CRC-valid mode_machine==4 LowState samples "
      "within 5 seconds (CRC rejects=" +
      std::to_string(monitor.crc_errors()) + ")");
}

void MonitorModeFour(
    ReadOnlyLowStateMonitor& monitor,
    OpenShadowGate state) {
  while (g_stop_requested == 0) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    const auto snapshot =
        monitor.WaitAfter(state.generation, deadline);
    if (!snapshot.has_value()) {
      throw std::runtime_error(
          "LowState freshness timeout in true23 shadow monitor");
    }
    state.generation = snapshot->generation;
    const auto observation = state.gate.Observe(
        snapshot->value.tick(), snapshot->value.mode_machine());
    if (observation == true23::ModeObservation::TickRegression) {
      throw std::runtime_error(
          "LowState tick regression latched true23 shadow failure");
    }
    if (observation == true23::ModeObservation::LatchedFailure ||
        observation == true23::ModeObservation::WrongMode) {
      throw std::runtime_error(
          "mode_machine changed away from 4; shadow gate latched");
    }
    if (observation != true23::ModeObservation::DuplicateTick) {
      ValidateNativeTelemetry(snapshot->value);
    }
  }
}

int Run(const Arguments& arguments) {
  const auto files = ResolveDistinctFiles(arguments);
  const auto encoder_sha_before =
      true23::Sha256File(files.encoder_identity.string());
  const auto decoder_sha_before =
      true23::Sha256File(files.decoder_identity.string());
  const auto metadata_sha_before =
      true23::Sha256File(files.metadata_identity.string());
  const auto promotion_sha_before =
      files.promotion_identity.has_value()
          ? std::optional<std::string>(true23::Sha256File(
                files.promotion_identity->string()))
          : std::nullopt;

  const auto sidecar =
      LoadSidecar(files.metadata_identity, "metadata sidecar");
  const auto promotion =
      files.promotion_identity.has_value()
          ? std::optional<json>(LoadSidecar(
                files.promotion_identity.value(),
                "MuJoCo promotion sidecar"))
          : std::nullopt;

  Ort::Env environment(
      ORT_LOGGING_LEVEL_WARNING, "g1_true23_shadow_gate");
  ReadOnlyOnnxModel encoder(environment, files.encoder_identity);
  ReadOnlyOnnxModel decoder(environment, files.decoder_identity);

  const auto encoder_sha_after =
      true23::Sha256File(files.encoder_identity.string());
  const auto decoder_sha_after =
      true23::Sha256File(files.decoder_identity.string());
  const auto metadata_sha_after =
      true23::Sha256File(files.metadata_identity.string());
  const auto promotion_sha_after =
      files.promotion_identity.has_value()
          ? std::optional<std::string>(true23::Sha256File(
                files.promotion_identity->string()))
          : std::nullopt;
  if (encoder_sha_before != encoder_sha_after ||
      decoder_sha_before != decoder_sha_after ||
      metadata_sha_before != metadata_sha_after ||
      promotion_sha_before != promotion_sha_after) {
    throw std::runtime_error(
        "artifact or sidecar changed while being loaded");
  }

  const true23::PairBinding binding{
      .encoder_filename =
          files.encoder_argument.filename().string(),
      .decoder_filename =
          files.decoder_argument.filename().string(),
      .metadata_filename =
          files.metadata_argument.filename().string(),
      .encoder_onnx_sha256 = encoder_sha_after,
      .decoder_onnx_sha256 = decoder_sha_after,
      .encoder_path = files.encoder_identity.string(),
      .decoder_path = files.decoder_identity.string(),
      .metadata_path = files.metadata_identity.string(),
      .metadata_sha256 = metadata_sha_after,
  };
  const auto validation =
      promotion.has_value()
          ? true23::ValidateMujocoShadowPromotion(
                promotion.value(), sidecar,
                encoder.embedded_metadata(),
                decoder.embedded_metadata(), encoder.signature(),
                decoder.signature(), binding,
                true23::RequestedMode::Shadow)
          : true23::ValidateShadowArtifact(
                sidecar, encoder.embedded_metadata(),
                decoder.embedded_metadata(), encoder.signature(),
                decoder.signature(), binding,
                true23::RequestedMode::Shadow);
  validation.Require();
  if (validation.authorization.lowcmd_publisher_allowed ||
      validation.authorization.motion_mode_release_allowed ||
      validation.authorization.command_writer_allowed) {
    throw std::logic_error(
        "shadow validator unexpectedly granted mutation authority");
  }

  if (promotion.has_value()) {
    std::cout
        << "[PASS] True23 MuJoCo promotion, candidate metadata, embedded "
           "metadata, signatures, identities, and unchanged ONNX bytes "
           "authorize shadow validation only.\n"
        << "[REQUIRED] Run the Python promotion verifier against the raw "
           "checkpoint, report, and traces before native shadow. Rated "
           "support remains mandatory for first actuation; this "
           "executable cannot authorize active motor control.\n";
  } else {
    std::cout
        << "[PASS] True23 encoder/decoder artifact metadata, signatures, "
           "bindings, and hashes are internally self-consistent.\n"
        << "[REQUIRED] This executable did not open the training checkpoint "
           "or raw simulation report; "
           "external Python readiness verification must validate both "
           "before any "
           "promotion.\n";
  }
  DryRunPair(encoder, decoder);
  if (true23::Sha256File(files.encoder_identity.string()) !=
          encoder_sha_after ||
      true23::Sha256File(files.decoder_identity.string()) !=
          decoder_sha_after ||
      true23::Sha256File(files.metadata_identity.string()) !=
          metadata_sha_after ||
      (files.promotion_identity.has_value() &&
       true23::Sha256File(files.promotion_identity->string()) !=
           promotion_sha_after.value())) {
    throw std::runtime_error(
        "artifact or sidecar changed during paired dry-run");
  }

  // DDS starts only after the complete immutable artifact pair passes.
  unitree::robot::ChannelFactory::Instance()->Init(
      0, arguments.network);
  ReadOnlyLowStateMonitor monitor;
  monitor.Start();
  auto open_gate = WaitForModeFour(monitor);
  std::cout
      << "[PASS] Five advancing CRC-valid LowState samples confirm "
         "mode_machine==4 and finite exact-slot telemetry.\n"
      << "[SHADOW ONLY] No command publisher, command writer, or "
         "control-mode transition exists in this executable.\n";

  if (!arguments.once) {
    MonitorModeFour(monitor, std::move(open_gate));
  }
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
    std::cerr << "[BLOCKED] true23 shadow gate: " << error.what()
              << '\n';
    if (argc > 0) {
      std::cerr << Usage(argv[0]) << '\n';
    }
    return 1;
  }
}
