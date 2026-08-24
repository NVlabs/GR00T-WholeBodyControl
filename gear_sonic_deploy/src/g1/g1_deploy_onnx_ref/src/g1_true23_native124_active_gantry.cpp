// Pinned native124 public-policy, stage-one gantry runtime.
// Reuse the reviewed DDS/operator/writer implementation without changing the
// read-only live-shadow translation unit.
#define main g1_true23_causal_active_gantry_embedded_main
#include "g1_true23_active_gantry.cpp"
#undef main

namespace native124_active_runtime {

constexpr int kFaultDampingCyclesNative124 = 250;

struct NativeArguments {
  std::string network;
  fs::path policy;
  std::string pico_endpoint;
  fs::path evidence;
  std::string authorization_id;
  std::string gantry_authorization;
  int post_arm_duration_seconds = 0;
  bool validate_only = false;
  bool execute_stage_one = false;
  bool help = false;
};

std::string NativeUsage(std::string_view executable) {
  return "Usage: " + std::string(executable) +
      " --policy <fightAndSports1_subject1.onnx> [--validate-only |"
      " --network <interface> --pico-endpoint <tcp://host:port>"
      " --execute-stage-one --evidence <new.jsonl>"
      " --authorization-id <session-id>"
      " --post-arm-duration-seconds <20..30> --gantry-authorize " +
      std::string(active::kGantryAuthorizationPhrase) +
      "]\nStage-one gantry only: L2 deadman + A edge arms; B/R2/L2 release STOP."
      " LowState<=40 ms, mode_machine==4, exact True23 motor slots, 10%"
      " action envelope, 0.25 rad/s target slew. Not free-standing.";
}

NativeArguments ParseNativeArguments(int argc, char** argv) {
  NativeArguments result;
  std::unordered_set<std::string> seen;
  const auto value = [&](int& index, std::string_view option) {
    if (++index >= argc) {
      throw std::runtime_error(std::string(option) + " requires a value");
    }
    return std::string(argv[index]);
  };
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (!seen.insert(option).second) {
      throw std::runtime_error("duplicate option rejected: " + option);
    }
    if (option == "--help" || option == "-h") result.help = true;
    else if (option == "--network") result.network = value(index, option);
    else if (option == "--policy") result.policy = value(index, option);
    else if (option == "--pico-endpoint") result.pico_endpoint = value(index, option);
    else if (option == "--evidence") result.evidence = value(index, option);
    else if (option == "--authorization-id") result.authorization_id = value(index, option);
    else if (option == "--gantry-authorize") result.gantry_authorization = value(index, option);
    else if (option == "--post-arm-duration-seconds") {
      const auto text = value(index, option);
      std::size_t parsed = 0;
      result.post_arm_duration_seconds = std::stoi(text, &parsed);
      if (parsed != text.size()) throw std::runtime_error("duration must be integer");
    } else if (option == "--validate-only") result.validate_only = true;
    else if (option == "--execute-stage-one") result.execute_stage_one = true;
    else throw std::runtime_error("unknown option: " + option);
  }
  if (result.help) return result;
  if (result.policy.empty()) throw std::runtime_error("--policy is required");
  if (result.validate_only) {
    if (result.execute_stage_one || !result.network.empty() ||
        !result.pico_endpoint.empty() || !result.evidence.empty()) {
      throw std::runtime_error("--validate-only rejects robot/network arguments");
    }
    return result;
  }
  if (!result.execute_stage_one || result.network.empty() ||
      result.pico_endpoint.empty() || result.evidence.empty() ||
      !active::IsSafeAuthorizationId(result.authorization_id) ||
      result.gantry_authorization != active::kGantryAuthorizationPhrase) {
    throw std::runtime_error("exact stage-one network/evidence/authorization arguments required");
  }
  if (!result.pico_endpoint.starts_with("tcp://") &&
      !result.pico_endpoint.starts_with("ipc://")) {
    throw std::runtime_error("PICO endpoint must use tcp:// or ipc://");
  }
  if (result.post_arm_duration_seconds < active::kMinimumStageOnePostArmSeconds ||
      result.post_arm_duration_seconds > active::kMaximumStageOnePostArmSeconds) {
    throw std::runtime_error("post-arm duration must be within 20..30 seconds");
  }
  return result;
}

class Native124Model {
 public:
  Native124Model(Ort::Env& environment, const fs::path& path)
      : options_(SessionOptions()), session_(environment, path.c_str(), options_) {
    Ort::AllocatorWithDefaultOptions allocator;
    if (session_.GetInputCount() != 2 || session_.GetOutputCount() < 1) {
      throw std::runtime_error("native124 ONNX input/output count mismatch");
    }
    const auto require = [](const Ort::TypeInfo& type,
                            std::initializer_list<std::int64_t> shape,
                            std::string_view role) {
      const auto info = type.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          info.GetShape() != std::vector<std::int64_t>(shape)) {
        throw std::runtime_error("native124 " + std::string(role) + " ABI mismatch");
      }
    };
    auto obs = session_.GetInputNameAllocated(0, allocator);
    auto time = session_.GetInputNameAllocated(1, allocator);
    if (!obs || !time || std::string_view(obs.get()) != "obs" ||
        std::string_view(time.get()) != "time_step") {
      throw std::runtime_error("native124 ONNX input names mismatch");
    }
    require(session_.GetInputTypeInfo(0), {1, 124}, "obs");
    require(session_.GetInputTypeInfo(1), {1, 1}, "time_step");
    bool actions = false;
    for (std::size_t index = 0; index < session_.GetOutputCount(); ++index) {
      auto name = session_.GetOutputNameAllocated(index, allocator);
      if (name && std::string_view(name.get()) == "actions") {
        require(session_.GetOutputTypeInfo(index), {1, 23}, "actions");
        actions = true;
      }
    }
    if (!actions) throw std::runtime_error("native124 actions output missing");
  }

  std::array<float, 23> Run(
      const std::array<float, live::kNative124ObservationDim>& observation) {
    if (!live::IsFinite(observation)) throw std::runtime_error("non-finite observation");
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<std::int64_t, 2> obs_shape{1, 124};
    std::array<std::int64_t, 2> time_shape{1, 1};
    std::array<float, 1> time_step{};
    auto obs = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(observation.data()), observation.size(),
        obs_shape.data(), obs_shape.size());
    auto time = Ort::Value::CreateTensor<float>(
        memory, time_step.data(), time_step.size(), time_shape.data(), time_shape.size());
    std::array<Ort::Value, 2> inputs{std::move(obs), std::move(time)};
    constexpr std::array<const char*, 2> names{"obs", "time_step"};
    constexpr std::array<const char*, 1> outputs{"actions"};
    auto values = session_.Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(),
                               inputs.size(), outputs.data(), outputs.size());
    if (values.size() != 1 || !values.front().IsTensor() ||
        values.front().GetTensorTypeAndShapeInfo().GetElementCount() != 23) {
      throw std::runtime_error("native124 inference output mismatch");
    }
    std::array<float, 23> result{};
    std::copy_n(values.front().GetTensorData<float>(), 23, result.begin());
    if (!live::IsFinite(result)) throw std::runtime_error("native124 action non-finite");
    return result;
  }

 private:
  Ort::SessionOptions options_;
  Ort::Session session_;
};

int RunNative(const NativeArguments& arguments) {
  const auto policy_path = ResolveFile(arguments.policy, "native124 policy");
  const auto policy_sha = true23::Sha256File(policy_path.string());
  if (policy_sha != active::kSelectedNative124PolicySha256) {
    throw std::runtime_error("selected native124 policy SHA-256 mismatch");
  }
  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "g1_true23_native124_active");
  Native124Model policy(environment, policy_path);
  const std::array<float, live::kNative124ObservationDim> zero_observation{};
  const auto dry_action = policy.Run(zero_observation);
  if (!live::IsFinite(dry_action) || true23::Sha256File(policy_path.string()) != policy_sha) {
    throw std::runtime_error("native124 dry-run/file-identity gate failed");
  }
  active::Native124ActiveArtifactBinding binding{
      .policy_sha256 = policy_sha, .observation_dim = 124, .action_dim = 23,
      .mode_machine = true23::kRequiredModeMachine, .onnx_signature_valid = true,
      .dry_run_finite = true, .gantry_authorized = true,
      .free_standing_authorized = false,
      .external_target_envelope_required = true,
      .stage_one_action_fraction = active::kStageOneActionFraction,
      .maximum_target_rate_rad_per_second = active::kStageOneTargetRateRadPerSecond};
  const auto binding_errors = active::ValidateNative124ActiveArtifactBinding(binding);
  if (!binding_errors.empty()) throw std::runtime_error(binding_errors.front());
  if (arguments.validate_only) {
    std::cout << "[PASS] pinned native124 ABI/hash/dry-run; no robot APIs opened.\n";
    return 0;
  }

  active::GantrySafetyCore core(binding);
  ExecutionEvidenceLog evidence(
      arguments.evidence, arguments.authorization_id,
      {{"backend", "pinned_native124_stage1_gantry"},
       {"policy_sha256", policy_sha}, {"mode_machine", 4},
       {"motor_ids", true23::kHardwareJointIds},
       {"excluded_motor_ids", true23::kExcludedHardwareJointIds},
       {"lowstate_freshness_limit_ms", 40}, {"pico_freshness_limit_ms", 100},
       {"stage_one_action_fraction", active::kStageOneActionFraction},
       {"target_rate_rad_per_second", active::kStageOneTargetRateRadPerSecond},
       {"warm_start_frames", active::kNative124WarmStartFrames},
       {"reference_ramp_frames", active::kNative124ReferenceRampFrames},
       {"gantry_only", true}, {"free_standing_authorized", false}});

  unitree::robot::ChannelFactory::Instance()->Init(0, arguments.network);
  StateMonitor monitor(core);
  monitor.Start();
  if (!monitor.WaitForMutationGate(std::chrono::steady_clock::now() +
                                   std::chrono::seconds(kStateGateTimeoutSeconds))) {
    throw std::runtime_error("five advancing CRC-valid mode-4 states not obtained");
  }
  ReleaseMotionModeAfterGate();
  monitor.WithCore([](active::GantrySafetyCore& value) { value.CheckWatchdogs(NowNs()); });
  if (monitor.fault() != active::Fault::None) throw std::runtime_error("state fault before publisher");
  auto publisher = std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(
      std::string(kLowCmdTopic));
  publisher->InitChannel();

  std::atomic<bool> stop{false};
  std::atomic<bool> ready{false};
  std::atomic<std::int64_t> first_active_write_ns{0};
  std::string inference_error;
  std::string writer_error;
  std::uint64_t active_frames = 0;
  int damping_frames = 0;

  std::jthread inference([&](std::stop_token token) {
    try {
      ZmqSocket socket(arguments.pico_endpoint);
      std::array<float, 23> previous_action{};
      std::optional<std::uint64_t> previous_frame;
      std::optional<std::int64_t> previous_time;
      std::optional<active::Native124AcquisitionGate> acquisition;
      while (!token.stop_requested() && !stop.load()) {
        const auto message = socket.Receive();
        if (!message) continue;
        auto reference = ParseCausalPicoReferenceTerms(*message);
        const auto now = NowNs();
        const auto age = now - reference.control_monotonic_ns;
        if (age < -active::kFutureClockToleranceNs || age > active::kPolicyFreshnessNs ||
            (previous_frame && reference.control_source_frame_index != *previous_frame + 1U) ||
            (previous_time && reference.control_monotonic_ns !=
                                  *previous_time + active::kShadowControlPeriodNs)) {
          monitor.WithCore([](auto& value) { value.ObservePicoTermsFailure(); });
          break;
        }
        previous_frame = reference.control_source_frame_index;
        previous_time = reference.control_monotonic_ns;
        const auto joined = monitor.TryJoinCausalReference(reference);
        if (joined.status == CausalJoinStatus::AwaitingCoverage) continue;
        if (joined.status != CausalJoinStatus::Ready || !joined.joined) {
          monitor.WithCore([](auto& value) { value.ObservePicoTermsFailure(); });
          break;
        }
        if (!acquisition) {
          acquisition.emplace(true23::HardwareCompactToNative(
              joined.joined->control_proprio_q10.hardware_q));
        }
        const auto staged = acquisition->Advance(
            reference.q_ref23_native, reference.qd_ref23_native);
        reference.q_ref23_native = staged.position_native;
        reference.qd_ref23_native = staged.velocity_native;
        const auto observation = live::BuildNative124Observation(
            reference, *joined.joined, previous_action);
        const auto action = policy.Run(observation);
        const auto raw_targets = live::Native124RawActionToClampedTargets(action);
        if (raw_targets.raw_target_limit_clamps != 0) {
          monitor.WithCore([](auto& value) { value.ObserveInternalFailure(); });
          break;
        }
        if (staged.ready_for_arm) {
          monitor.WithCore([&](auto& value) {
            value.SubmitPolicy({.native_action = action, .produced_monotonic_ns = now}, now);
            ready.store(value.policy_ready_for_arm(now));
          });
        }
        previous_action = action;
      }
    } catch (const std::exception& error) {
      inference_error = error.what();
      monitor.WithCore([](auto& value) { value.ObservePicoTermsFailure(); });
    }
  });

  std::jthread writer([&](std::stop_token token) {
    try {
      auto deadline = NowNs();
      int fault_cycles = 0;
      while (!token.stop_requested() && !stop.load()) {
        std::this_thread::sleep_until(std::chrono::steady_clock::time_point(
            std::chrono::nanoseconds(deadline)));
        active::Fault fault = active::Fault::None;
        bool active_write = false;
        monitor.WithCore([&](auto& value) {
          if (g_stop_requested) value.Stop();
          const bool armed = value.armed();
          const auto command = value.BuildCommand(NowNs());
          fault = value.fault();
          active_write = armed && value.armed() && fault == active::Fault::None;
          publisher->Write(ToLowCmd(command));
        });
        deadline = active::NextNoCatchUpWriterDeadlineNs(NowNs());
        if (active_write) {
          ++active_frames;
          std::int64_t expected = 0;
          first_active_write_ns.compare_exchange_strong(expected, NowNs());
        }
        if (fault != active::Fault::None && ++fault_cycles >= kFaultDampingCyclesNative124) {
          damping_frames = fault_cycles;
          stop.store(true);
        }
      }
    } catch (const std::exception& error) {
      writer_error = error.what();
      monitor.WithCore([](auto& value) { value.ObserveInternalFailure(); });
      for (int index = 0; index < kFaultDampingCyclesNative124; ++index) {
        try { monitor.WithCore([&](auto& value) {
          publisher->Write(ToLowCmd(value.BuildDampingCommand()));
        }); } catch (...) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      damping_frames = kFaultDampingCyclesNative124;
      stop.store(true);
    }
  });

  std::cout << "[WAIT] 25 neutral warm frames + 100-frame qref ramp. Do not arm.\n";
  bool announced = false;
  std::int64_t stop_requested_ns = 0;
  while (!stop.load()) {
    if (ready.load() && !announced) {
      std::cout << "[READY] Gantry secure: hold L2, press A once. B/R2/L2 release STOP.\n";
      announced = true;
    }
    const auto now = NowNs();
    const auto first = first_active_write_ns.load();
    if (g_stop_requested ||
        (first > 0 && now - first >=
            static_cast<std::int64_t>(arguments.post_arm_duration_seconds) * 1'000'000'000LL)) {
      stop_requested_ns = now;
      monitor.WithCore([](auto& value) { value.Stop(); });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  inference.request_stop(); writer.request_stop();
  inference.join(); writer.join();
  const auto fault = monitor.fault();
  const bool passed = first_active_write_ns.load() > 0 &&
      active_frames >= static_cast<std::uint64_t>(active::kMinimumPromotedShadowActionFrames) &&
      damping_frames >= kFaultDampingCyclesNative124 &&
      fault == active::Fault::OperatorStop && inference_error.empty() && writer_error.empty();
  evidence.Finalize(passed ? "session_complete" : "session_failed",
      {{"passed", passed}, {"active_frames", active_frames},
       {"damping_frames", damping_frames},
       {"final_fault", std::string(active::FaultName(fault))},
       {"stop_requested_ns", stop_requested_ns},
       {"inference_error", inference_error}, {"writer_error", writer_error}});
  return passed ? 0 : 2;
}

}  // namespace native124_active_runtime

int main(int argc, char** argv) {
  try {
    const auto arguments = native124_active_runtime::ParseNativeArguments(argc, argv);
    if (arguments.help) {
      std::cout << native124_active_runtime::NativeUsage(argv[0]) << '\n';
      return 0;
    }
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    return native124_active_runtime::RunNative(arguments);
  } catch (const std::exception& error) {
    std::cerr << "[BLOCKED] native124 active gantry: " << error.what() << '\n';
    return 1;
  }
}
