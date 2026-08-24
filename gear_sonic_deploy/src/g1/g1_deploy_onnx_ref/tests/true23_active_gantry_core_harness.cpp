#include "true23_active_gantry_core.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace active = gear_sonic::true23::active;
namespace true23 = gear_sonic::true23;

class Runner {
 public:
  void Check(bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
  [[nodiscard]] int failures() const { return failures_; }

 private:
  int failures_ = 0;
};

std::string Hash(char character) {
  return std::string(64, character);
}

active::ActiveArtifactBinding ValidArtifact() {
  return {
      .base_promotion_valid = true,
      .stage = active::ArtifactStage::MujocoPromotion,
      .decoder_output_dim = 23,
      .mode_machine = 4,
      .action_clip_value = 20.0,
      .deployment_ready = true,
      .active_motor_control_authorized = true,
      .gantry_authorized = true,
      .free_standing_authorized = false,
      .decoder_output_semantics = "applied_safe_native_action",
      .previous_action_semantics = "applied_safe_native_action",
      .external_safe_target_transform_allowed = false,
      .safe_target_transform_sha256 =
          std::string(gear_sonic::true23::live::kSafeTargetTransformSha256),
      .source_promotion_sha256 = Hash('1'),
      .checkpoint_sha256 = Hash('5'),
      .lineage_sha256 = Hash('6'),
      .policy_state_sha256 = Hash('7'),
      .encoder_onnx_sha256 = Hash('2'),
      .decoder_onnx_sha256 = Hash('3'),
      .metadata_sha256 = Hash('4'),
      .full_campaign_aggregate_sha256 = Hash('8'),
      .full_campaign_shard_manifest_sha256 = Hash('9'),
      .live_shadow_evidence_sha256 = Hash('a'),
      .authorization_id = "reviewed-stage-one-fixture",
  };
}

active::Native124ActiveArtifactBinding ValidNative124Artifact() {
  return {
      .policy_sha256 = std::string(active::kSelectedNative124PolicySha256),
      .observation_dim = 124,
      .action_dim = 23,
      .mode_machine = 4,
      .onnx_signature_valid = true,
      .dry_run_finite = true,
      .gantry_authorized = true,
      .free_standing_authorized = false,
      .external_target_envelope_required = true,
      .stage_one_action_fraction = active::kStageOneActionFraction,
      .maximum_target_rate_rad_per_second =
          active::kStageOneTargetRateRadPerSecond,
  };
}

active::StateSample State(
    std::uint32_t tick, std::int64_t time_ns, std::uint8_t mode = 4) {
  active::StateSample sample;
  sample.tick = tick;
  sample.mode_machine = mode;
  sample.crc_valid = true;
  sample.received_monotonic_ns = time_ns;
  for (std::size_t compact = 0; compact < 23; ++compact) {
    sample.q[static_cast<std::size_t>(
        true23::kHardwareJointIds[compact])] =
        gear_sonic::true23::live::kHardwareDefaultQ[compact];
  }
  return sample;
}

void OpenGate(active::GantrySafetyCore& core,
              std::int64_t start_ns = 1'000'000'000) {
  for (std::uint32_t tick = 1; tick <= 5; ++tick) {
    const auto time = start_ns + static_cast<std::int64_t>(tick) * 2'000'000;
    core.ObserveState(State(tick, time), time);
  }
}

active::PolicySample ZeroPolicy(std::int64_t time_ns) {
  active::PolicySample sample;
  sample.produced_monotonic_ns = time_ns;
  return sample;
}

active::LiveShadowEvidenceBinding EvidenceBinding() {
  return {
      .encoder_sha256 = Hash('2'),
      .decoder_sha256 = Hash('3'),
      .metadata_sha256 = Hash('4'),
      .promotion_sha256 = Hash('1'),
      .network = "eth0",
      .pico_endpoint = "tcp://127.0.0.1:5557",
  };
}

std::vector<nlohmann::json> PassingEvidenceRecords(int action_frames = 100) {
  using nlohmann::json;
  std::vector<json> records;
  records.push_back({
      {"schema_version", 1},
      {"kind", "g1_true23_integrated_live_shadow_evidence"},
      {"event", "session_start"},
      {"started_monotonic_ns", 1'000'000'000'000LL},
      {"reference_profile", "true23_causal_step1_history_0p02s_v1"},
      {"reference_contract_sha256",
       "bd046467325fe7f7f585fd692f01223ed7a3b2742c51ed414072c98fe12806f7"},
      {"artifact_class", "promoted_shadow"},
      {"decoder_output_semantics", "applied_safe_native_action"},
      {"external_safe_target_transform_applied", false},
      {"encoder_sha256", Hash('2')},
      {"decoder_sha256", Hash('3')},
      {"metadata_sha256", Hash('4')},
      {"promotion_sha256", Hash('1')},
      {"network", "eth0"},
      {"pico_endpoint", "tcp://127.0.0.1:5557"},
      {"requested_action_frames", action_frames},
      {"robot_mutation_authorized", false},
  });
  records.push_back({
      {"schema_version", 1},
      {"kind", "g1_true23_integrated_live_shadow_evidence"},
      {"event", "lowstate_gate_open"},
      {"mode_machine", 4},
      {"crc_rejects", 0},
      {"history_warmup_span_ns", 40'000'000},
  });
  std::int64_t control_frame = 10;
  std::int64_t control_ns = 2'000'000'000'000LL;
  for (int index = 0; index < 9; ++index) {
    records.push_back({
        {"schema_version", 1},
        {"kind", "g1_true23_integrated_live_shadow_evidence"},
        {"event", "causal_warmup_frame"},
        {"control_source_frame_index", control_frame},
        {"control_monotonic_ns", control_ns},
        {"history_samples", index + 1},
        {"packet_age_ns", 1'000'000},
        {"sdk_derivatives_consumed", false},
    });
    ++control_frame;
    control_ns += 20'000'000;
  }
  const std::array<double, 23> zero_action{};
  for (int index = 0; index < action_frames; ++index) {
    const auto received_ns = control_ns + 1'000'000;
    const auto produced_ns = received_ns + 1'000'000;
    records.push_back({
        {"schema_version", 1},
        {"kind", "g1_true23_integrated_live_shadow_evidence"},
        {"event", "action_frame"},
        {"action_frame_index", index},
        {"control_source_frame_index", control_frame},
        {"pico_anchor_source_frame_index", control_frame - 1},
        {"pico_anchor_monotonic_ns", control_ns - 20'000'000},
        {"control_monotonic_ns", control_ns},
        {"received_monotonic_ns", received_ns},
        {"produced_monotonic_ns", produced_ns},
        {"packet_age_ns", 1'000'000},
        {"end_to_end_age_ns", 2'000'000},
        {"lowstate_age_ns", 1'000'000},
        {"inference_ns", 1'000'000},
        {"native_action", zero_action},
        {"decoder_output_semantics", "applied_safe_native_action"},
        {"external_safe_target_transform_applied", false},
        {"normalized_max_abs", 0.0},
        {"target_position_min_margin_rad", 0.1},
        {"target_limit_violations", 0},
        {"slew_checked", index != 0},
        {"target_slew_ratio_max", 0.0},
        {"target_slew_violations", 0},
        {"sdk_derivatives_consumed", false},
        {"accepted", true},
    });
    ++control_frame;
    control_ns += 20'000'000;
  }
  records.push_back({
      {"schema_version", 1},
      {"kind", "g1_true23_integrated_live_shadow_evidence"},
      {"event", "session_complete"},
      {"passed", true},
      {"action_frames", action_frames},
      {"causal_warmup_frames", 9},
      {"maximum_normalized_abs", 0.0},
      {"minimum_target_position_margin_rad", 0.1},
      {"maximum_target_slew_ratio", 0.0},
      {"crc_rejects", 0},
      {"robot_mutation_authorized", false},
  });
  return records;
}

std::string DumpEvidence(const std::vector<nlohmann::json>& records) {
  std::string bytes;
  for (const auto& record : records) {
    bytes += record.dump();
    bytes.push_back('\n');
  }
  return bytes;
}

bool EvidenceRejected(
    std::string_view bytes,
    const active::LiveShadowEvidenceBinding& binding = EvidenceBinding()) {
  try {
    (void)active::ValidateLiveShadowEvidenceJsonl(bytes, binding);
    return false;
  } catch (const std::invalid_argument&) {
    return true;
  }
}

void TestLiveShadowEvidenceGate(Runner& runner) {
  const auto passing_records = PassingEvidenceRecords();
  const auto passing = DumpEvidence(passing_records);
  const auto summary = active::ValidateLiveShadowEvidenceJsonl(
      passing, EvidenceBinding());
  runner.Check(summary.action_frames == 100 &&
                   summary.maximum_normalized_abs == 0.0 &&
                   summary.minimum_target_position_margin_rad == 0.1,
               "exact 100-frame promoted shadow PASS accepted");

  auto failed = passing_records;
  failed.back()["passed"] = false;
  runner.Check(EvidenceRejected(DumpEvidence(failed)),
               "failed terminal shadow evidence rejected");

  auto diagnostic = passing_records;
  diagnostic.front()["artifact_class"] = "diagnostic_shadow_only";
  runner.Check(EvidenceRejected(DumpEvidence(diagnostic)),
               "diagnostic shadow evidence rejected");

  auto rejected_frame_diagnostic = passing_records;
  rejected_frame_diagnostic.at(11)["accepted"] = false;
  rejected_frame_diagnostic.at(11)["diagnostic_only"] = true;
  rejected_frame_diagnostic.at(11)["active_runtime_eligible"] = false;
  rejected_frame_diagnostic.back()["event"] =
      "diagnostic_session_complete";
  rejected_frame_diagnostic.back()["passed"] = false;
  rejected_frame_diagnostic.back()["diagnostic_only"] = true;
  rejected_frame_diagnostic.back()["active_runtime_eligible"] = false;
  runner.Check(
      EvidenceRejected(DumpEvidence(rejected_frame_diagnostic)),
      "continued rejected-frame diagnostic cannot authorize active runtime");

  auto unsafe_action = passing_records;
  unsafe_action.at(11)["accepted"] = false;
  runner.Check(EvidenceRejected(DumpEvidence(unsafe_action)),
               "unaccepted action frame rejected");

  auto bad_anchor = passing_records;
  bad_anchor.at(11)["pico_anchor_source_frame_index"] = 0;
  runner.Check(EvidenceRejected(DumpEvidence(bad_anchor)),
               "non-causal anchor rejected");

  auto trailing = passing;
  trailing += nlohmann::json({
      {"schema_version", 1},
      {"kind", "g1_true23_integrated_live_shadow_evidence"},
      {"event", "session_failed"},
      {"passed", false},
  }).dump();
  trailing.push_back('\n');
  runner.Check(EvidenceRejected(trailing),
               "trailing record after PASS rejected");

  auto no_terminal_lf = passing;
  no_terminal_lf.pop_back();
  runner.Check(EvidenceRejected(no_terminal_lf),
               "JSONL lacking terminal LF rejected");

  auto duplicate = passing;
  const std::string needle = "\"event\":\"session_start\"";
  const auto duplicate_offset = duplicate.find(needle);
  runner.Check(duplicate_offset != std::string::npos,
               "duplicate-key fixture located session event");
  duplicate.replace(
      duplicate_offset, needle.size(),
      needle + ",\"event\":\"session_start\"");
  runner.Check(EvidenceRejected(duplicate), "duplicate JSON key rejected");

  auto wrong_binding = EvidenceBinding();
  wrong_binding.promotion_sha256 = Hash('b');
  runner.Check(EvidenceRejected(passing, wrong_binding),
               "hand-crafted sidecar hash cannot replace loaded promotion binding");

  runner.Check(EvidenceRejected(DumpEvidence(PassingEvidenceRecords(99))),
               "fewer than 100 promoted action frames rejected");
}

void TestArtifactGate(Runner& runner) {
  runner.Check(active::ValidateActiveArtifactBinding(ValidArtifact()).empty(),
               "valid promotion opens artifact gate");

  auto diagnostic = ValidArtifact();
  diagnostic.stage = active::ArtifactStage::Diagnostic;
  diagnostic.deployment_ready = false;
  diagnostic.active_motor_control_authorized = false;
  const auto diagnostic_errors =
      active::ValidateActiveArtifactBinding(diagnostic);
  runner.Check(diagnostic_errors.size() >= 3,
               "diagnostic pair cannot be upgraded by gantry flag");

  auto generic = ValidArtifact();
  generic.decoder_output_dim = 29;
  runner.Check(!active::ValidateActiveArtifactBinding(generic).empty(),
               "generic 29-output model rejected");

  const auto document = nlohmann::json{
      {"schema_version", 2},
      {"kind", std::string(active::kActivePromotionKind)},
      {"robot_model", std::string(true23::kRobotModel)},
      {"decoder_output_dim", 23},
      {"mode_machine", 4},
      {"action_clip_value", 20.0},
      {"deployment_ready", true},
      {"active_motor_control_authorized", true},
      {"gantry_authorized", true},
      {"free_standing_authorized", false},
      {"decoder_output_semantics", "applied_safe_native_action"},
      {"previous_action_semantics", "applied_safe_native_action"},
      {"external_safe_target_transform_allowed", false},
      {"safe_target_transform_sha256",
       std::string(gear_sonic::true23::live::kSafeTargetTransformSha256)},
      {"source_promotion_sha256", Hash('1')},
      {"checkpoint_sha256", Hash('5')},
      {"lineage_sha256", Hash('6')},
      {"policy_state_sha256", Hash('7')},
      {"encoder_onnx_sha256", Hash('2')},
      {"decoder_onnx_sha256", Hash('3')},
      {"metadata_sha256", Hash('4')},
      {"full_campaign_aggregate_sha256", Hash('8')},
      {"full_campaign_shard_manifest_sha256", Hash('9')},
      {"live_shadow_evidence_sha256", Hash('a')},
      {"authorization_id", "reviewed-stage-one-fixture"},
  };
  const auto parsed = active::ParseActivePromotion(
      document, true, Hash('1'), Hash('2'), Hash('3'), Hash('4'), Hash('a'));
  runner.Check(active::ValidateActiveArtifactBinding(parsed).empty(),
               "active promotion exact hash binding accepted");
  auto wrong_hash = document;
  wrong_hash["encoder_onnx_sha256"] = Hash('5');
  bool rejected = false;
  try {
    (void)active::ParseActivePromotion(
        wrong_hash, true, Hash('1'), Hash('2'), Hash('3'), Hash('4'),
        Hash('a'));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  runner.Check(rejected, "active promotion wrong ONNX hash rejected");

  auto double_transform = document;
  double_transform["external_safe_target_transform_allowed"] = true;
  rejected = false;
  try {
    const auto parsed_double = active::ParseActivePromotion(
        double_transform, true, Hash('1'), Hash('2'), Hash('3'), Hash('4'),
        Hash('a'));
    rejected = !active::ValidateActiveArtifactBinding(parsed_double).empty();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  runner.Check(rejected, "active promotion double transform rejected");

  auto legacy = document;
  legacy["schema_version"] = 1;
  rejected = false;
  try {
    (void)active::ParseActivePromotion(
        legacy, true, Hash('1'), Hash('2'), Hash('3'), Hash('4'), Hash('a'));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  runner.Check(rejected, "legacy active promotion rejected");

  rejected = false;
  try {
    (void)active::ParseActivePromotion(
        document, true, Hash('1'), Hash('2'), Hash('3'), Hash('4'), Hash('b'));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  runner.Check(rejected, "unreviewed live-shadow evidence file rejected");

  auto unsafe_id = ValidArtifact();
  unsafe_id.authorization_id = "bad id";
  runner.Check(!active::ValidateActiveArtifactBinding(unsafe_id).empty(),
               "unsafe authorization-id rejected");
}

void TestFiveSampleMutationGate(Runner& runner) {
  active::GantrySafetyCore core(ValidArtifact());
  const auto base = 1'000'000'000LL;
  core.ObserveState(State(1, base, 3), base);
  runner.Check(!core.stopped() && !core.mutation_surface_allowed(),
               "initial wrong embodiment waits without mutation");
  for (std::uint32_t tick = 2; tick <= 5; ++tick) {
    const auto time = base + tick * 2'000'000LL;
    core.ObserveState(State(tick, time), time);
    runner.Check(!core.mutation_surface_allowed(),
                 "fewer than five advancing mode-4 states stay closed");
  }
  const auto sixth_time = base + 6 * 2'000'000LL;
  core.ObserveState(State(6, sixth_time), sixth_time);
  runner.Check(core.mutation_surface_allowed() &&
                   core.stable_samples() == 5,
               "fifth advancing CRC-valid mode-4 state opens surface");
  core.ObserveState(State(7, sixth_time + 2'000'000LL, 5),
                    sixth_time + 2'000'000LL);
  runner.Check(core.stopped() && core.fault() == active::Fault::WrongMode,
               "post-gate mode change irreversibly latches stop");
}

void TestNoCatchUpWriterCadence(Runner& runner) {
  const auto first_completion = 1'000'000'000LL;
  const auto first_deadline =
      active::NextNoCatchUpWriterDeadlineNs(first_completion);
  runner.Check(first_deadline == first_completion + active::kWriterPeriodNs,
               "writer waits one full period after completed write");
  const auto stalled_completion = first_deadline + 25'000'000LL;
  runner.Check(
      active::NextNoCatchUpWriterDeadlineNs(stalled_completion) ==
          stalled_completion + active::kWriterPeriodNs,
      "writer stall resets cadence without catch-up burst");
  bool rejected = false;
  try {
    (void)active::NextNoCatchUpWriterDeadlineNs(-1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  runner.Check(rejected, "invalid writer completion clock rejected");
}

void TestStateFaults(Runner& runner) {
  {
    active::GantrySafetyCore core(ValidArtifact());
    core.ObserveCrcFailure();
    runner.Check(core.fault() == active::Fault::CrcFailure,
                 "CRC failure latches before publisher gate");
  }
  {
    active::GantrySafetyCore core(ValidArtifact());
    const auto now = 2'000'000'000LL;
    core.ObserveState(State(10, now), now);
    core.ObserveState(State(9, now + 2'000'000LL), now + 2'000'000LL);
    runner.Check(core.fault() == active::Fault::TickRegression,
                 "tick regression latches");
  }
  {
    active::GantrySafetyCore core(ValidArtifact());
    const auto now = 3'000'000'000LL;
    core.ObserveState(State(1, now), now);
    core.ObserveState(State(1, now + 2'000'000LL), now + 2'000'000LL);
    core.CheckWatchdogs(now + active::kStateFreshnessNs + 1);
    runner.Check(core.fault() == active::Fault::StateStale,
                 "duplicate tick cannot refresh advancing-state watchdog");
  }
  {
    active::GantrySafetyCore core(ValidArtifact());
    auto sample = State(1, 4'000'000'000LL);
    sample.q[0] = std::numeric_limits<double>::quiet_NaN();
    core.ObserveState(sample, sample.received_monotonic_ns);
    runner.Check(core.fault() == active::Fault::StateNonFinite,
                 "non-finite joint telemetry latches");
  }
  {
    active::GantrySafetyCore core(ValidArtifact());
    auto sample = State(1, 5'000'000'000LL);
    sample.dq[0] = 33.0;
    core.ObserveState(sample, sample.received_monotonic_ns);
    runner.Check(core.fault() == active::Fault::JointVelocityLimit,
                 "per-joint hard velocity limit latches");
  }
  {
    active::GantrySafetyCore core(ValidArtifact());
    auto sample = State(1, 6'000'000'000LL);
    sample.tau_est[1] = 140.0;
    core.ObserveState(sample, sample.received_monotonic_ns);
    runner.Check(core.fault() == active::Fault::JointEffortLimit,
                 "per-joint hard effort limit latches");
  }
}

void TestOperatorAndCommandSafety(Runner& runner) {
  std::array<std::uint8_t, 40> remote{};
  remote[2] = static_cast<std::uint8_t>(1U << 5U);
  remote[3] = static_cast<std::uint8_t>(1U << 0U);
  auto decoded = active::DecodeWirelessOperator(remote);
  runner.Check(decoded.arm_pressed && decoded.deadman_held &&
                   !decoded.stop_pressed,
               "wireless A arm and L2 deadman decode exactly");
  remote[3] |= static_cast<std::uint8_t>(1U << 1U);
  decoded = active::DecodeWirelessOperator(remote);
  runner.Check(decoded.stop_pressed,
               "wireless B button decodes as explicit STOP");

  const auto now = 7'020'000'000LL;
  active::GantrySafetyCore core(ValidArtifact());
  OpenGate(core, 7'000'000'000LL);
  runner.Check(!core.policy_ready_for_arm(now),
               "arming stays closed before first fresh policy");
  core.SubmitPolicy(ZeroPolicy(now), now);
  runner.Check(core.policy_ready_for_arm(now),
               "fresh accepted policy opens policy-ready gate");
  runner.Check(
      !core.policy_ready_for_arm(now + active::kPolicyFreshnessNs + 1),
      "policy-ready gate closes after freshness deadline");
  core.ObserveOperator(
      {.arm_edge = true, .deadman_held = true, .stop_requested = false},
      now);
  runner.Check(core.armed(), "arm edge plus held deadman arms after gates");
  const auto command = core.BuildCommand(now + 1'000'000LL);
  for (const int excluded : true23::kExcludedHardwareJointIds) {
    const auto& joint = command[static_cast<std::size_t>(excluded)];
    runner.Check(joint.mode == 0 && joint.kp == 0.0 && joint.tau == 0.0,
                 "excluded motor stays disabled and never policy-driven");
  }
  for (const int included : true23::kHardwareJointIds) {
    const auto& joint = command[static_cast<std::size_t>(included)];
    runner.Check(joint.mode == 1 && joint.tau == 0.0 &&
                     std::isfinite(joint.q),
                 "included motor receives finite low-gain zero-FF command");
  }

  core.ObserveOperator(
      {.arm_edge = false, .deadman_held = false, .stop_requested = false},
      now + 2'000'000LL);
  runner.Check(core.fault() == active::Fault::DeadmanReleased,
               "deadman release latches stop");
  runner.Check(!core.policy_ready_for_arm(now + 2'000'000LL),
               "policy-ready gate closes after terminal stop");
  const auto damping = core.BuildCommand(now + 3'000'000LL);
  for (const int included : true23::kHardwareJointIds) {
    const auto& joint = damping[static_cast<std::size_t>(included)];
    runner.Check(joint.tau == 0.0 && joint.kp == 0.0 &&
                     joint.kd == active::kFailSafeKd,
                 "latched failure produces zero-torque damping");
  }

  active::GantrySafetyCore stop_core(ValidArtifact());
  stop_core.ObserveOperator(
      {.arm_edge = false, .deadman_held = false, .stop_requested = true},
      now);
  runner.Check(stop_core.fault() == active::Fault::OperatorStop,
               "explicit STOP latches without state or publisher");
}

void TestPolicyAndMapping(Runner& runner) {
  active::RealProprioWarmupGate warmup;
  for (std::uint64_t frame = 10; frame < 19; ++frame) {
    runner.Check(!warmup.Observe(frame),
                 "fewer than ten real frames cannot arm policy history");
  }
  runner.Check(warmup.Observe(19) && warmup.ready() &&
                   warmup.sample_count() == 10,
               "ten advancing real frames open policy history gate");
  active::RealProprioWarmupGate regressed_warmup;
  runner.Check(!regressed_warmup.Observe(20),
               "first real history frame waits");
  runner.Check(!regressed_warmup.Observe(20) &&
                   regressed_warmup.rejected(),
               "duplicate history frame permanently rejects warmup");
  active::RealProprioWarmupGate skipped_warmup;
  runner.Check(!skipped_warmup.Observe(30),
               "first skipped-frame fixture sample waits");
  runner.Check(!skipped_warmup.Observe(32) && skipped_warmup.rejected(),
               "skipped 50 Hz history frame permanently rejects warmup");

  const auto now = 8'020'000'000LL;
  active::GantrySafetyCore core(ValidArtifact());
  OpenGate(core, 8'000'000'000LL);
  auto policy = ZeroPolicy(now);
  policy.native_action[1] = 0.5F;
  core.SubmitPolicy(policy, now);
  core.ObserveOperator(
      {.arm_edge = true, .deadman_held = true, .stop_requested = false},
      now);
  const auto command = core.BuildCommand(now + 1'000'000LL);
  const auto expected_slot = static_cast<std::size_t>(
      true23::kNativeToHardwareMotorIds[1]);
  runner.Check(expected_slot == 6,
               "native IL23 index 1 maps to hardware motor 6");
  runner.Check(command[expected_slot].q >
                   gear_sonic::true23::live::kHardwareDefaultQ[6],
               "one native action affects exact mapped motor target");
  runner.Check(
      command[expected_slot].q -
              gear_sonic::true23::live::kHardwareDefaultQ[6] <=
          active::kStageOneTargetRateRadPerSecond *
              active::kControlPeriodSeconds + 1e-9,
      "target change is stage-one slew clamped");

  active::GantrySafetyCore magnitude(ValidArtifact());
  OpenGate(magnitude, 9'000'000'000LL);
  auto bad = ZeroPolicy(9'020'000'000LL);
  bad.native_action[0] = 20.01F;
  magnitude.SubmitPolicy(bad, bad.produced_monotonic_ns);
  runner.Check(magnitude.fault() == active::Fault::PolicyMagnitude,
               "stage-one native action magnitude hard gate latches");

  active::GantrySafetyCore stale(ValidArtifact());
  OpenGate(stale, 10'000'000'000LL);
  stale.SubmitPolicy(
      ZeroPolicy(10'000'000'000LL),
      10'000'000'000LL + active::kPolicyFreshnessNs + 1);
  runner.Check(stale.fault() == active::Fault::PolicyStale,
               "stale inference output latches");
}

void TestHoldSmokeIsPolicyFreeAndGated(Runner& runner) {
  active::HoldSmokeSafetyCore hold;
  const auto base = 11'000'000'000LL;
  for (std::uint32_t tick = 1; tick <= 4; ++tick) {
    const auto now = base + tick * 2'000'000LL;
    hold.ObserveState(State(tick, now), now);
  }
  runner.Check(!hold.mutation_surface_allowed(),
               "hold smoke cannot create publisher before five samples");
  const auto ready_time = base + 10'000'000LL;
  hold.ObserveState(State(5, ready_time), ready_time);
  runner.Check(hold.mutation_surface_allowed(),
               "hold smoke opens only after five exact state samples");
  hold.ObserveOperator(
      {.arm_edge = true, .deadman_held = true, .stop_requested = false},
      ready_time);
  runner.Check(!hold.armed(),
               "hold smoke cannot arm before publisher-ready handoff");
  hold.EnableOperatorArming();
  hold.ObserveOperator(
      {.arm_edge = true, .deadman_held = true, .stop_requested = false},
      ready_time);
  runner.Check(hold.armed(),
               "hold smoke arms only after explicit runtime handoff");
  const auto command = hold.BuildCommand(ready_time + 1'000'000LL);
  for (std::size_t compact = 0; compact < 23; ++compact) {
    const auto slot = static_cast<std::size_t>(
        true23::kHardwareJointIds[compact]);
    runner.Check(command[slot].mode == 1 && command[slot].tau == 0.0 &&
                     command[slot].kp ==
                         0.25 * active::kStageOneKp[compact],
                 "hold smoke uses fixed low gain and zero feedforward");
    runner.Check(std::abs(
                     command[slot].q -
                     gear_sonic::true23::live::kHardwareDefaultQ[compact]) <
                     1e-12,
                 "hold smoke target equals sampled arm posture");
  }
  for (const int excluded : true23::kExcludedHardwareJointIds) {
    runner.Check(command[static_cast<std::size_t>(excluded)].mode == 0,
                 "hold smoke keeps excluded slot disabled");
  }
  hold.ObserveOperator(
      {.arm_edge = false, .deadman_held = false, .stop_requested = false},
      ready_time + 2'000'000LL);
  runner.Check(hold.fault() == active::Fault::DeadmanReleased,
               "hold smoke deadman release latches STOP");
}

void TestNative124BindingAcquisitionAndStageOneCore(Runner& runner) {
  const auto binding = ValidNative124Artifact();
  runner.Check(
      active::ValidateNative124ActiveArtifactBinding(binding).empty(),
      "selected native124 artifact binding passes exact gate");
  auto wrong_hash = binding;
  wrong_hash.policy_sha256 = Hash('0');
  active::GantrySafetyCore rejected(wrong_hash);
  runner.Check(rejected.fault() == active::Fault::ArtifactRejected,
               "native124 wrong policy hash latches artifact fault");

  std::array<float, 23> measured{};
  std::array<float, 23> desired{};
  std::array<float, 23> velocity{};
  desired.fill(0.4F);
  velocity.fill(0.1F);
  active::Native124AcquisitionGate acquisition(measured);
  for (int frame = 0; frame < active::kNative124WarmStartFrames; ++frame) {
    const auto staged = acquisition.Advance(desired, velocity);
    runner.Check(staged.alpha == 0.0 && !staged.ready_for_arm,
                 "native124 acquisition holds measured start for 25 frames");
  }
  for (int frame = 1; frame <= active::kNative124ReferenceRampFrames; ++frame) {
    const auto staged = acquisition.Advance(desired, velocity);
    runner.Check(staged.alpha >= 0.0 && staged.alpha <= 1.0,
                 "native124 reference ramp alpha stays bounded");
    if (frame < active::kNative124ReferenceRampFrames) {
      runner.Check(!staged.ready_for_arm,
                   "native124 cannot arm before full reference ramp");
    } else {
      runner.Check(staged.ready_for_arm &&
                       std::abs(staged.position_native[0] - desired[0]) < 1e-6,
                   "native124 arms only after exact 25+100 acquisition");
    }
  }

  const auto now = 12'020'000'000LL;
  active::GantrySafetyCore core(binding);
  OpenGate(core, 12'000'000'000LL);
  auto policy = ZeroPolicy(now);
  policy.native_action[0] = 0.5F;
  core.SubmitPolicy(policy, now);
  core.ObserveOperator(
      {.arm_edge = true, .deadman_held = true, .stop_requested = false},
      now);
  const auto command = core.BuildCommand(now + 1'000'000LL);
  runner.Check(core.armed() && core.fault() == active::Fault::None,
               "selected native124 binding reaches stage-one arm core");
  for (const int excluded : true23::kExcludedHardwareJointIds) {
    runner.Check(command[static_cast<std::size_t>(excluded)].mode == 0,
                 "native124 stage one keeps excluded motor slot disabled");
  }
}

}  // namespace

int main() {
  Runner runner;
  TestLiveShadowEvidenceGate(runner);
  TestArtifactGate(runner);
  TestNoCatchUpWriterCadence(runner);
  TestFiveSampleMutationGate(runner);
  TestStateFaults(runner);
  TestOperatorAndCommandSafety(runner);
  TestPolicyAndMapping(runner);
  TestHoldSmokeIsPolicyFreeAndGated(runner);
  TestNative124BindingAcquisitionAndStageOneCore(runner);
  if (runner.failures() != 0) {
    std::cerr << runner.failures()
              << " true23 active-gantry check(s) failed\n";
    return 1;
  }
  std::cout << "true23 active gantry core harness: all checks passed\n";
  return 0;
}
