#include "true23_live_shadow_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

namespace live = gear_sonic::true23::live;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void TestExactEncoderLayout() {
  live::PicoEncoderTerms terms;
  terms.schema_version = live::kPicoTermsSchemaVersion;
  terms.kind = live::kPicoTermsKind;
  terms.source_frame_index = 1;
  terms.source_monotonic_ns = 1;
  terms.future_frame_offsets_s =
      live::kFutureFrameOffsetsSeconds;
  for (std::size_t index = 0;
       index < terms.command_multi_future_lower_body.size(); ++index) {
    terms.command_multi_future_lower_body[index] =
        static_cast<float>(index);
  }
  terms.vr_3point_local_target.fill(240.0F);
  terms.vr_3point_local_orn_target.fill(249.0F);
  terms.motion_anchor_ori_b.fill(261.0F);
  const auto input = live::BuildEncoderInput(terms);
  Require(input.size() == 267, "encoder size");
  Require(input[0] == 0.0F && input[239] == 239.0F,
          "command order");
  Require(input[240] == 240.0F && input[249] == 249.0F &&
              input[261] == 261.0F,
          "term boundaries");

  terms.future_frame_offsets_s[1] = -0.1;
  const auto errors = live::ValidatePicoEncoderTerms(terms);
  Require(
      std::find(
          errors.begin(), errors.end(),
          "future_frame_offsets_s") != errors.end(),
      "past-frame window must be rejected");
}

void TestExactCausalEncoderLayout() {
  live::CausalPicoEncoderTerms terms;
  terms.schema_version = live::kCausalPicoTermsSchemaVersion;
  terms.kind = live::kCausalPicoTermsKind;
  terms.reference_profile = live::kCausalReferenceProfile;
  terms.reference_contract_sha256 =
      live::kCausalReferenceContractSha256;
  terms.pico_anchor_source_frame_index = 9;
  terms.pico_anchor_monotonic_ns = 1'000'000'000;
  terms.robot_anchor_monotonic_ns = terms.pico_anchor_monotonic_ns;
  terms.robot_anchor_source_contract =
      live::kCausalRobotAnchorContract;
  terms.control_source_frame_index = 10;
  terms.control_monotonic_ns = 1'020'000'000;
  terms.control_derivative_contract =
      live::kCausalDerivativeContract;
  terms.sdk_derivatives_consumed = false;
  for (std::size_t index = 0;
       index < terms.causal_history_lower_body.size(); ++index) {
    terms.causal_history_lower_body[index] =
        static_cast<float>(index);
  }
  terms.vr_3point_local_target.fill(240.0F);
  terms.vr_3point_local_orn_target.fill(249.0F);
  terms.motion_anchor_ori_b.fill(261.0F);
  const auto input = live::BuildCausalEncoderInput(terms);
  Require(input[0] == 0.0F && input[239] == 239.0F,
          "causal-history command order");
  Require(input[240] == 240.0F && input[249] == 249.0F &&
              input[261] == 261.0F,
          "causal term boundaries");

  terms.control_monotonic_ns += 1;
  auto errors = live::ValidateCausalPicoEncoderTerms(terms);
  Require(
      std::find(errors.begin(), errors.end(), "control_monotonic_ns") !=
          errors.end(),
      "causal q10 timestamp mismatch must fail");
  terms.control_monotonic_ns -= 1;
  terms.kind = std::string(live::kPicoTermsKind);
  errors = live::ValidateCausalPicoEncoderTerms(terms);
  Require(
      std::find(errors.begin(), errors.end(), "producer_kind") !=
          errors.end(),
      "released packet namespace must fail causal validator");
}

void TestCausalLowStateQ9Q10Join() {
  live::CausalLowStateHistory history;
  for (int index = 0; index < 3; ++index) {
    live::TimedProprioSample sample;
    sample.received_monotonic_ns =
        1'000'000'000 + index * 20'000'000;
    sample.hardware_q.fill(static_cast<float>(index));
    sample.hardware_dq.fill(static_cast<float>(index + 10));
    sample.gyroscope.fill(static_cast<float>(index + 20));
    sample.quaternion_wxyz = {1.0F, 0.0F, 0.0F, 0.0F};
    Require(history.Push(sample), "advancing LowState history accepted");
  }
  live::CausalPicoReferenceTerms reference;
  reference.schema_version = live::kCausalPicoTermsSchemaVersion;
  reference.kind = live::kCausalReferenceTermsKind;
  reference.reference_profile = live::kCausalReferenceProfile;
  reference.reference_contract_sha256 =
      live::kCausalReferenceContractSha256;
  reference.pico_anchor_source_frame_index = 9;
  reference.pico_anchor_monotonic_ns = 1'010'000'000;
  reference.control_source_frame_index = 10;
  reference.control_monotonic_ns = 1'030'000'000;
  reference.reference_anchor_quaternion_xyzw =
      {0.0F, 0.0F, 0.0F, 1.0F};
  for (std::size_t index = 0; index < 23; ++index) {
    reference.q_ref23_native[index] = static_cast<float>(100 + index);
    reference.qd_ref23_native[index] = static_cast<float>(200 + index);
    const auto canonical = static_cast<std::size_t>(
        gear_sonic::true23::kNativeToCanonicalIl29[index]);
    reference.anchor_joint_pos_il29[canonical] =
        reference.q_ref23_native[index];
    reference.proof_joint_pos_il29[canonical] =
        reference.q_ref23_native[index] +
        reference.qd_ref23_native[index] / 50.0F;
  }
  reference.control_derivative_contract =
      live::kCausalDerivativeContract;
  reference.sdk_derivatives_consumed = false;
  const auto joined = live::JoinCausalReferenceWithLowState(
      reference, history);
  Require(
      std::abs(joined.control_proprio_q10.hardware_q[0] - 1.5F) <
          1e-6F,
      "q10 proprio interpolated at exact timestamp");
  Require(
      joined.encoder_terms.robot_anchor_monotonic_ns ==
          reference.pico_anchor_monotonic_ns,
      "robot anchor bound to q9 timestamp");
  Require(
      joined.encoder_terms.motion_anchor_ori_b ==
          std::array<float, 6>{1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F},
      "C++ computes q9 relative reference orientation");
  (void)live::BuildCausalEncoderInput(joined.encoder_terms);
  std::array<float, 23> previous_raw_action{};
  previous_raw_action.fill(300.0F);
  const auto native124 = live::BuildNative124Observation(
      reference, joined, previous_raw_action);
  Require(
      native124.size() == 124 && native124[0] == 100.0F &&
          native124[22] == 122.0F && native124[23] == 200.0F &&
          native124[45] == 222.0F && native124[46] == 1.0F &&
          native124[52] == 21.5F && native124[123] == 300.0F,
      "native124 observation boundaries/order");
  std::array<float, 23> large_raw_action{};
  large_raw_action.fill(100.0F);
  const auto clamped =
      live::Native124RawActionToClampedTargets(large_raw_action);
  Require(
      clamped.raw_target_limit_clamps == 23,
      "native124 public targets must report every raw clamp");
  Require(
      clamped.maximum_raw_target_excess_rad > 1.0,
      "native124 public targets must report raw clamp excess");
  for (std::size_t index = 0; index < 23; ++index) {
    Require(
        clamped.hardware_targets[index] >= live::kHardwareLowerLimit[index] &&
            clamped.hardware_targets[index] <= live::kHardwareUpperLimit[index],
        "native124 applied target must remain inside URDF limit");
  }

  live::CausalLowStateHistory missing_history;
  bool rejected = false;
  try {
    (void)live::JoinCausalReferenceWithLowState(
        reference, missing_history);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "unbracketed q9/q10 LowState join rejected");
}

void TestCausalReferenceJsonParser() {
  std::array<float, live::kCommandDim> lower_body{};
  std::array<float, live::kVrPositionDim> vr_position{};
  std::array<float, live::kVrOrientationDim> vr_orientation{};
  const nlohmann::json document = {
      {"schema_version", live::kCausalPicoTermsSchemaVersion},
      {"kind", std::string(live::kCausalReferenceTermsKind)},
      {"reference_profile", std::string(live::kCausalReferenceProfile)},
      {"reference_contract_sha256",
       std::string(live::kCausalReferenceContractSha256)},
      {"pico_anchor_source_frame_index", 9},
      {"pico_anchor_monotonic_ns", 1'000'000'000},
      {"control_source_frame_index", 10},
      {"control_monotonic_ns", 1'020'000'000},
      {"causal_history_lower_body", lower_body},
      {"vr_3point_local_target", vr_position},
      {"vr_3point_local_orn_target", vr_orientation},
      {"reference_anchor_quaternion_xyzw",
       std::array<float, 4>{0.0F, 0.0F, 0.0F, 1.0F}},
      {"anchor_joint_pos_il29", std::array<float, 29>{}},
      {"proof_joint_pos_il29", std::array<float, 29>{}},
      {"q_ref23_native", std::array<float, 23>{}},
      {"qd_ref23_native", std::array<float, 23>{}},
      {"control_derivative_contract",
       std::string(live::kCausalDerivativeContract)},
      {"sdk_derivatives_consumed", false},
  };
  const auto parsed =
      live::ParseCausalPicoReferenceTermsDocument(document);
  Require(
      parsed.control_source_frame_index == 10,
      "exact causal reference JSON accepted");

  auto future_namespace = document;
  future_namespace["future_frame_offsets_s"] = {0.0, 0.1};
  bool rejected = false;
  try {
    (void)live::ParseCausalPicoReferenceTermsDocument(future_namespace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "mixed future-command JSON namespace rejected");

  auto extra_field = document;
  extra_field["unexpected"] = 1;
  rejected = false;
  try {
    (void)live::ParseCausalPicoReferenceTermsDocument(extra_field);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "non-exact causal JSON field set rejected");
}

void TestTermMajorHistoryAndFixedSlots() {
  live::ProprioSource source;
  for (std::size_t index = 0;
       index < source.hardware_q.size(); ++index) {
    source.hardware_q[index] =
        static_cast<float>(live::kHardwareDefaultQ[index]);
    source.hardware_dq[index] =
        static_cast<float>(index + 1);
    source.previous_action_native[index] =
        static_cast<float>(100 + index);
  }
  source.imu_gyroscope = {1.0F, 2.0F, 3.0F};
  source.imu_quaternion_wxyz = {1.0F, 0.0F, 0.0F, 0.0F};
  auto first = live::BuildProprioFrame(source);
  live::ProprioHistory history;
  history.Push(first);

  source.imu_gyroscope = {4.0F, 5.0F, 6.0F};
  source.hardware_q[0] += 0.25F;
  auto second = live::BuildProprioFrame(source);
  history.Push(second);
  const auto flat = history.Flatten();
  Require(flat.size() == 930, "history size");
  Require(
      flat[live::kAngvelHistoryOffset] == 1.0F &&
          flat[live::kAngvelHistoryOffset + 27] == 4.0F,
      "angular velocity history must be term-major oldest-to-newest");
  Require(
      flat[live::kJointPositionHistoryOffset] == 0.0F &&
          std::abs(
              flat[live::kJointPositionHistoryOffset + 9 * 29] -
              0.25F) < 1e-6F,
      "joint history must be term-major");
  for (std::size_t frame = 0; frame < 10; ++frame) {
    for (const auto block : {
             live::kJointPositionHistoryOffset,
             live::kJointVelocityHistoryOffset,
             live::kPreviousActionHistoryOffset}) {
      for (const int missing : live::kMissingCanonicalIl29) {
        Require(
            flat[block + frame * 29 +
                 static_cast<std::size_t>(missing)] == 0.0F,
            "missing canonical slot must be zero");
      }
    }
  }

  std::array<float, 64> token{};
  token[0] = 7.0F;
  const auto decoder = live::BuildDecoderInput(token, flat);
  Require(
      decoder[0] == 7.0F &&
          decoder[64 + live::kAngvelHistoryOffset] == 1.0F &&
          decoder[64 + live::kJointPositionHistoryOffset + 9 * 29] ==
              flat[live::kJointPositionHistoryOffset + 9 * 29],
      "decoder must be token then term-major history");
}

void TestOutputAssessment() {
  std::array<float, 23> action{};
  auto assessment =
      live::AssessOutput(action, std::nullopt, 0.02);
  Require(assessment.finite, "zero output finite");
  Require(
      assessment.target_limit_violations == 0,
      "default pose within limits");
  Require(!assessment.slew_checked, "first output no slew");

  std::array<float, 23> previous{};
  action[0] = 0.1F;
  assessment =
      live::AssessOutput(action, previous, 0.02);
  Require(assessment.slew_checked, "second output checks slew");
  Require(
      assessment.target_limit_violations == 0 &&
          assessment.target_slew_violations == 0,
      "small output within physical bounds");

  action[0] = std::numeric_limits<float>::quiet_NaN();
  assessment =
      live::AssessOutput(action, previous, 0.02);
  Require(!assessment.finite, "non-finite output rejected");
}

void TestNative124RejectedFrameDiagnosticGate() {
  const auto accepted = live::AssessNative124ShadowFrame(
      1.0, 1'000'000, 1'000'000, 2'000'000, 0, 0.0);
  Require(
      accepted.accepted && accepted.rejection_reasons.empty(),
      "native124 in-envelope frame accepted");

  const auto rejected = live::AssessNative124ShadowFrame(
      6.1, 20'000'001, 40'000'001, 100'000'001, 5, 1.500001);
  Require(!rejected.accepted, "native124 diagnostic preserves rejection");
  Require(
      rejected.rejection_reasons ==
          std::vector<std::string>{
              "raw_action_abs", "inference_deadline",
              "lowstate_freshness", "pico_end_to_end_freshness",
              "raw_target_limit_clamps", "raw_target_excess"},
      "native124 diagnostic records every exact gate reason");
  Require(
      live::kNative124RejectedDiagnosticFrames == 100,
      "native124 rejected-frame diagnostic remains exactly 100 frames");
}

nlohmann::json ExactSafeTargetTransform() {
  return nlohmann::json::parse(R"json(
{"action_scale_hardware":[0.55,0.35,0.55,0.35,0.44,0.44,0.55,0.35,0.55,0.35,0.44,0.44,0.55,0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44],"constants_sha256":"9b07ebcd080e18d7e574755cd399275e89f25583a50214b3678581b06540c113","default_q_hardware":[-0.312,0.0,0.0,0.669,-0.363,0.0,-0.312,0.0,0.0,0.669,-0.363,0.0,0.0,0.2,0.2,0.0,0.6,0.0,0.2,-0.2,0.0,0.6,0.0],"encoder_bias_envelope_rad":0.01,"float_dtype":"float32","formula":"hw=raw_native[index(ISAACLAB_TO_MUJOCO_DOF)];d=hw*scale;d_safe=where(d>=0,p*tanh(d/p),n*tanh(d/n));p=soft_hi-margin-default;n=default-(soft_lo+margin);target_unbiased=default+d_safe;safe_hw=d_safe/scale;safe_native=safe_hw[index(MUJOCO_TO_ISAACLAB_DOF)]","formula_sha256":"edb425f1a1df9ec4d2b1d192f282f9a13eedd113b1ee72638bb1c7382e90f8b6","guaranteed_post_bias_guard_rad":0.0019,"hard_lower_hardware":[-2.5307,-0.5236,-2.7576,-0.087267,-0.87267,-0.2618,-2.5307,-2.9671,-2.7576,-0.087267,-0.87267,-0.2618,-2.618,-3.0892,-1.5882,-2.618,-1.0472,-1.97222,-3.0892,-2.2515,-2.618,-1.0472,-1.97222],"hard_upper_hardware":[2.8798,2.9671,2.7576,2.8798,0.5236,0.2618,2.8798,0.5236,2.7576,2.8798,0.5236,0.2618,2.618,2.6704,2.2515,2.618,2.0944,1.97222,2.6704,1.5882,2.618,2.0944,1.97222],"inner_lower_hardware":[-2.248175,-0.33706500000000017,-2.46984,0.07308634999999984,-0.7908565,-0.22361999999999999,-2.248175,-2.7805649999999997,-2.46984,0.07308634999999984,-0.7908565,-0.22361999999999999,-2.3442,-2.78922,-1.384215,-2.3442,-0.8781199999999999,-1.762998,-2.78922,-2.047515,-2.3442,-0.8781199999999999,-1.762998],"inner_upper_hardware":[2.597275,2.7805649999999997,2.46984,2.7194466499999996,0.44178649999999997,0.22361999999999999,2.597275,0.33706500000000017,2.46984,2.7194466499999996,0.44178649999999997,0.22361999999999999,2.3442,2.3704199999999997,2.047515,2.3442,1.9253199999999997,1.762998,2.3704199999999997,1.384215,2.3442,1.9253199999999997,1.762998],"input_order":"native_isaaclab_23","isaaclab_to_mujoco":[0,3,7,11,15,19,1,4,8,12,16,20,2,5,9,13,17,21,6,10,14,18,22],"kind":"asymmetric_zero_preserving_tanh_v1","margin_rad":0.012,"mujoco_to_isaaclab":[0,6,12,1,7,13,18,2,8,14,19,3,9,15,20,4,10,16,21,5,11,17,22],"nominal_post_bias_guard_rad":0.002,"output_action_order":"native_isaaclab_23","previous_action_semantics":"applied_safe_native_action","schema":"g1_true23_safe_target_transform_v1","soft_limit_factor":0.9,"soft_lower_hardware":[-2.260175,-0.3490650000000002,-2.48184,0.061086349999999845,-0.8028565,-0.23562,-2.260175,-2.7925649999999997,-2.48184,0.061086349999999845,-0.8028565,-0.23562,-2.3562,-2.80122,-1.396215,-2.3562,-0.8901199999999999,-1.774998,-2.80122,-2.059515,-2.3562,-0.8901199999999999,-1.774998],"soft_upper_hardware":[2.609275,2.7925649999999997,2.48184,2.7314466499999996,0.4537865,0.23562,2.609275,0.3490650000000002,2.48184,2.7314466499999996,0.4537865,0.23562,2.3562,2.3824199999999998,2.059515,2.3562,1.9373199999999997,1.774998,2.3824199999999998,1.396215,2.3562,1.9373199999999997,1.774998],"target_order":"hardware_mujoco_23"}
)json");
}

void TestV11SafeOutputContractAndNoDoubleTransform() {
  const auto transform = ExactSafeTargetTransform();
  Require(
      gear_sonic::true23::Sha256CanonicalJson(transform) ==
          live::kSafeTargetTransformSha256,
      "exact V11 transform hash");
  nlohmann::json contract = {
      {"decoder_output_semantics",
       std::string(live::kAppliedSafeNativeActionSemantics)},
      {"external_safe_target_transform_allowed", false},
      {"previous_action_semantics",
       std::string(live::kAppliedSafeNativeActionSemantics)},
      {"safe_target_transform", transform},
  };
  nlohmann::json hashes = {
      {"safe_target_transform_sha256",
       std::string(live::kSafeTargetTransformSha256)},
  };
  nlohmann::json embedded = {
      {"decoder_output_semantics",
       std::string(live::kAppliedSafeNativeActionSemantics)},
      {"external_safe_target_transform_allowed", false},
      {"safe_target_transform_sha256",
       std::string(live::kSafeTargetTransformSha256)},
  };
  live::ValidateAppliedSafeDecoderContract(
      contract, hashes, embedded, embedded);

  const auto require_rejected = [&](const nlohmann::json& candidate_contract,
                                    const nlohmann::json& candidate_hashes,
                                    const nlohmann::json& candidate_embedded,
                                    const std::string& message) {
    bool rejected = false;
    try {
      live::ValidateAppliedSafeDecoderContract(
          candidate_contract, candidate_hashes, candidate_embedded,
          candidate_embedded);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, message);
  };

  auto changed = contract;
  changed["safe_target_transform"]["margin_rad"] = 0.013;
  require_rejected(changed, hashes, embedded, "wrong transform rejected");
  changed = hashes;
  changed["safe_target_transform_sha256"] = std::string(64, '0');
  require_rejected(contract, changed, embedded, "wrong transform hash rejected");
  changed = contract;
  changed["external_safe_target_transform_allowed"] = true;
  require_rejected(changed, hashes, embedded, "external transform rejected");
  changed = contract;
  changed["decoder_output_semantics"] = "raw_native_action";
  require_rejected(changed, hashes, embedded, "raw output semantics rejected");
  changed = contract;
  changed.erase("previous_action_semantics");
  require_rejected(
      changed, hashes, embedded, "missing applied-safe history rejected");

  std::array<float, 23> applied_safe{};
  applied_safe[0] = 2.0F;
  const auto targets =
      live::AppliedSafeNativeActionToHardwareTargets(applied_safe);
  Require(
      std::abs(
          targets[0] -
          (live::kHardwareDefaultQ[0] +
           2.0 * gear_sonic::true23::kHardwareActionScale[0])) < 1e-12,
      "decoder output receives affine mapping only, without second tanh");
}

}  // namespace

int main() {
  try {
    TestExactEncoderLayout();
    TestExactCausalEncoderLayout();
    TestCausalLowStateQ9Q10Join();
    TestCausalReferenceJsonParser();
    TestTermMajorHistoryAndFixedSlots();
    TestOutputAssessment();
    TestNative124RejectedFrameDiagnosticGate();
    TestV11SafeOutputContractAndNoDoubleTransform();
    std::cout << "true23 live-shadow core harness: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "true23 live-shadow core harness: FAIL: "
              << error.what() << '\n';
    return 1;
  }
}
