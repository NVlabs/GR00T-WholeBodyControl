#pragma once

// Dependency-free core for native-23 read-only live inference. It owns only
// observation construction, history, permutations, and numeric assessments.

#include "true23_shadow_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gear_sonic::true23::live {

inline constexpr int kEvidenceSchemaVersion = 1;
inline constexpr std::string_view kEvidenceKind =
    "g1_true23_integrated_live_shadow_evidence";
inline constexpr std::string_view kProducerKind =
    "g1_true23_integrated_readonly_shadow_probe";
inline constexpr int kPicoTermsSchemaVersion = 1;
inline constexpr std::string_view kPicoTermsKind =
    "g1_true23_pico_encoder_terms";
inline constexpr int kCausalPicoTermsSchemaVersion = 2;
inline constexpr std::string_view kCausalPicoTermsKind =
    "g1_true23_causal_history_encoder_terms";
inline constexpr std::string_view kCausalReferenceTermsKind =
    "g1_true23_causal_history_reference_terms";
inline constexpr std::string_view kCausalReferenceProfile =
    "true23_causal_step1_history_0p02s_v1";
inline constexpr std::string_view kCausalReferenceContractSha256 =
    "bd046467325fe7f7f585fd692f01223ed7a3b2742c51ed414072c98fe12806f7";
inline constexpr std::string_view kCausalDerivativeContract =
    "soma_il29_q_50hz_forward_difference_dq_v1";
inline constexpr std::string_view kCausalRobotAnchorContract =
    "robot_imu_quaternion_interpolated_at_pico_q9_v1";
inline constexpr int kSafeTargetDiagnosticSchemaVersion = 2;
inline constexpr std::string_view kAppliedSafeNativeActionSemantics =
    "applied_safe_native_action";
inline constexpr std::string_view kSafeTargetTransformSha256 =
    "74f2277042da83e81ee8a37d90ba6e723bf6e0651ee9b9987ee7effc78fca516";
inline constexpr std::size_t kCommandDim = 240;
inline constexpr std::size_t kVrPositionDim = 9;
inline constexpr std::size_t kVrOrientationDim = 12;
inline constexpr std::size_t kAnchorOrientationDim = 6;
inline constexpr std::size_t kNative124ObservationDim = 124;
inline constexpr int kNative124RejectedDiagnosticFrames = 100;
inline constexpr double kNative124ShadowMaximumRawAction = 6.0;
inline constexpr int kNative124ShadowMaximumRawTargetClamps = 4;
inline constexpr double kNative124ShadowMaximumRawTargetExcessRad = 1.5;
inline constexpr std::int64_t kNative124ShadowStateFreshnessNs = 40'000'000;
inline constexpr std::int64_t kNative124ShadowPicoFreshnessNs = 100'000'000;
inline constexpr std::int64_t kNative124ShadowFutureClockToleranceNs = 5'000'000;
inline constexpr std::int64_t kNative124ShadowInferenceDeadlineNs = 20'000'000;
inline constexpr std::size_t kProprioFrameDim = 93;
inline constexpr std::size_t kProprioHistoryDim = 930;
inline constexpr std::size_t kAngvelHistoryOffset = 0;
inline constexpr std::size_t kJointPositionHistoryOffset = 30;
inline constexpr std::size_t kJointVelocityHistoryOffset = 320;
inline constexpr std::size_t kPreviousActionHistoryOffset = 610;
inline constexpr std::size_t kGravityHistoryOffset = 900;
inline constexpr double kControlPeriodSeconds = 0.02;
inline constexpr std::array<double, 10> kFutureFrameOffsetsSeconds = {
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
};
inline constexpr std::array<int, 6> kMissingCanonicalIl29 = {
    5, 8, 25, 26, 27, 28,
};

inline constexpr std::array<double, 23> kHardwareDefaultQ = {
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0,
    0.2, 0.2, 0.0, 0.6, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0,
};
inline constexpr std::array<double, 23> kHardwareLowerLimit = {
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618,
    -3.0892, -1.5882, -2.618, -1.0472, -1.972222054,
    -3.0892, -2.2515, -2.618, -1.0472, -1.972222054,
};
inline constexpr std::array<double, 23> kHardwareUpperLimit = {
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618,
    2.6704, 2.2515, 2.618, 2.0944, 1.972222054,
    2.6704, 1.5882, 2.618, 2.0944, 1.972222054,
};
inline constexpr std::array<double, 23> kHardwareVelocityLimit = {
    32.0, 20.0, 32.0, 20.0, 30.0, 30.0,
    32.0, 20.0, 32.0, 20.0, 30.0, 30.0,
    32.0,
    37.0, 37.0, 37.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0, 37.0,
};
inline constexpr std::array<double, 23> kPublicNative124ActionScale = {
    0.548, 0.548, 0.439, 0.351, 0.351, 0.439, 0.439, 0.548,
    0.548, 0.439, 0.439, 0.351, 0.351, 0.439, 0.439, 0.439,
    0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439,
};

struct PicoEncoderTerms {
  int schema_version = 0;
  std::string kind;
  std::uint64_t source_frame_index = 0;
  std::int64_t source_monotonic_ns = 0;
  std::array<double, 10> future_frame_offsets_s{};
  std::array<float, kCommandDim> command_multi_future_lower_body{};
  std::array<float, kVrPositionDim> vr_3point_local_target{};
  std::array<float, kVrOrientationDim> vr_3point_local_orn_target{};
  std::array<float, kAnchorOrientationDim> motion_anchor_ori_b{};
};

struct CausalPicoEncoderTerms {
  int schema_version = 0;
  std::string kind;
  std::string reference_profile;
  std::string reference_contract_sha256;
  std::uint64_t pico_anchor_source_frame_index = 0;
  std::int64_t pico_anchor_monotonic_ns = 0;
  std::int64_t robot_anchor_monotonic_ns = 0;
  std::string robot_anchor_source_contract;
  std::uint64_t control_source_frame_index = 0;
  std::int64_t control_monotonic_ns = 0;
  std::array<float, kCommandDim> causal_history_lower_body{};
  std::array<float, kVrPositionDim> vr_3point_local_target{};
  std::array<float, kVrOrientationDim> vr_3point_local_orn_target{};
  std::array<float, kAnchorOrientationDim> motion_anchor_ori_b{};
  std::string control_derivative_contract;
  bool sdk_derivatives_consumed = true;
};

struct CausalPicoReferenceTerms {
  int schema_version = 0;
  std::string kind;
  std::string reference_profile;
  std::string reference_contract_sha256;
  std::uint64_t pico_anchor_source_frame_index = 0;
  std::int64_t pico_anchor_monotonic_ns = 0;
  std::uint64_t control_source_frame_index = 0;
  std::int64_t control_monotonic_ns = 0;
  std::array<float, kCommandDim> causal_history_lower_body{};
  std::array<float, kVrPositionDim> vr_3point_local_target{};
  std::array<float, kVrOrientationDim> vr_3point_local_orn_target{};
  std::array<float, 4> reference_anchor_quaternion_xyzw{};
  std::array<float, 29> anchor_joint_pos_il29{};
  std::array<float, 29> proof_joint_pos_il29{};
  std::array<float, 23> q_ref23_native{};
  std::array<float, 23> qd_ref23_native{};
  std::string control_derivative_contract;
  bool sdk_derivatives_consumed = true;
};

struct TimedProprioSample {
  std::int64_t received_monotonic_ns = 0;
  std::array<float, 23> hardware_q{};
  std::array<float, 23> hardware_dq{};
  std::array<float, 3> gyroscope{};
  std::array<float, 4> quaternion_wxyz{};
};

inline bool IsFinite(std::span<const float> values) {
  return std::all_of(
      values.begin(), values.end(),
      [](float value) { return std::isfinite(value); });
}

inline std::vector<std::string> ValidatePicoEncoderTerms(
    const PicoEncoderTerms& terms) {
  std::vector<std::string> errors;
  if (terms.schema_version != kPicoTermsSchemaVersion) {
    errors.emplace_back("producer_schema_version");
  }
  if (terms.kind != kPicoTermsKind) {
    errors.emplace_back("producer_kind");
  }
  if (terms.source_monotonic_ns <= 0) {
    errors.emplace_back("source_monotonic_ns");
  }
  for (std::size_t index = 0;
       index < terms.future_frame_offsets_s.size(); ++index) {
    if (!std::isfinite(terms.future_frame_offsets_s[index]) ||
        std::abs(
            terms.future_frame_offsets_s[index] -
            kFutureFrameOffsetsSeconds[index]) > 1e-7) {
      errors.emplace_back("future_frame_offsets_s");
      break;
    }
  }
  if (!IsFinite(terms.command_multi_future_lower_body)) {
    errors.emplace_back("command_multi_future_lower_body");
  }
  if (!IsFinite(terms.vr_3point_local_target)) {
    errors.emplace_back("vr_3point_local_target");
  }
  if (!IsFinite(terms.vr_3point_local_orn_target)) {
    errors.emplace_back("vr_3point_local_orn_target");
  }
  if (!IsFinite(terms.motion_anchor_ori_b)) {
    errors.emplace_back("motion_anchor_ori_b");
  }
  return errors;
}

inline std::array<float, kEncoderInputDim> BuildEncoderInput(
    const PicoEncoderTerms& terms) {
  const auto errors = ValidatePicoEncoderTerms(terms);
  if (!errors.empty()) {
    throw std::invalid_argument(
        "PICO semantic encoder terms are incomplete; first rejected field: " +
        errors.front());
  }
  std::array<float, kEncoderInputDim> result{};
  auto output = result.begin();
  output = std::copy(
      terms.command_multi_future_lower_body.begin(),
      terms.command_multi_future_lower_body.end(), output);
  output = std::copy(
      terms.vr_3point_local_target.begin(),
      terms.vr_3point_local_target.end(), output);
  output = std::copy(
      terms.vr_3point_local_orn_target.begin(),
      terms.vr_3point_local_orn_target.end(), output);
  output = std::copy(
      terms.motion_anchor_ori_b.begin(),
      terms.motion_anchor_ori_b.end(), output);
  if (output != result.end()) {
    throw std::logic_error("encoder input layout drift");
  }
  return result;
}

inline std::vector<std::string> ValidateCausalPicoEncoderTerms(
    const CausalPicoEncoderTerms& terms) {
  std::vector<std::string> errors;
  if (terms.schema_version != kCausalPicoTermsSchemaVersion) {
    errors.emplace_back("producer_schema_version");
  }
  if (terms.kind != kCausalPicoTermsKind) {
    errors.emplace_back("producer_kind");
  }
  if (terms.reference_profile != kCausalReferenceProfile) {
    errors.emplace_back("reference_profile");
  }
  if (terms.reference_contract_sha256 !=
      kCausalReferenceContractSha256) {
    errors.emplace_back("reference_contract_sha256");
  }
  if (terms.pico_anchor_monotonic_ns <= 0) {
    errors.emplace_back("pico_anchor_monotonic_ns");
  }
  if (terms.robot_anchor_monotonic_ns !=
      terms.pico_anchor_monotonic_ns) {
    errors.emplace_back("robot_anchor_monotonic_ns");
  }
  if (terms.robot_anchor_source_contract !=
      kCausalRobotAnchorContract) {
    errors.emplace_back("robot_anchor_source_contract");
  }
  if (terms.control_source_frame_index !=
      terms.pico_anchor_source_frame_index + 1) {
    errors.emplace_back("control_source_frame_index");
  }
  if (terms.control_monotonic_ns !=
      terms.pico_anchor_monotonic_ns + 20'000'000) {
    errors.emplace_back("control_monotonic_ns");
  }
  if (terms.control_derivative_contract !=
      kCausalDerivativeContract) {
    errors.emplace_back("control_derivative_contract");
  }
  if (terms.sdk_derivatives_consumed) {
    errors.emplace_back("sdk_derivatives_consumed");
  }
  if (!IsFinite(terms.causal_history_lower_body)) {
    errors.emplace_back("causal_history_lower_body");
  }
  if (!IsFinite(terms.vr_3point_local_target)) {
    errors.emplace_back("vr_3point_local_target");
  }
  if (!IsFinite(terms.vr_3point_local_orn_target)) {
    errors.emplace_back("vr_3point_local_orn_target");
  }
  if (!IsFinite(terms.motion_anchor_ori_b)) {
    errors.emplace_back("motion_anchor_ori_b");
  }
  return errors;
}

inline std::array<float, kEncoderInputDim> BuildCausalEncoderInput(
    const CausalPicoEncoderTerms& terms) {
  const auto errors = ValidateCausalPicoEncoderTerms(terms);
  if (!errors.empty()) {
    throw std::invalid_argument(
        "causal PICO encoder terms are incomplete; first rejected field: " +
        errors.front());
  }
  std::array<float, kEncoderInputDim> result{};
  auto output = result.begin();
  output = std::copy(
      terms.causal_history_lower_body.begin(),
      terms.causal_history_lower_body.end(), output);
  output = std::copy(
      terms.vr_3point_local_target.begin(),
      terms.vr_3point_local_target.end(), output);
  output = std::copy(
      terms.vr_3point_local_orn_target.begin(),
      terms.vr_3point_local_orn_target.end(), output);
  output = std::copy(
      terms.motion_anchor_ori_b.begin(),
      terms.motion_anchor_ori_b.end(), output);
  if (output != result.end()) {
    throw std::logic_error("causal encoder input layout drift");
  }
  return result;
}

inline std::vector<std::string> ValidateCausalPicoReferenceTerms(
    const CausalPicoReferenceTerms& terms) {
  CausalPicoEncoderTerms completed;
  completed.schema_version = terms.schema_version;
  completed.kind = terms.kind == kCausalReferenceTermsKind
                       ? std::string(kCausalPicoTermsKind)
                       : terms.kind;
  completed.reference_profile = terms.reference_profile;
  completed.reference_contract_sha256 = terms.reference_contract_sha256;
  completed.pico_anchor_source_frame_index =
      terms.pico_anchor_source_frame_index;
  completed.pico_anchor_monotonic_ns = terms.pico_anchor_monotonic_ns;
  completed.robot_anchor_monotonic_ns = terms.pico_anchor_monotonic_ns;
  completed.robot_anchor_source_contract =
      std::string(kCausalRobotAnchorContract);
  completed.control_source_frame_index = terms.control_source_frame_index;
  completed.control_monotonic_ns = terms.control_monotonic_ns;
  completed.causal_history_lower_body = terms.causal_history_lower_body;
  completed.vr_3point_local_target = terms.vr_3point_local_target;
  completed.vr_3point_local_orn_target =
      terms.vr_3point_local_orn_target;
  completed.motion_anchor_ori_b.fill(0.0F);
  completed.control_derivative_contract =
      terms.control_derivative_contract;
  completed.sdk_derivatives_consumed = terms.sdk_derivatives_consumed;
  auto errors = ValidateCausalPicoEncoderTerms(completed);
  if (terms.kind != kCausalReferenceTermsKind) {
    errors.emplace_back("producer_kind");
  }
  if (!IsFinite(terms.reference_anchor_quaternion_xyzw)) {
    errors.emplace_back("reference_anchor_quaternion_xyzw");
  } else {
    double norm_squared = 0.0;
    for (const auto value : terms.reference_anchor_quaternion_xyzw) {
      norm_squared += static_cast<double>(value) * value;
    }
    if (norm_squared < 1e-12) {
      errors.emplace_back("reference_anchor_quaternion_xyzw");
    }
  }
  if (!IsFinite(terms.q_ref23_native)) {
    errors.emplace_back("q_ref23_native");
  }
  if (!IsFinite(terms.qd_ref23_native)) {
    errors.emplace_back("qd_ref23_native");
  }
  if (!IsFinite(terms.anchor_joint_pos_il29)) {
    errors.emplace_back("anchor_joint_pos_il29");
  }
  if (!IsFinite(terms.proof_joint_pos_il29)) {
    errors.emplace_back("proof_joint_pos_il29");
  }
  for (std::size_t native = 0; native < 23; ++native) {
    const auto canonical = static_cast<std::size_t>(
        kNativeToCanonicalIl29[native]);
    const auto expected_q = terms.anchor_joint_pos_il29[canonical];
    const auto expected_dq =
        (terms.proof_joint_pos_il29[canonical] - expected_q) * 50.0F;
    if (std::abs(terms.q_ref23_native[native] - expected_q) > 1e-6F ||
        std::abs(terms.qd_ref23_native[native] - expected_dq) > 1e-3F) {
      errors.emplace_back("native23_reference_q9_q10_binding");
      break;
    }
  }
  return errors;
}

template <std::size_t Size>
inline std::array<float, Size> CausalJsonFloatArray(
    const nlohmann::json& root,
    std::string_view key) {
  const auto iterator = root.find(std::string(key));
  if (iterator == root.end() || !iterator->is_array() ||
      iterator->size() != Size) {
    throw std::invalid_argument(
        "causal PICO field has wrong size/type: " + std::string(key));
  }
  std::array<float, Size> result{};
  for (std::size_t index = 0; index < Size; ++index) {
    result[index] = iterator->at(index).get<float>();
  }
  return result;
}

inline CausalPicoReferenceTerms ParseCausalPicoReferenceTermsDocument(
    const nlohmann::json& root) {
  if (!root.is_object()) {
    throw std::invalid_argument(
        "causal PICO reference terms root must be object");
  }
  if (root.contains("future_frame_offsets_s") ||
      root.contains("command_multi_future_lower_body")) {
    throw std::invalid_argument(
        "causal artifact rejects future-command PICO fields");
  }
  const auto kind = root.at("kind").get<std::string>();
  if (kind != kCausalReferenceTermsKind) {
    throw std::invalid_argument(
        "causal artifact rejects non-reference PICO namespace: " + kind);
  }
  constexpr std::array<std::string_view, 18> expected_keys = {
      "schema_version",
      "kind",
      "reference_profile",
      "reference_contract_sha256",
      "pico_anchor_source_frame_index",
      "pico_anchor_monotonic_ns",
      "control_source_frame_index",
      "control_monotonic_ns",
      "causal_history_lower_body",
      "vr_3point_local_target",
      "vr_3point_local_orn_target",
      "reference_anchor_quaternion_xyzw",
      "anchor_joint_pos_il29",
      "proof_joint_pos_il29",
      "q_ref23_native",
      "qd_ref23_native",
      "control_derivative_contract",
      "sdk_derivatives_consumed",
  };
  if (root.size() != expected_keys.size()) {
    throw std::invalid_argument(
        "causal PICO reference field set is not exact");
  }
  for (const auto& [key, value] : root.items()) {
    (void)value;
    if (std::find(expected_keys.begin(), expected_keys.end(), key) ==
        expected_keys.end()) {
      throw std::invalid_argument(
          "causal PICO reference contains unexpected field: " + key);
    }
  }

  CausalPicoReferenceTerms result;
  result.schema_version = root.at("schema_version").get<int>();
  result.kind = kind;
  result.reference_profile = root.at("reference_profile").get<std::string>();
  result.reference_contract_sha256 =
      root.at("reference_contract_sha256").get<std::string>();
  result.pico_anchor_source_frame_index =
      root.at("pico_anchor_source_frame_index").get<std::uint64_t>();
  result.pico_anchor_monotonic_ns =
      root.at("pico_anchor_monotonic_ns").get<std::int64_t>();
  result.control_source_frame_index =
      root.at("control_source_frame_index").get<std::uint64_t>();
  result.control_monotonic_ns =
      root.at("control_monotonic_ns").get<std::int64_t>();
  result.causal_history_lower_body = CausalJsonFloatArray<kCommandDim>(
      root, "causal_history_lower_body");
  result.vr_3point_local_target = CausalJsonFloatArray<kVrPositionDim>(
      root, "vr_3point_local_target");
  result.vr_3point_local_orn_target =
      CausalJsonFloatArray<kVrOrientationDim>(
          root, "vr_3point_local_orn_target");
  result.reference_anchor_quaternion_xyzw =
      CausalJsonFloatArray<4>(
          root, "reference_anchor_quaternion_xyzw");
  result.anchor_joint_pos_il29 = CausalJsonFloatArray<29>(
      root, "anchor_joint_pos_il29");
  result.proof_joint_pos_il29 = CausalJsonFloatArray<29>(
      root, "proof_joint_pos_il29");
  result.q_ref23_native = CausalJsonFloatArray<23>(
      root, "q_ref23_native");
  result.qd_ref23_native = CausalJsonFloatArray<23>(
      root, "qd_ref23_native");
  result.control_derivative_contract =
      root.at("control_derivative_contract").get<std::string>();
  result.sdk_derivatives_consumed =
      root.at("sdk_derivatives_consumed").get<bool>();
  const auto errors = ValidateCausalPicoReferenceTerms(result);
  if (!errors.empty()) {
    throw std::invalid_argument(
        "causal PICO reference terms rejected: " + errors.front());
  }
  return result;
}

inline std::optional<std::array<float, 4>> NormalizeQuaternionWxyz(
    std::array<float, 4> value) {
  double norm_squared = 0.0;
  for (const auto component : value) {
    if (!std::isfinite(component)) {
      return std::nullopt;
    }
    norm_squared += static_cast<double>(component) * component;
  }
  if (norm_squared < 1e-12) {
    return std::nullopt;
  }
  const auto inverse = static_cast<float>(1.0 / std::sqrt(norm_squared));
  for (auto& component : value) {
    component *= inverse;
  }
  return value;
}

inline std::array<float, 4> SlerpWxyz(
    std::array<float, 4> left,
    std::array<float, 4> right,
    double alpha) {
  left = *NormalizeQuaternionWxyz(left);
  right = *NormalizeQuaternionWxyz(right);
  double dot = 0.0;
  for (std::size_t index = 0; index < 4; ++index) {
    dot += static_cast<double>(left[index]) * right[index];
  }
  if (dot < 0.0) {
    dot = -dot;
    for (auto& value : right) {
      value = -value;
    }
  }
  dot = std::clamp(dot, -1.0, 1.0);
  std::array<float, 4> result{};
  if (dot > 0.9995) {
    for (std::size_t index = 0; index < 4; ++index) {
      result[index] = static_cast<float>(
          left[index] + alpha * (right[index] - left[index]));
    }
    return *NormalizeQuaternionWxyz(result);
  }
  const auto theta = std::acos(dot);
  const auto sine = std::sin(theta);
  const auto left_weight = std::sin((1.0 - alpha) * theta) / sine;
  const auto right_weight = std::sin(alpha * theta) / sine;
  for (std::size_t index = 0; index < 4; ++index) {
    result[index] = static_cast<float>(
        left_weight * left[index] + right_weight * right[index]);
  }
  return *NormalizeQuaternionWxyz(result);
}

class CausalLowStateHistory {
 public:
  static constexpr std::size_t kCapacity = 512;
  static constexpr std::int64_t kMaximumBracketSpanNs = 20'000'000;

  bool Push(TimedProprioSample sample) {
    if (sample.received_monotonic_ns <= 0 ||
        (!samples_.empty() && sample.received_monotonic_ns <=
                                  samples_.back().received_monotonic_ns) ||
        !IsFinite(sample.hardware_q) || !IsFinite(sample.hardware_dq) ||
        !IsFinite(sample.gyroscope)) {
      return false;
    }
    const auto normalized = NormalizeQuaternionWxyz(sample.quaternion_wxyz);
    if (!normalized.has_value()) {
      return false;
    }
    sample.quaternion_wxyz = *normalized;
    if (samples_.size() == kCapacity) {
      samples_.erase(samples_.begin());
    }
    samples_.push_back(sample);
    return true;
  }

  [[nodiscard]] bool Covers(
      std::int64_t first_target_ns,
      std::int64_t last_target_ns) const {
    return first_target_ns > 0 &&
           last_target_ns >= first_target_ns &&
           !samples_.empty() &&
           samples_.front().received_monotonic_ns <= first_target_ns &&
           samples_.back().received_monotonic_ns >= last_target_ns;
  }

  std::optional<TimedProprioSample> Interpolate(
      std::int64_t target_ns) const {
    if (target_ns <= 0 || samples_.empty()) {
      return std::nullopt;
    }
    const auto right = std::lower_bound(
        samples_.begin(), samples_.end(), target_ns,
        [](const TimedProprioSample& sample, std::int64_t timestamp) {
          return sample.received_monotonic_ns < timestamp;
        });
    if (right == samples_.end()) {
      return std::nullopt;
    }
    if (right->received_monotonic_ns == target_ns) {
      return *right;
    }
    if (right == samples_.begin()) {
      return std::nullopt;
    }
    const auto left = std::prev(right);
    const auto span =
        right->received_monotonic_ns - left->received_monotonic_ns;
    if (span <= 0 || span > kMaximumBracketSpanNs) {
      return std::nullopt;
    }
    const auto alpha = static_cast<double>(
        target_ns - left->received_monotonic_ns) / span;
    TimedProprioSample result;
    result.received_monotonic_ns = target_ns;
    const auto interpolate = [alpha](float lhs, float rhs) {
      return static_cast<float>(lhs + alpha * (rhs - lhs));
    };
    for (std::size_t index = 0; index < 23; ++index) {
      result.hardware_q[index] = interpolate(
          left->hardware_q[index], right->hardware_q[index]);
      result.hardware_dq[index] = interpolate(
          left->hardware_dq[index], right->hardware_dq[index]);
    }
    for (std::size_t index = 0; index < 3; ++index) {
      result.gyroscope[index] = interpolate(
          left->gyroscope[index], right->gyroscope[index]);
    }
    result.quaternion_wxyz = SlerpWxyz(
        left->quaternion_wxyz, right->quaternion_wxyz, alpha);
    return result;
  }

 private:
  std::vector<TimedProprioSample> samples_;
};

inline std::array<float, 6> RelativeAnchorRotation6d(
    const std::array<float, 4>& robot_wxyz,
    const std::array<float, 4>& reference_xyzw) {
  const auto robot = *NormalizeQuaternionWxyz(robot_wxyz);
  const auto reference_wxyz = *NormalizeQuaternionWxyz(
      {reference_xyzw[3], reference_xyzw[0],
       reference_xyzw[1], reference_xyzw[2]});
  const auto rw = robot[0];
  const auto rx = -robot[1];
  const auto ry = -robot[2];
  const auto rz = -robot[3];
  const auto sw = reference_wxyz[0];
  const auto sx = reference_wxyz[1];
  const auto sy = reference_wxyz[2];
  const auto sz = reference_wxyz[3];
  const std::array<float, 4> relative_wxyz = {
      rw * sw - rx * sx - ry * sy - rz * sz,
      rw * sx + rx * sw + ry * sz - rz * sy,
      rw * sy - rx * sz + ry * sw + rz * sx,
      rw * sz + rx * sy - ry * sx + rz * sw,
  };
  const auto normalized = *NormalizeQuaternionWxyz(relative_wxyz);
  const auto w = normalized[0];
  const auto x = normalized[1];
  const auto y = normalized[2];
  const auto z = normalized[3];
  return {
      1.0F - 2.0F * (y * y + z * z),
      2.0F * (x * y - z * w),
      2.0F * (x * y + z * w),
      1.0F - 2.0F * (x * x + z * z),
      2.0F * (x * z - y * w),
      2.0F * (y * z + x * w),
  };
}

struct CausalEncoderJoin {
  CausalPicoEncoderTerms encoder_terms;
  TimedProprioSample control_proprio_q10;
};

inline CausalEncoderJoin JoinCausalReferenceWithLowState(
    const CausalPicoReferenceTerms& reference,
    const CausalLowStateHistory& low_state_history) {
  const auto errors = ValidateCausalPicoReferenceTerms(reference);
  if (!errors.empty()) {
    throw std::invalid_argument(
        "causal PICO reference terms rejected: " + errors.front());
  }
  const auto robot_q9 = low_state_history.Interpolate(
      reference.pico_anchor_monotonic_ns);
  const auto robot_q10 = low_state_history.Interpolate(
      reference.control_monotonic_ns);
  if (!robot_q9.has_value() || !robot_q10.has_value()) {
    throw std::invalid_argument(
        "LowState history does not bracket exact causal q9/q10");
  }
  CausalPicoEncoderTerms terms;
  terms.schema_version = kCausalPicoTermsSchemaVersion;
  terms.kind = std::string(kCausalPicoTermsKind);
  terms.reference_profile = reference.reference_profile;
  terms.reference_contract_sha256 = reference.reference_contract_sha256;
  terms.pico_anchor_source_frame_index =
      reference.pico_anchor_source_frame_index;
  terms.pico_anchor_monotonic_ns = reference.pico_anchor_monotonic_ns;
  terms.robot_anchor_monotonic_ns = reference.pico_anchor_monotonic_ns;
  terms.robot_anchor_source_contract =
      std::string(kCausalRobotAnchorContract);
  terms.control_source_frame_index = reference.control_source_frame_index;
  terms.control_monotonic_ns = reference.control_monotonic_ns;
  terms.causal_history_lower_body = reference.causal_history_lower_body;
  terms.vr_3point_local_target = reference.vr_3point_local_target;
  terms.vr_3point_local_orn_target =
      reference.vr_3point_local_orn_target;
  terms.motion_anchor_ori_b = RelativeAnchorRotation6d(
      robot_q9->quaternion_wxyz,
      reference.reference_anchor_quaternion_xyzw);
  terms.control_derivative_contract =
      reference.control_derivative_contract;
  terms.sdk_derivatives_consumed = reference.sdk_derivatives_consumed;
  const auto completed_errors = ValidateCausalPicoEncoderTerms(terms);
  if (!completed_errors.empty()) {
    throw std::logic_error(
        "joined causal encoder terms rejected: " +
        completed_errors.front());
  }
  return {.encoder_terms = terms, .control_proprio_q10 = *robot_q10};
}

inline std::array<float, 29> NativeToPaddedIl29(
    const std::array<float, 23>& native) {
  std::array<float, 29> result{};
  for (std::size_t index = 0; index < native.size(); ++index) {
    result[static_cast<std::size_t>(kNativeToCanonicalIl29[index])] =
        native[index];
  }
  return result;
}

inline std::array<float, 3> ProjectedGravity(
    std::array<float, 4> quaternion_wxyz) {
  const double norm = std::sqrt(
      static_cast<double>(quaternion_wxyz[0]) * quaternion_wxyz[0] +
      static_cast<double>(quaternion_wxyz[1]) * quaternion_wxyz[1] +
      static_cast<double>(quaternion_wxyz[2]) * quaternion_wxyz[2] +
      static_cast<double>(quaternion_wxyz[3]) * quaternion_wxyz[3]);
  if (!std::isfinite(norm) || norm < 1e-9) {
    throw std::invalid_argument("IMU quaternion is non-finite or degenerate");
  }
  for (auto& value : quaternion_wxyz) {
    value = static_cast<float>(value / norm);
  }
  const auto [w, x, y, z] = quaternion_wxyz;
  return {
      2.0F * (w * y - x * z),
      -2.0F * (w * x + y * z),
      -(1.0F - 2.0F * (x * x + y * y)),
  };
}

struct ProprioSource {
  std::array<float, 23> hardware_q{};
  std::array<float, 23> hardware_dq{};
  std::array<float, 3> imu_gyroscope{};
  std::array<float, 4> imu_quaternion_wxyz{1.0F, 0.0F, 0.0F, 0.0F};
  std::array<float, 23> previous_action_native{};
};

inline std::array<float, kNative124ObservationDim> BuildNative124Observation(
    const CausalPicoReferenceTerms& reference,
    const CausalEncoderJoin& joined,
    const std::array<float, 23>& previous_raw_action_native) {
  const auto reference_errors = ValidateCausalPicoReferenceTerms(reference);
  if (!reference_errors.empty()) {
    throw std::invalid_argument(
        "native124 reference rejected: " + reference_errors.front());
  }
  if (!IsFinite(previous_raw_action_native)) {
    throw std::invalid_argument("native124 previous action is non-finite");
  }
  std::array<float, 23> relative_hardware_q{};
  for (std::size_t index = 0; index < relative_hardware_q.size(); ++index) {
    relative_hardware_q[index] =
        joined.control_proprio_q10.hardware_q[index] -
        static_cast<float>(kHardwareDefaultQ[index]);
  }
  const auto relative_native_q = HardwareCompactToNative(relative_hardware_q);
  const auto native_dq =
      HardwareCompactToNative(joined.control_proprio_q10.hardware_dq);

  std::array<float, kNative124ObservationDim> result{};
  auto output = result.begin();
  output = std::copy(
      reference.q_ref23_native.begin(), reference.q_ref23_native.end(), output);
  output = std::copy(
      reference.qd_ref23_native.begin(), reference.qd_ref23_native.end(), output);
  output = std::copy(
      joined.encoder_terms.motion_anchor_ori_b.begin(),
      joined.encoder_terms.motion_anchor_ori_b.end(), output);
  output = std::copy(
      joined.control_proprio_q10.gyroscope.begin(),
      joined.control_proprio_q10.gyroscope.end(), output);
  output = std::copy(
      relative_native_q.begin(), relative_native_q.end(), output);
  output = std::copy(native_dq.begin(), native_dq.end(), output);
  output = std::copy(
      previous_raw_action_native.begin(),
      previous_raw_action_native.end(), output);
  if (output != result.end() || !IsFinite(result)) {
    throw std::logic_error("native124 observation layout drift");
  }
  return result;
}

struct Native124ClampedTargets {
  std::array<double, 23> hardware_targets{};
  int raw_target_limit_clamps = 0;
  double maximum_raw_target_excess_rad = 0.0;
};

struct Native124ShadowFrameAssessment {
  bool accepted = false;
  std::vector<std::string> rejection_reasons;
};

inline Native124ShadowFrameAssessment AssessNative124ShadowFrame(
    double raw_action_abs,
    std::int64_t inference_ns,
    std::int64_t lowstate_age_ns,
    std::int64_t end_to_end_age_ns,
    int raw_target_limit_clamps,
    double maximum_raw_target_excess_rad) {
  Native124ShadowFrameAssessment result;
  const auto reject = [&](bool condition, std::string_view reason) {
    if (condition) {
      result.rejection_reasons.emplace_back(reason);
    }
  };
  reject(
      !std::isfinite(raw_action_abs) ||
          raw_action_abs > kNative124ShadowMaximumRawAction,
      "raw_action_abs");
  reject(
      inference_ns < 0 ||
          inference_ns > kNative124ShadowInferenceDeadlineNs,
      "inference_deadline");
  reject(
      lowstate_age_ns < 0 ||
          lowstate_age_ns > kNative124ShadowStateFreshnessNs,
      "lowstate_freshness");
  reject(
      end_to_end_age_ns < -kNative124ShadowFutureClockToleranceNs ||
          end_to_end_age_ns > kNative124ShadowPicoFreshnessNs,
      "pico_end_to_end_freshness");
  reject(
      raw_target_limit_clamps < 0 ||
          raw_target_limit_clamps >
              kNative124ShadowMaximumRawTargetClamps,
      "raw_target_limit_clamps");
  reject(
      !std::isfinite(maximum_raw_target_excess_rad) ||
          maximum_raw_target_excess_rad < 0.0 ||
          maximum_raw_target_excess_rad >
              kNative124ShadowMaximumRawTargetExcessRad,
      "raw_target_excess");
  result.accepted = result.rejection_reasons.empty();
  return result;
}

inline Native124ClampedTargets Native124RawActionToClampedTargets(
    const std::array<float, 23>& raw_action_native) {
  if (!IsFinite(raw_action_native)) {
    throw std::invalid_argument("native124 raw action is non-finite");
  }
  std::array<float, 23> target_native{};
  const auto default_native = HardwareCompactToNative([] {
    std::array<float, 23> value{};
    std::transform(
        kHardwareDefaultQ.begin(), kHardwareDefaultQ.end(), value.begin(),
        [](double item) { return static_cast<float>(item); });
    return value;
  }());
  for (std::size_t index = 0; index < target_native.size(); ++index) {
    target_native[index] = static_cast<float>(
        default_native[index] +
        kPublicNative124ActionScale[index] * raw_action_native[index]);
  }
  const auto raw_hardware = NativeToHardwareCompact(target_native);
  Native124ClampedTargets result;
  for (std::size_t index = 0; index < result.hardware_targets.size(); ++index) {
    const auto raw = static_cast<double>(raw_hardware[index]);
    const auto clamped = std::clamp(
        raw, kHardwareLowerLimit[index], kHardwareUpperLimit[index]);
    result.hardware_targets[index] = clamped;
    if (clamped != raw) {
      ++result.raw_target_limit_clamps;
      result.maximum_raw_target_excess_rad = std::max(
          result.maximum_raw_target_excess_rad, std::abs(raw - clamped));
    }
  }
  return result;
}

inline std::array<float, kProprioFrameDim> BuildProprioFrame(
    const ProprioSource& source) {
  if (!IsFinite(source.hardware_q) ||
      !IsFinite(source.hardware_dq) ||
      !IsFinite(source.imu_gyroscope) ||
      !IsFinite(source.imu_quaternion_wxyz) ||
      !IsFinite(source.previous_action_native)) {
    throw std::invalid_argument("proprio source contains non-finite data");
  }
  std::array<float, 23> relative_hardware_q{};
  for (std::size_t index = 0; index < relative_hardware_q.size(); ++index) {
    relative_hardware_q[index] =
        source.hardware_q[index] -
        static_cast<float>(kHardwareDefaultQ[index]);
  }
  const auto relative_native_q =
      HardwareCompactToNative(relative_hardware_q);
  const auto native_dq =
      HardwareCompactToNative(source.hardware_dq);
  const auto padded_q = NativeToPaddedIl29(relative_native_q);
  const auto padded_dq = NativeToPaddedIl29(native_dq);
  const auto padded_previous =
      NativeToPaddedIl29(source.previous_action_native);
  const auto gravity = ProjectedGravity(source.imu_quaternion_wxyz);

  std::array<float, kProprioFrameDim> result{};
  auto output = result.begin();
  output = std::copy(
      source.imu_gyroscope.begin(),
      source.imu_gyroscope.end(), output);
  output = std::copy(padded_q.begin(), padded_q.end(), output);
  output = std::copy(padded_dq.begin(), padded_dq.end(), output);
  output = std::copy(
      padded_previous.begin(), padded_previous.end(), output);
  output = std::copy(gravity.begin(), gravity.end(), output);
  if (output != result.end()) {
    throw std::logic_error("proprio frame layout drift");
  }
  return result;
}

inline void ValidateFixedMissingSlots(
    std::span<const float, kProprioHistoryDim> history) {
  for (std::size_t frame_index = 0;
       frame_index < kHistoryLength; ++frame_index) {
    for (const auto block_offset : {
             kJointPositionHistoryOffset,
             kJointVelocityHistoryOffset,
             kPreviousActionHistoryOffset}) {
      for (const int missing_index : kMissingCanonicalIl29) {
        if (history[
                block_offset + frame_index * 29 +
                static_cast<std::size_t>(missing_index)] != 0.0F) {
          throw std::invalid_argument(
              "canonical IL29 missing proprio slot is non-zero");
        }
      }
    }
  }
}

class ProprioHistory {
 public:
  void Push(const std::array<float, kProprioFrameDim>& frame) {
    if (!IsFinite(frame)) {
      throw std::invalid_argument("proprio frame is non-finite");
    }
    if (size_ < kHistoryLength) {
      frames_[size_++] = frame;
      if (size_ == 1) {
        // First real sample supplies deterministic warm-up history.
        std::fill(frames_.begin() + 1, frames_.end(), frame);
        size_ = kHistoryLength;
      }
    } else {
      std::move(frames_.begin() + 1, frames_.end(), frames_.begin());
      frames_.back() = frame;
    }
  }

  [[nodiscard]] bool ready() const {
    return size_ == kHistoryLength;
  }

  [[nodiscard]] std::array<float, kProprioHistoryDim> Flatten() const {
    if (!ready()) {
      throw std::runtime_error("proprio history is not ready");
    }
    std::array<float, kProprioHistoryDim> result{};
    for (std::size_t frame = 0; frame < frames_.size(); ++frame) {
      std::copy_n(
          frames_[frame].begin(), 3,
          result.begin() + kAngvelHistoryOffset + frame * 3);
      std::copy_n(
          frames_[frame].begin() + 3, 29,
          result.begin() + kJointPositionHistoryOffset + frame * 29);
      std::copy_n(
          frames_[frame].begin() + 32, 29,
          result.begin() + kJointVelocityHistoryOffset + frame * 29);
      std::copy_n(
          frames_[frame].begin() + 61, 29,
          result.begin() + kPreviousActionHistoryOffset + frame * 29);
      std::copy_n(
          frames_[frame].begin() + 90, 3,
          result.begin() + kGravityHistoryOffset + frame * 3);
    }
    ValidateFixedMissingSlots(result);
    return result;
  }

 private:
  std::array<std::array<float, kProprioFrameDim>, kHistoryLength>
      frames_{};
  std::size_t size_ = 0;
};

inline std::array<float, kDecoderInputDim> BuildDecoderInput(
    const std::array<float, kEncoderOutputDim>& token,
    const std::array<float, kProprioHistoryDim>& history) {
  if (!IsFinite(token) || !IsFinite(history)) {
    throw std::invalid_argument("decoder input source is non-finite");
  }
  ValidateFixedMissingSlots(history);
  std::array<float, kDecoderInputDim> result{};
  auto output = std::copy(token.begin(), token.end(), result.begin());
  output = std::copy(history.begin(), history.end(), output);
  if (output != result.end()) {
    throw std::logic_error("decoder input layout drift");
  }
  return result;
}

struct OutputAssessment {
  bool finite = false;
  double normalized_max_abs = 0.0;
  double target_position_min_margin_rad =
      std::numeric_limits<double>::infinity();
  int target_limit_violations = 0;
  bool slew_checked = false;
  double target_slew_ratio_max = 0.0;
  int target_slew_violations = 0;
};

// The V11 decoder already embeds the asymmetric tanh safe-target transform.
// Deployment must therefore perform only the established native-to-hardware
// permutation and affine target conversion. Applying tanh again would change
// the trained policy and is forbidden by the artifact contract.
inline std::array<double, 23> AppliedSafeNativeActionToHardwareTargets(
    const std::array<float, 23>& applied_safe_native_action) {
  if (!IsFinite(applied_safe_native_action)) {
    throw std::invalid_argument("applied safe native action is non-finite");
  }
  const auto hardware_action =
      NativeToHardwareCompact(applied_safe_native_action);
  std::array<double, 23> targets{};
  for (std::size_t index = 0; index < targets.size(); ++index) {
    targets[index] =
        kHardwareDefaultQ[index] +
        static_cast<double>(hardware_action[index]) *
            kHardwareActionScale[index];
  }
  return targets;
}

inline void ValidateAppliedSafeDecoderContract(
    const nlohmann::json& contract,
    const nlohmann::json& hashes,
    const nlohmann::json& encoder_embedded,
    const nlohmann::json& decoder_embedded) {
  const auto require_applied_safe_fields = [](const nlohmann::json& value,
                                               std::string_view context) {
    if (!value.contains("decoder_output_semantics") ||
        value.at("decoder_output_semantics") !=
            kAppliedSafeNativeActionSemantics ||
        !value.contains("external_safe_target_transform_allowed") ||
        value.at("external_safe_target_transform_allowed") != false ||
        !value.contains("safe_target_transform_sha256") ||
        value.at("safe_target_transform_sha256") !=
            kSafeTargetTransformSha256) {
      throw std::invalid_argument(
          std::string(context) + " safe-output contract mismatch");
    }
  };

  if (!contract.contains("decoder_output_semantics") ||
      contract.at("decoder_output_semantics") !=
          kAppliedSafeNativeActionSemantics ||
      !contract.contains("external_safe_target_transform_allowed") ||
      contract.at("external_safe_target_transform_allowed") != false ||
      !contract.contains("previous_action_semantics") ||
      contract.at("previous_action_semantics") !=
          kAppliedSafeNativeActionSemantics ||
      !contract.contains("safe_target_transform") ||
      !contract.at("safe_target_transform").is_object()) {
    throw std::invalid_argument("diagnostic safe-output contract mismatch");
  }
  const auto& transform = contract.at("safe_target_transform");
  if (!transform.contains("previous_action_semantics") ||
      transform.at("previous_action_semantics") !=
          kAppliedSafeNativeActionSemantics ||
      true23::Sha256CanonicalJson(transform) != kSafeTargetTransformSha256) {
    throw std::invalid_argument(
        "diagnostic safe-target transform/hash mismatch");
  }
  if (!hashes.contains("safe_target_transform_sha256") ||
      hashes.at("safe_target_transform_sha256") !=
          kSafeTargetTransformSha256) {
    throw std::invalid_argument(
        "diagnostic safe-target hash binding mismatch");
  }
  require_applied_safe_fields(encoder_embedded, "encoder embedded");
  require_applied_safe_fields(decoder_embedded, "decoder embedded");
}

inline OutputAssessment AssessOutput(
    const std::array<float, 23>& native_action,
    const std::optional<std::array<float, 23>>&
        previous_native_action,
    double dt_seconds) {
  OutputAssessment result;
  result.finite = IsFinite(native_action);
  if (!result.finite) {
    return result;
  }
  const auto hardware_action = NativeToHardwareCompact(native_action);
  const auto hardware_targets =
      AppliedSafeNativeActionToHardwareTargets(native_action);
  std::optional<std::array<double, 23>> previous_targets;
  if (previous_native_action.has_value()) {
    if (!IsFinite(*previous_native_action) ||
        !std::isfinite(dt_seconds) || dt_seconds <= 0.0) {
      throw std::invalid_argument(
          "previous action or output sample period is invalid");
    }
    previous_targets = AppliedSafeNativeActionToHardwareTargets(
        *previous_native_action);
    result.slew_checked = true;
  }
  for (std::size_t index = 0; index < hardware_action.size(); ++index) {
    result.normalized_max_abs = std::max(
        result.normalized_max_abs,
        std::abs(static_cast<double>(hardware_action[index])));
    const double target = hardware_targets[index];
    const double margin = std::min(
        target - kHardwareLowerLimit[index],
        kHardwareUpperLimit[index] - target);
    result.target_position_min_margin_rad =
        std::min(result.target_position_min_margin_rad, margin);
    if (margin < 0.0) {
      ++result.target_limit_violations;
    }
    if (previous_targets.has_value()) {
      const double previous_target = (*previous_targets)[index];
      const double ratio =
          std::abs(target - previous_target) /
          (kHardwareVelocityLimit[index] * dt_seconds);
      result.target_slew_ratio_max =
          std::max(result.target_slew_ratio_max, ratio);
      if (ratio > 1.0) {
        ++result.target_slew_violations;
      }
    }
  }
  return result;
}

}  // namespace gear_sonic::true23::live
