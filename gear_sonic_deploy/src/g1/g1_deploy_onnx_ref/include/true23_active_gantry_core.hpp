#pragma once

// Dependency-light, fail-closed safety core for first native True23 gantry
// actuation.  DDS, ONNX Runtime, threads, and wall-clock policy live outside
// this file so every transition can be exercised without a robot.

#include "true23_live_shadow_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace gear_sonic::true23::active {

inline constexpr int kActivePromotionSchemaVersion = 2;
inline constexpr std::string_view kActivePromotionKind =
    "g1_true23_causal_gantry_active_promotion";
inline constexpr std::string_view kGantryAuthorizationPhrase =
    "I_CONFIRM_G1_TRUE23_STAGE1_GANTRY";
inline constexpr std::size_t kMotorSlotCount = 29;
inline constexpr std::int64_t kStateFreshnessNs = 40'000'000;
// PICO/reference latency is a high-level input bound.  Keep the local robot
// state watchdog at 40 ms, while allowing a bounded 100 ms high-level age for
// the initial gantry teleoperation campaign.
inline constexpr std::int64_t kPolicyFreshnessNs = 100'000'000;
inline constexpr std::int64_t kFutureClockToleranceNs = 5'000'000;
inline constexpr double kControlPeriodSeconds = 0.002;
inline constexpr std::int64_t kWriterPeriodNs = 2'000'000;
inline constexpr double kStageOneActionFraction = 0.10;
inline constexpr double kStageOneTargetRateRadPerSecond = 0.25;
inline constexpr double kTargetLimitMarginRad = 0.05;
inline constexpr double kTrainingActionClipValue = 20.0;
inline constexpr double kMaximumNormalizedAction = kTrainingActionClipValue;
inline constexpr int kMinimumPromotedShadowActionFrames = 100;
inline constexpr int kMinimumStageOnePostArmSeconds = 20;
inline constexpr int kMaximumStageOnePostArmSeconds = 30;
inline constexpr std::int64_t kShadowControlPeriodNs = 20'000'000;
inline constexpr std::int64_t kShadowInferenceDeadlineNs = 20'000'000;
inline constexpr std::size_t kMaximumShadowEvidenceBytes = 16U << 20U;
inline constexpr std::size_t kMaximumShadowEvidenceLineBytes = 2U << 20U;
inline constexpr double kMaximumShadowEvidenceAgeSeconds = 300.0;
inline constexpr std::string_view kSelectedNative124PolicySha256 =
    "cc644839807b6ef522e47b3bcb69845843aa345b4fb895847c76642830b5d2b9";
inline constexpr int kNative124WarmStartFrames = 25;
inline constexpr int kNative124ReferenceRampFrames = 100;

inline bool IsLowerSha256(std::string_view value);

inline std::int64_t NextNoCatchUpWriterDeadlineNs(
    std::int64_t completed_write_ns) {
  if (completed_write_ns < 0 ||
      completed_write_ns >
          std::numeric_limits<std::int64_t>::max() - kWriterPeriodNs) {
    throw std::invalid_argument("writer completion time is invalid");
  }
  return completed_write_ns + kWriterPeriodNs;
}

inline bool IsSafeAuthorizationId(std::string_view value) {
  if (value.size() < 8U || value.size() > 128U ||
      !((value.front() >= 'A' && value.front() <= 'Z') ||
        (value.front() >= 'a' && value.front() <= 'z') ||
        (value.front() >= '0' && value.front() <= '9'))) {
    return false;
  }
  return std::all_of(value.begin() + 1, value.end(), [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '.' ||
           character == '_' || character == ':' || character == '-';
  });
}

struct LiveShadowEvidenceBinding {
  std::string encoder_sha256;
  std::string decoder_sha256;
  std::string metadata_sha256;
  std::string promotion_sha256;
  std::string network;
  std::string pico_endpoint;
};

struct LiveShadowEvidenceSummary {
  int action_frames = 0;
  double maximum_normalized_abs = 0.0;
  double minimum_target_position_margin_rad = 0.0;
  double maximum_target_slew_ratio = 0.0;
};

inline LiveShadowEvidenceSummary ValidateLiveShadowEvidenceJsonl(
    std::string_view bytes,
    const LiveShadowEvidenceBinding& binding) {
  using nlohmann::json;
  const auto reject = [](std::string message) -> void {
    throw std::invalid_argument("live-shadow evidence: " + message);
  };
  if (bytes.empty() || bytes.size() > kMaximumShadowEvidenceBytes ||
      bytes.back() != '\n' || bytes.find('\r') != std::string_view::npos) {
    reject("bytes are empty, oversized, non-LF, or lack terminal LF");
  }
  for (const auto value : {
           binding.encoder_sha256, binding.decoder_sha256,
           binding.metadata_sha256, binding.promotion_sha256}) {
    if (!IsLowerSha256(value)) {
      reject("expected artifact binding contains invalid SHA-256");
    }
  }
  if (binding.network.empty() || binding.pico_endpoint.empty()) {
    reject("expected network/PICO binding is empty");
  }

  std::vector<json> records;
  std::size_t begin = 0;
  std::size_t line_number = 0;
  while (begin < bytes.size()) {
    const auto end = bytes.find('\n', begin);
    if (end == std::string_view::npos) {
      reject("terminal line is incomplete");
    }
    const auto line = bytes.substr(begin, end - begin);
    ++line_number;
    if (line.empty() || line.size() > kMaximumShadowEvidenceLineBytes) {
      reject("line " + std::to_string(line_number) +
             " is empty or oversized");
    }
    std::vector<std::unordered_set<std::string>> object_keys;
    const json::parser_callback_t callback =
        [&](int, json::parse_event_t event, json& parsed) {
          if (event == json::parse_event_t::object_start) {
            object_keys.emplace_back();
          } else if (event == json::parse_event_t::key) {
            const auto key = parsed.get<std::string>();
            if (object_keys.empty() ||
                !object_keys.back().insert(key).second) {
              throw std::invalid_argument("duplicate JSON key: " + key);
            }
          } else if (event == json::parse_event_t::object_end) {
            if (object_keys.empty()) {
              throw std::invalid_argument("JSON object stack underflow");
            }
            object_keys.pop_back();
          }
          return true;
        };
    try {
      auto record = json::parse(
          line.begin(), line.end(), callback, true, true);
      if (!record.is_object() || !object_keys.empty()) {
        reject("line " + std::to_string(line_number) +
               " is not one complete JSON object");
      }
      records.push_back(std::move(record));
    } catch (const std::invalid_argument&) {
      throw;
    } catch (const std::exception& error) {
      reject("line " + std::to_string(line_number) +
             " is invalid JSON: " + error.what());
    }
    begin = end + 1;
  }

  const auto exact_keys = [&](const json& value,
                              std::initializer_list<std::string_view> keys,
                              std::string_view context) {
    if (!value.is_object() || value.size() != keys.size()) {
      reject(std::string(context) + " field set is not exact");
    }
    for (const auto key : keys) {
      if (!value.contains(std::string(key))) {
        reject(std::string(context) + " missing field: " +
               std::string(key));
      }
    }
  };
  const auto integer = [&](const json& value,
                           std::string_view context) -> std::int64_t {
    if (!value.is_number_integer()) {
      reject(std::string(context) + " must be an integer");
    }
    return value.get<std::int64_t>();
  };
  const auto number = [&](const json& value,
                          std::string_view context) -> double {
    if (!value.is_number()) {
      reject(std::string(context) + " must be numeric");
    }
    const auto result = value.get<double>();
    if (!std::isfinite(result)) {
      reject(std::string(context) + " must be finite");
    }
    return result;
  };
  const auto common = [&](const json& value, std::string_view event) {
    if (value.at("schema_version") != live::kEvidenceSchemaVersion ||
        value.at("kind") != live::kEvidenceKind ||
        value.at("event") != event) {
      reject(std::string(event) + " identity mismatch");
    }
  };

  if (records.size() < 2U + (kHistoryLength - 1U) +
                               kMinimumPromotedShadowActionFrames + 1U) {
    reject("record count cannot prove the minimum promoted session");
  }
  const auto& session = records.front();
  exact_keys(
      session,
      {"schema_version", "kind", "event", "started_monotonic_ns",
       "reference_profile", "reference_contract_sha256", "artifact_class",
       "decoder_output_semantics", "external_safe_target_transform_applied",
       "encoder_sha256", "decoder_sha256", "metadata_sha256",
       "promotion_sha256", "network", "pico_endpoint",
       "requested_action_frames", "robot_mutation_authorized"},
      "session_start");
  common(session, "session_start");
  const auto requested_frames = integer(
      session.at("requested_action_frames"), "requested_action_frames");
  if (integer(session.at("started_monotonic_ns"), "started_monotonic_ns") <= 0 ||
      requested_frames < kMinimumPromotedShadowActionFrames ||
      requested_frames > 10'000 ||
      session.at("reference_profile") != live::kCausalReferenceProfile ||
      session.at("reference_contract_sha256") !=
          live::kCausalReferenceContractSha256 ||
      session.at("artifact_class") != "promoted_shadow" ||
      session.at("decoder_output_semantics") !=
          live::kAppliedSafeNativeActionSemantics ||
      session.at("external_safe_target_transform_applied") != false ||
      session.at("encoder_sha256") != binding.encoder_sha256 ||
      session.at("decoder_sha256") != binding.decoder_sha256 ||
      session.at("metadata_sha256") != binding.metadata_sha256 ||
      session.at("promotion_sha256") != binding.promotion_sha256 ||
      session.at("network") != binding.network ||
      session.at("pico_endpoint") != binding.pico_endpoint ||
      session.at("robot_mutation_authorized") != false) {
    reject("session_start contract/binding mismatch");
  }
  const auto expected_records =
      2U + (kHistoryLength - 1U) +
      static_cast<std::size_t>(requested_frames) + 1U;
  if (records.size() != expected_records) {
    reject("record count does not exactly match requested frames");
  }

  const auto& lowstate = records.at(1);
  exact_keys(
      lowstate,
      {"schema_version", "kind", "event", "mode_machine", "crc_rejects",
       "history_warmup_span_ns"},
      "lowstate_gate_open");
  common(lowstate, "lowstate_gate_open");
  if (integer(lowstate.at("mode_machine"), "mode_machine") !=
          kRequiredModeMachine ||
      integer(lowstate.at("crc_rejects"), "crc_rejects") != 0 ||
      integer(lowstate.at("history_warmup_span_ns"),
              "history_warmup_span_ns") != 40'000'000) {
    reject("LowState gate did not prove mode 4, CRC 0, and real warmup");
  }

  std::optional<std::int64_t> previous_control_frame;
  std::optional<std::int64_t> previous_control_ns;
  std::size_t cursor = 2;
  for (std::size_t index = 0; index + 1U < kHistoryLength;
       ++index, ++cursor) {
    const auto& warmup = records.at(cursor);
    exact_keys(
        warmup,
        {"schema_version", "kind", "event", "control_source_frame_index",
         "control_monotonic_ns", "history_samples", "packet_age_ns",
         "sdk_derivatives_consumed"},
        "causal_warmup_frame");
    common(warmup, "causal_warmup_frame");
    const auto frame = integer(
        warmup.at("control_source_frame_index"),
        "warmup control_source_frame_index");
    const auto control_ns = integer(
        warmup.at("control_monotonic_ns"),
        "warmup control_monotonic_ns");
    const auto packet_age = integer(
        warmup.at("packet_age_ns"), "warmup packet_age_ns");
    if (frame < 0 || control_ns <= 0 ||
        integer(warmup.at("history_samples"), "history_samples") !=
            static_cast<std::int64_t>(index + 1U) ||
        packet_age < -kFutureClockToleranceNs ||
        packet_age > kPolicyFreshnessNs ||
        warmup.at("sdk_derivatives_consumed") != false ||
        (previous_control_frame.has_value() &&
         frame != *previous_control_frame + 1) ||
        (previous_control_ns.has_value() &&
         control_ns != *previous_control_ns + kShadowControlPeriodNs)) {
      reject("causal warmup continuity/freshness contract failed");
    }
    previous_control_frame = frame;
    previous_control_ns = control_ns;
  }

  double recomputed_max_abs = 0.0;
  double recomputed_minimum_margin = std::numeric_limits<double>::infinity();
  double recomputed_maximum_slew = 0.0;
  for (std::int64_t index = 0; index < requested_frames;
       ++index, ++cursor) {
    const auto& frame_record = records.at(cursor);
    exact_keys(
        frame_record,
        {"schema_version", "kind", "event", "action_frame_index",
         "control_source_frame_index", "pico_anchor_source_frame_index",
         "pico_anchor_monotonic_ns", "control_monotonic_ns",
         "received_monotonic_ns", "produced_monotonic_ns", "packet_age_ns",
         "end_to_end_age_ns", "lowstate_age_ns", "inference_ns",
         "native_action", "decoder_output_semantics",
         "external_safe_target_transform_applied", "normalized_max_abs",
         "target_position_min_margin_rad", "target_limit_violations",
         "slew_checked", "target_slew_ratio_max", "target_slew_violations",
         "sdk_derivatives_consumed", "accepted"},
        "action_frame");
    common(frame_record, "action_frame");
    const auto frame = integer(
        frame_record.at("control_source_frame_index"),
        "action control_source_frame_index");
    const auto anchor_frame = integer(
        frame_record.at("pico_anchor_source_frame_index"),
        "pico_anchor_source_frame_index");
    const auto anchor_ns = integer(
        frame_record.at("pico_anchor_monotonic_ns"),
        "pico_anchor_monotonic_ns");
    const auto control_ns = integer(
        frame_record.at("control_monotonic_ns"), "control_monotonic_ns");
    const auto received_ns = integer(
        frame_record.at("received_monotonic_ns"), "received_monotonic_ns");
    const auto produced_ns = integer(
        frame_record.at("produced_monotonic_ns"), "produced_monotonic_ns");
    const auto packet_age = integer(
        frame_record.at("packet_age_ns"), "packet_age_ns");
    const auto end_to_end_age = integer(
        frame_record.at("end_to_end_age_ns"), "end_to_end_age_ns");
    const auto lowstate_age = integer(
        frame_record.at("lowstate_age_ns"), "lowstate_age_ns");
    const auto inference_ns = integer(
        frame_record.at("inference_ns"), "inference_ns");
    if (integer(frame_record.at("action_frame_index"),
                "action_frame_index") != index ||
        !previous_control_frame.has_value() ||
        frame != *previous_control_frame + 1 ||
        !previous_control_ns.has_value() ||
        control_ns != *previous_control_ns + kShadowControlPeriodNs ||
        anchor_frame < 0 || anchor_frame + 1 != frame ||
        anchor_ns <= 0 || anchor_ns + kShadowControlPeriodNs != control_ns ||
        received_ns <= 0 || produced_ns < received_ns ||
        packet_age != received_ns - control_ns ||
        end_to_end_age != produced_ns - control_ns ||
        packet_age < -kFutureClockToleranceNs ||
        packet_age > kPolicyFreshnessNs ||
        end_to_end_age < -kFutureClockToleranceNs ||
        end_to_end_age > kPolicyFreshnessNs ||
        lowstate_age < -kFutureClockToleranceNs ||
        lowstate_age > kStateFreshnessNs ||
        inference_ns < 0 || inference_ns > kShadowInferenceDeadlineNs) {
      reject("action frame continuity/freshness/deadline contract failed");
    }
    previous_control_frame = frame;
    previous_control_ns = control_ns;

    const auto& action = frame_record.at("native_action");
    if (!action.is_array() || action.size() != kDecoderOutputDim) {
      reject("native_action must have exactly 23 entries");
    }
    double action_max_abs = 0.0;
    for (const auto& value : action) {
      action_max_abs = std::max(action_max_abs,
                                std::abs(number(value, "native_action")));
    }
    const auto normalized_max_abs = number(
        frame_record.at("normalized_max_abs"), "normalized_max_abs");
    const auto target_margin = number(
        frame_record.at("target_position_min_margin_rad"),
        "target_position_min_margin_rad");
    const auto slew_ratio = number(
        frame_record.at("target_slew_ratio_max"),
        "target_slew_ratio_max");
    if (std::abs(action_max_abs - normalized_max_abs) > 1e-6 ||
        normalized_max_abs < 0.0 ||
        normalized_max_abs > kMaximumNormalizedAction ||
        target_margin < 0.0 || slew_ratio < 0.0 || slew_ratio > 1.0 ||
        integer(frame_record.at("target_limit_violations"),
                "target_limit_violations") != 0 ||
        frame_record.at("slew_checked") != (index != 0) ||
        integer(frame_record.at("target_slew_violations"),
                "target_slew_violations") != 0 ||
        frame_record.at("decoder_output_semantics") !=
            live::kAppliedSafeNativeActionSemantics ||
        frame_record.at("external_safe_target_transform_applied") != false ||
        frame_record.at("sdk_derivatives_consumed") != false ||
        frame_record.at("accepted") != true) {
      reject("action frame numeric/safety contract failed");
    }
    recomputed_max_abs = std::max(recomputed_max_abs, normalized_max_abs);
    recomputed_minimum_margin =
        std::min(recomputed_minimum_margin, target_margin);
    recomputed_maximum_slew =
        std::max(recomputed_maximum_slew, slew_ratio);
  }

  const auto& complete = records.at(cursor);
  exact_keys(
      complete,
      {"schema_version", "kind", "event", "passed", "action_frames",
       "causal_warmup_frames", "maximum_normalized_abs",
       "minimum_target_position_margin_rad", "maximum_target_slew_ratio",
       "crc_rejects", "robot_mutation_authorized"},
      "session_complete");
  common(complete, "session_complete");
  const auto summary_max = number(
      complete.at("maximum_normalized_abs"), "maximum_normalized_abs");
  const auto summary_margin = number(
      complete.at("minimum_target_position_margin_rad"),
      "minimum_target_position_margin_rad");
  const auto summary_slew = number(
      complete.at("maximum_target_slew_ratio"),
      "maximum_target_slew_ratio");
  if (complete.at("passed") != true ||
      integer(complete.at("action_frames"), "action_frames") !=
          requested_frames ||
      integer(complete.at("causal_warmup_frames"),
              "causal_warmup_frames") !=
          static_cast<std::int64_t>(kHistoryLength - 1U) ||
      integer(complete.at("crc_rejects"), "completion crc_rejects") != 0 ||
      complete.at("robot_mutation_authorized") != false ||
      std::abs(summary_max - recomputed_max_abs) > 1e-6 ||
      std::abs(summary_margin - recomputed_minimum_margin) > 1e-6 ||
      std::abs(summary_slew - recomputed_maximum_slew) > 1e-6 ||
      summary_max > kMaximumNormalizedAction || summary_margin < 0.0 ||
      summary_slew > 1.0) {
    reject("session_complete did not exactly summarize a passing session");
  }
  return {
      .action_frames = static_cast<int>(requested_frames),
      .maximum_normalized_abs = summary_max,
      .minimum_target_position_margin_rad = summary_margin,
      .maximum_target_slew_ratio = summary_slew,
  };
}

// URDF limits, compact hardware order.  These are absolute reject limits,
// not clipping envelopes.
inline constexpr std::array<double, 23> kHardwareEffortLimitNm = {
    88.0, 139.0, 88.0, 139.0, 35.0, 35.0,
    88.0, 139.0, 88.0, 139.0, 35.0, 35.0,
    88.0,
    25.0, 25.0, 25.0, 25.0, 25.0,
    25.0, 25.0, 25.0, 25.0, 25.0,
};

// Deliberately low first-actuation gains.  Higher gains require a different
// reviewed promotion schema rather than a CLI override.
inline constexpr std::array<double, 23> kStageOneKp = {
    12.0, 12.0, 12.0, 16.0, 8.0, 8.0,
    12.0, 12.0, 12.0, 16.0, 8.0, 8.0,
    8.0,
    4.0, 4.0, 4.0, 4.0, 3.0,
    4.0, 4.0, 4.0, 4.0, 3.0,
};
inline constexpr std::array<double, 23> kStageOneKd = {
    1.0, 1.0, 1.0, 1.25, 0.8, 0.8,
    1.0, 1.0, 1.0, 1.25, 0.8, 0.8,
    0.8,
    0.45, 0.45, 0.45, 0.45, 0.35,
    0.45, 0.45, 0.45, 0.45, 0.35,
};
inline constexpr double kFailSafeKd = 1.5;

enum class ArtifactStage {
  Diagnostic,
  MujocoPromotion,
};

struct ActiveArtifactBinding {
  bool base_promotion_valid = false;
  ArtifactStage stage = ArtifactStage::Diagnostic;
  int decoder_output_dim = 0;
  int mode_machine = 0;
  double action_clip_value = 0.0;
  bool deployment_ready = false;
  bool active_motor_control_authorized = false;
  bool gantry_authorized = false;
  bool free_standing_authorized = false;
  std::string decoder_output_semantics;
  std::string previous_action_semantics;
  bool external_safe_target_transform_allowed = true;
  std::string safe_target_transform_sha256;
  std::string source_promotion_sha256;
  std::string checkpoint_sha256;
  std::string lineage_sha256;
  std::string policy_state_sha256;
  std::string encoder_onnx_sha256;
  std::string decoder_onnx_sha256;
  std::string metadata_sha256;
  std::string full_campaign_aggregate_sha256;
  std::string full_campaign_shard_manifest_sha256;
  std::string live_shadow_evidence_sha256;
  std::string authorization_id;
};

// Separate binding for the pinned public native124 actor.  It deliberately
// cannot impersonate the causal encoder/decoder promotion schema.
struct Native124ActiveArtifactBinding {
  std::string policy_sha256;
  int observation_dim = 0;
  int action_dim = 0;
  int mode_machine = 0;
  bool onnx_signature_valid = false;
  bool dry_run_finite = false;
  bool gantry_authorized = false;
  bool free_standing_authorized = false;
  bool external_target_envelope_required = false;
  double stage_one_action_fraction = 0.0;
  double maximum_target_rate_rad_per_second = 0.0;
};

inline std::vector<std::string> ValidateNative124ActiveArtifactBinding(
    const Native124ActiveArtifactBinding& binding) {
  std::vector<std::string> errors;
  if (binding.policy_sha256 != kSelectedNative124PolicySha256) {
    errors.emplace_back("native124 policy SHA-256 is not the selected gantry artifact");
  }
  if (binding.observation_dim != 124 || binding.action_dim != 23 ||
      !binding.onnx_signature_valid || !binding.dry_run_finite) {
    errors.emplace_back("native124 ONNX ABI/dry-run contract failed");
  }
  if (binding.mode_machine != kRequiredModeMachine) {
    errors.emplace_back("native124 gantry requires mode_machine==4");
  }
  if (!binding.gantry_authorized || binding.free_standing_authorized) {
    errors.emplace_back("native124 artifact is gantry-only");
  }
  if (!binding.external_target_envelope_required ||
      binding.stage_one_action_fraction != kStageOneActionFraction ||
      binding.maximum_target_rate_rad_per_second !=
          kStageOneTargetRateRadPerSecond) {
    errors.emplace_back("native124 stage-one target envelope mismatch");
  }
  return errors;
}

struct Native124AcquisitionReference {
  std::array<float, 23> position_native{};
  std::array<float, 23> velocity_native{};
  double alpha = 0.0;
  bool ready_for_arm = false;
};

class Native124AcquisitionGate {
 public:
  explicit Native124AcquisitionGate(std::array<float, 23> measured_start_native)
      : start_(measured_start_native) {
    if (!live::IsFinite(start_)) {
      throw std::invalid_argument("native124 acquisition start is non-finite");
    }
  }

  Native124AcquisitionReference Advance(
      const std::array<float, 23>& desired_position_native,
      const std::array<float, 23>& desired_velocity_native) {
    if (!live::IsFinite(desired_position_native) ||
        !live::IsFinite(desired_velocity_native)) {
      throw std::invalid_argument("native124 acquisition reference is non-finite");
    }
    Native124AcquisitionReference result;
    if (frame_ < kNative124WarmStartFrames) {
      result.position_native = start_;
    } else {
      const auto ramp_frame = frame_ - kNative124WarmStartFrames + 1;
      const double u = std::min(
          static_cast<double>(ramp_frame) /
              static_cast<double>(kNative124ReferenceRampFrames),
          1.0);
      result.alpha = u * u * (3.0 - 2.0 * u);
      const double alpha_rate = u < 1.0
          ? 6.0 * u * (1.0 - u) /
                (static_cast<double>(kNative124ReferenceRampFrames) * 0.02)
          : 0.0;
      for (std::size_t index = 0; index < 23; ++index) {
        const double displacement =
            static_cast<double>(desired_position_native[index] - start_[index]);
        result.position_native[index] = static_cast<float>(
            start_[index] + result.alpha * displacement);
        result.velocity_native[index] = static_cast<float>(
            result.alpha * desired_velocity_native[index] +
            alpha_rate * displacement);
      }
    }
    ++frame_;
    result.ready_for_arm =
        frame_ >= kNative124WarmStartFrames + kNative124ReferenceRampFrames;
    return result;
  }

  [[nodiscard]] int frame() const { return frame_; }

 private:
  std::array<float, 23> start_{};
  int frame_ = 0;
};

inline bool IsLowerSha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

inline std::vector<std::string> ValidateActiveArtifactBinding(
    const ActiveArtifactBinding& binding) {
  std::vector<std::string> errors;
  if (!binding.base_promotion_valid) {
    errors.emplace_back("base MuJoCo promotion validation did not pass");
  }
  if (binding.stage != ArtifactStage::MujocoPromotion) {
    errors.emplace_back("diagnostic ONNX pair cannot authorize motor control");
  }
  if (binding.decoder_output_dim != kDecoderOutputDim) {
    errors.emplace_back("decoder must have exactly 23 native outputs");
  }
  if (binding.mode_machine != kRequiredModeMachine) {
    errors.emplace_back("promotion requires mode_machine==4");
  }
  if (!std::isfinite(binding.action_clip_value) ||
      binding.action_clip_value != kTrainingActionClipValue) {
    errors.emplace_back("active promotion action_clip_value must be exactly 20");
  }
  if (!binding.deployment_ready) {
    errors.emplace_back("active promotion deployment_ready is not true");
  }
  if (!binding.active_motor_control_authorized) {
    errors.emplace_back(
        "active promotion does not authorize motor control");
  }
  if (!binding.gantry_authorized) {
    errors.emplace_back("active promotion does not authorize gantry test");
  }
  if (binding.free_standing_authorized) {
    errors.emplace_back("stage-one promotion must forbid free standing");
  }
  if (binding.decoder_output_semantics !=
          live::kAppliedSafeNativeActionSemantics ||
      binding.previous_action_semantics !=
          live::kAppliedSafeNativeActionSemantics) {
    errors.emplace_back(
        "active promotion must consume applied safe native action/history");
  }
  if (binding.external_safe_target_transform_allowed) {
    errors.emplace_back(
        "active promotion must forbid a second safe-target transform");
  }
  if (binding.safe_target_transform_sha256 !=
      live::kSafeTargetTransformSha256) {
    errors.emplace_back("active promotion safe-target transform mismatch");
  }
  for (const auto& [value, label] :
       std::array<std::pair<std::string_view, std::string_view>, 11>{
           std::pair<std::string_view, std::string_view>{
               binding.source_promotion_sha256, "source promotion"},
           {binding.checkpoint_sha256, "checkpoint"},
           {binding.lineage_sha256, "lineage"},
           {binding.policy_state_sha256, "policy state"},
           {binding.safe_target_transform_sha256, "safe-target transform"},
           {binding.encoder_onnx_sha256, "encoder ONNX"},
           {binding.decoder_onnx_sha256, "decoder ONNX"},
           {binding.metadata_sha256, "candidate metadata"},
           {binding.full_campaign_aggregate_sha256, "campaign aggregate"},
           {binding.full_campaign_shard_manifest_sha256, "shard manifest"},
           {binding.live_shadow_evidence_sha256, "live-shadow evidence"},
       }) {
    if (!IsLowerSha256(value)) {
      errors.emplace_back(std::string(label) + " SHA-256 is invalid");
    }
  }
  if (!IsSafeAuthorizationId(binding.authorization_id)) {
    errors.emplace_back("active promotion authorization_id is unsafe");
  }
  return errors;
}

inline ActiveArtifactBinding ParseActivePromotion(
    const nlohmann::json& document,
    bool base_promotion_valid,
    std::string_view expected_source_promotion_sha256,
    std::string_view expected_encoder_sha256,
    std::string_view expected_decoder_sha256,
    std::string_view expected_metadata_sha256,
    std::string_view expected_live_shadow_evidence_sha256) {
  constexpr std::array<std::string_view, 25> kExactKeys = {
      "schema_version", "kind", "robot_model", "decoder_output_dim",
      "mode_machine", "action_clip_value", "deployment_ready",
      "active_motor_control_authorized", "gantry_authorized",
      "free_standing_authorized", "decoder_output_semantics",
      "previous_action_semantics",
      "external_safe_target_transform_allowed",
      "safe_target_transform_sha256", "source_promotion_sha256",
      "checkpoint_sha256", "lineage_sha256", "policy_state_sha256",
      "encoder_onnx_sha256", "decoder_onnx_sha256",
      "metadata_sha256", "full_campaign_aggregate_sha256",
      "full_campaign_shard_manifest_sha256",
      "live_shadow_evidence_sha256", "authorization_id",
  };
  if (!document.is_object() || document.size() != kExactKeys.size() ||
      !std::all_of(kExactKeys.begin(), kExactKeys.end(),
                   [&](std::string_view key) {
                     return document.contains(std::string(key));
                   })) {
    throw std::invalid_argument(
        "active promotion keys do not match exact schema");
  }
  ActiveArtifactBinding result;
  result.base_promotion_valid = base_promotion_valid;
  result.stage = ArtifactStage::MujocoPromotion;
  const auto require = [&](std::string_view key) -> const nlohmann::json& {
    const auto iterator = document.find(std::string(key));
    if (iterator == document.end()) {
      throw std::invalid_argument(
          "active promotion missing field: " + std::string(key));
    }
    return *iterator;
  };
  if (require("schema_version") != kActivePromotionSchemaVersion ||
      require("kind") != kActivePromotionKind ||
      require("robot_model") != kRobotModel) {
    throw std::invalid_argument("active promotion identity mismatch");
  }
  result.decoder_output_dim = require("decoder_output_dim").get<int>();
  result.mode_machine = require("mode_machine").get<int>();
  result.action_clip_value = require("action_clip_value").get<double>();
  result.deployment_ready = require("deployment_ready").get<bool>();
  result.active_motor_control_authorized =
      require("active_motor_control_authorized").get<bool>();
  result.gantry_authorized = require("gantry_authorized").get<bool>();
  result.free_standing_authorized =
      require("free_standing_authorized").get<bool>();
  result.decoder_output_semantics =
      require("decoder_output_semantics").get<std::string>();
  result.previous_action_semantics =
      require("previous_action_semantics").get<std::string>();
  result.external_safe_target_transform_allowed =
      require("external_safe_target_transform_allowed").get<bool>();
  result.safe_target_transform_sha256 =
      require("safe_target_transform_sha256").get<std::string>();
  result.source_promotion_sha256 =
      require("source_promotion_sha256").get<std::string>();
  result.checkpoint_sha256 =
      require("checkpoint_sha256").get<std::string>();
  result.lineage_sha256 =
      require("lineage_sha256").get<std::string>();
  result.policy_state_sha256 =
      require("policy_state_sha256").get<std::string>();
  result.encoder_onnx_sha256 =
      require("encoder_onnx_sha256").get<std::string>();
  result.decoder_onnx_sha256 =
      require("decoder_onnx_sha256").get<std::string>();
  result.metadata_sha256 =
      require("metadata_sha256").get<std::string>();
  result.full_campaign_aggregate_sha256 =
      require("full_campaign_aggregate_sha256").get<std::string>();
  result.full_campaign_shard_manifest_sha256 =
      require("full_campaign_shard_manifest_sha256").get<std::string>();
  result.live_shadow_evidence_sha256 =
      require("live_shadow_evidence_sha256").get<std::string>();
  result.authorization_id = require("authorization_id").get<std::string>();
  if (result.source_promotion_sha256 !=
          expected_source_promotion_sha256 ||
      result.encoder_onnx_sha256 != expected_encoder_sha256 ||
      result.decoder_onnx_sha256 != expected_decoder_sha256 ||
      result.metadata_sha256 != expected_metadata_sha256 ||
      result.live_shadow_evidence_sha256 !=
          expected_live_shadow_evidence_sha256) {
    throw std::invalid_argument(
        "active promotion hash binding does not match loaded files");
  }
  return result;
}

enum class Fault {
  None,
  ArtifactRejected,
  CrcFailure,
  WrongMode,
  TickRegression,
  StateStale,
  StateNonFinite,
  JointPositionLimit,
  JointVelocityLimit,
  JointEffortLimit,
  PolicyStale,
  PicoTermsInvalid,
  PicoFrameRegression,
  PolicyNonFinite,
  PolicyMagnitude,
  PredictedEffortLimit,
  DeadmanReleased,
  OperatorStop,
  InternalInvariant,
};

inline std::string_view FaultName(Fault fault) {
  switch (fault) {
    case Fault::None: return "none";
    case Fault::ArtifactRejected: return "artifact_rejected";
    case Fault::CrcFailure: return "crc_failure";
    case Fault::WrongMode: return "wrong_mode";
    case Fault::TickRegression: return "tick_regression";
    case Fault::StateStale: return "state_stale";
    case Fault::StateNonFinite: return "state_non_finite";
    case Fault::JointPositionLimit: return "joint_position_limit";
    case Fault::JointVelocityLimit: return "joint_velocity_limit";
    case Fault::JointEffortLimit: return "joint_effort_limit";
    case Fault::PolicyStale: return "policy_stale";
    case Fault::PicoTermsInvalid: return "pico_terms_invalid";
    case Fault::PicoFrameRegression: return "pico_frame_regression";
    case Fault::PolicyNonFinite: return "policy_non_finite";
    case Fault::PolicyMagnitude: return "policy_magnitude";
    case Fault::PredictedEffortLimit: return "predicted_effort_limit";
    case Fault::DeadmanReleased: return "deadman_released";
    case Fault::OperatorStop: return "operator_stop";
    case Fault::InternalInvariant: return "internal_invariant";
  }
  return "unknown";
}

struct StateSample {
  std::uint32_t tick = 0;
  std::uint8_t mode_machine = 0;
  bool crc_valid = false;
  std::int64_t received_monotonic_ns = 0;
  std::array<double, kMotorSlotCount> q{};
  std::array<double, kMotorSlotCount> dq{};
  std::array<double, kMotorSlotCount> tau_est{};
};

struct OperatorInput {
  bool arm_edge = false;
  bool deadman_held = false;
  bool stop_requested = false;
};

// Unitree wireless_remote is a fixed 40-byte little-endian packet.  Buttons
// occupy bytes 2..3.  Stage one uses A rising edge to arm, L2 as continuously
// held deadman, and either B or R2 as software STOP.
struct WirelessOperatorState {
  bool arm_pressed = false;
  bool deadman_held = false;
  bool stop_pressed = false;
};

inline WirelessOperatorState DecodeWirelessOperator(
    std::span<const std::uint8_t, 40> packet) {
  const std::uint16_t buttons =
      static_cast<std::uint16_t>(packet[2]) |
      (static_cast<std::uint16_t>(packet[3]) << 8U);
  return {
      .arm_pressed = (buttons & (1U << 8U)) != 0,
      .deadman_held = (buttons & (1U << 5U)) != 0,
      .stop_pressed =
          (buttons & ((1U << 9U) | (1U << 4U))) != 0,
  };
}

struct PolicySample {
  std::array<float, 23> native_action{};
  std::int64_t produced_monotonic_ns = 0;
};

struct MotorSlotCommand {
  std::uint8_t mode = 0;
  double q = 0.0;
  double dq = 0.0;
  double kp = 0.0;
  double kd = 0.0;
  double tau = 0.0;
};

using MotorCommand = std::array<MotorSlotCommand, kMotorSlotCount>;

class RealProprioWarmupGate {
 public:
  bool Observe(std::uint64_t source_frame_index) {
    if (rejected_) {
      return false;
    }
    if (have_frame_ && source_frame_index != last_frame_ + 1U) {
      rejected_ = true;
      return false;
    }
    have_frame_ = true;
    last_frame_ = source_frame_index;
    if (sample_count_ < kHistoryLength) {
      ++sample_count_;
    }
    return ready();
  }
  [[nodiscard]] bool ready() const {
    return !rejected_ && sample_count_ == kHistoryLength;
  }
  [[nodiscard]] bool rejected() const { return rejected_; }
  [[nodiscard]] int sample_count() const { return sample_count_; }

 private:
  std::uint64_t last_frame_ = 0;
  int sample_count_ = 0;
  bool have_frame_ = false;
  bool rejected_ = false;
};

class GantrySafetyCore {
 public:
  explicit GantrySafetyCore(ActiveArtifactBinding artifact)
      : artifact_(std::move(artifact)) {
    if (!ValidateActiveArtifactBinding(artifact_).empty()) {
      Latch(Fault::ArtifactRejected);
    }
  }

  explicit GantrySafetyCore(Native124ActiveArtifactBinding artifact)
      : native124_artifact_(std::move(artifact)) {
    if (!ValidateNative124ActiveArtifactBinding(native124_artifact_).empty()) {
      Latch(Fault::ArtifactRejected);
    }
  }

  void ObserveCrcFailure() { Latch(Fault::CrcFailure); }

  void ObservePicoTermsFailure() { Latch(Fault::PicoTermsInvalid); }

  void ObservePicoFrameRegression() {
    Latch(Fault::PicoFrameRegression);
  }

  void ObserveInternalFailure() { Latch(Fault::InternalInvariant); }

  void ObserveState(const StateSample& sample,
                    std::int64_t now_monotonic_ns) {
    if (fault_ != Fault::None) {
      return;
    }
    if (!sample.crc_valid) {
      Latch(Fault::CrcFailure);
      return;
    }
    if (!Fresh(sample.received_monotonic_ns, now_monotonic_ns,
               kStateFreshnessNs)) {
      Latch(Fault::StateStale);
      return;
    }
    if (sample.mode_machine != kRequiredModeMachine) {
      if (seen_mode_four_) {
        Latch(Fault::WrongMode);
      }
      stable_samples_ = 0;
      return;
    }
    seen_mode_four_ = true;
    if (have_tick_) {
      const auto delta = static_cast<std::int32_t>(sample.tick - last_tick_);
      if (delta < 0) {
        Latch(Fault::TickRegression);
        return;
      }
      if (delta == 0) {
        return;
      }
    }
    if (!ValidateTelemetry(sample)) {
      return;
    }
    state_ = sample;
    have_tick_ = true;
    last_tick_ = sample.tick;
    last_advancing_state_ns_ = sample.received_monotonic_ns;
    if (stable_samples_ < kStableModeSamples) {
      ++stable_samples_;
    }
    mutation_surface_allowed_ =
        stable_samples_ == kStableModeSamples;
  }

  void ObserveOperator(const OperatorInput& input,
                       std::int64_t now_monotonic_ns) {
    if (input.stop_requested) {
      Latch(Fault::OperatorStop);
      return;
    }
    if (fault_ != Fault::None) {
      return;
    }
    if (armed_ && !input.deadman_held) {
      Latch(Fault::DeadmanReleased);
      return;
    }
    deadman_held_ = input.deadman_held;
    if (input.arm_edge) {
      CheckWatchdogs(now_monotonic_ns);
      if (fault_ != Fault::None || !mutation_surface_allowed_ ||
          !input.deadman_held || !policy_.has_value()) {
        return;
      }
      if (!Fresh(policy_->produced_monotonic_ns, now_monotonic_ns,
                 kPolicyFreshnessNs)) {
        Latch(Fault::PolicyStale);
        return;
      }
      armed_ = true;
    }
  }

  void SubmitPolicy(const PolicySample& sample,
                    std::int64_t now_monotonic_ns) {
    if (fault_ != Fault::None) {
      return;
    }
    if (!Fresh(sample.produced_monotonic_ns, now_monotonic_ns,
               kPolicyFreshnessNs)) {
      Latch(Fault::PolicyStale);
      return;
    }
    if (!live::IsFinite(sample.native_action)) {
      Latch(Fault::PolicyNonFinite);
      return;
    }
    for (const float value : sample.native_action) {
      if (std::abs(static_cast<double>(value)) >
          kMaximumNormalizedAction) {
        Latch(Fault::PolicyMagnitude);
        return;
      }
    }
    policy_ = sample;
  }

  void CheckWatchdogs(std::int64_t now_monotonic_ns) {
    if (fault_ != Fault::None) {
      return;
    }
    if (have_tick_ &&
        !Fresh(last_advancing_state_ns_, now_monotonic_ns,
               kStateFreshnessNs)) {
      Latch(Fault::StateStale);
      return;
    }
    if (armed_ &&
        (!policy_.has_value() ||
         !Fresh(policy_->produced_monotonic_ns, now_monotonic_ns,
                kPolicyFreshnessNs))) {
      Latch(Fault::PolicyStale);
    }
  }

  [[nodiscard]] MotorCommand BuildCommand(
      std::int64_t now_monotonic_ns,
      double dt_seconds = kControlPeriodSeconds) {
    CheckWatchdogs(now_monotonic_ns);
    if (fault_ != Fault::None || !armed_ || !deadman_held_ ||
        !state_.has_value() || !policy_.has_value()) {
      return BuildDampingCommand();
    }
    if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0 ||
        dt_seconds > 0.02) {
      Latch(Fault::InternalInvariant);
      return BuildDampingCommand();
    }

    const auto hardware_action =
        NativeToHardwareCompact(policy_->native_action);
    if (!have_last_target_) {
      for (std::size_t compact = 0; compact < 23; ++compact) {
        const auto slot =
            static_cast<std::size_t>(kHardwareJointIds[compact]);
        last_target_[compact] = state_->q[slot];
      }
      have_last_target_ = true;
    }

    MotorCommand command = BuildDampingCommand();
    const double max_slew =
        kStageOneTargetRateRadPerSecond * dt_seconds;
    for (std::size_t compact = 0; compact < 23; ++compact) {
      const double raw_target =
          live::kHardwareDefaultQ[compact] +
          static_cast<double>(hardware_action[compact]) *
              kHardwareActionScale[compact] *
              kStageOneActionFraction;
      if (!std::isfinite(raw_target) ||
          raw_target < live::kHardwareLowerLimit[compact] +
                           kTargetLimitMarginRad ||
          raw_target > live::kHardwareUpperLimit[compact] -
                           kTargetLimitMarginRad) {
        Latch(Fault::JointPositionLimit);
        return BuildDampingCommand();
      }
      const double target = std::clamp(
          raw_target,
          last_target_[compact] - max_slew,
          last_target_[compact] + max_slew);
      const auto slot =
          static_cast<std::size_t>(kHardwareJointIds[compact]);
      const double predicted_effort =
          kStageOneKp[compact] * (target - state_->q[slot]) -
          kStageOneKd[compact] * state_->dq[slot];
      if (!std::isfinite(predicted_effort) ||
          std::abs(predicted_effort) >
              0.25 * kHardwareEffortLimitNm[compact]) {
        Latch(Fault::PredictedEffortLimit);
        return BuildDampingCommand();
      }
      command[slot] = MotorSlotCommand{
          .mode = 1,
          .q = target,
          .dq = 0.0,
          .kp = kStageOneKp[compact],
          .kd = kStageOneKd[compact],
          .tau = 0.0,
      };
      last_target_[compact] = target;
    }
    // Excluded slots stay mode=0.  No policy-derived field reaches them.
    for (const int slot : kExcludedHardwareJointIds) {
      const auto index = static_cast<std::size_t>(slot);
      command[index].mode = 0;
      command[index].q = state_->q[index];
      command[index].dq = 0.0;
      command[index].kp = 0.0;
      command[index].kd = kFailSafeKd;
      command[index].tau = 0.0;
    }
    return command;
  }

  [[nodiscard]] MotorCommand BuildDampingCommand() const {
    MotorCommand command{};
    for (std::size_t slot = 0; slot < command.size(); ++slot) {
      command[slot] = MotorSlotCommand{
          .mode = 1,
          .q = state_.has_value() ? state_->q[slot] : 0.0,
          .dq = 0.0,
          .kp = 0.0,
          .kd = kFailSafeKd,
          .tau = 0.0,
      };
    }
    for (const int slot : kExcludedHardwareJointIds) {
      command[static_cast<std::size_t>(slot)].mode = 0;
    }
    return command;
  }

  void Stop() { Latch(Fault::OperatorStop); }

  [[nodiscard]] bool mutation_surface_allowed() const {
    return mutation_surface_allowed_ && fault_ == Fault::None;
  }
  [[nodiscard]] bool armed() const {
    return armed_ && fault_ == Fault::None;
  }
  [[nodiscard]] bool policy_ready_for_arm(
      std::int64_t now_monotonic_ns) const {
    return fault_ == Fault::None && policy_.has_value() &&
           Fresh(policy_->produced_monotonic_ns, now_monotonic_ns,
                 kPolicyFreshnessNs);
  }
  [[nodiscard]] bool stopped() const { return fault_ != Fault::None; }
  [[nodiscard]] Fault fault() const { return fault_; }
  [[nodiscard]] int stable_samples() const { return stable_samples_; }
  [[nodiscard]] std::optional<StateSample> latest_state() const {
    return state_;
  }

 private:
  static bool Fresh(std::int64_t produced_ns, std::int64_t now_ns,
                    std::int64_t maximum_age_ns) {
    if (produced_ns <= 0 || now_ns <= 0) {
      return false;
    }
    const auto age = now_ns - produced_ns;
    return age >= -kFutureClockToleranceNs && age <= maximum_age_ns;
  }

  bool ValidateTelemetry(const StateSample& sample) {
    const auto finite = [](const auto& values) {
      return std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
      });
    };
    if (!finite(sample.q) || !finite(sample.dq) ||
        !finite(sample.tau_est)) {
      Latch(Fault::StateNonFinite);
      return false;
    }
    for (std::size_t compact = 0; compact < 23; ++compact) {
      const auto slot =
          static_cast<std::size_t>(kHardwareJointIds[compact]);
      if (sample.q[slot] < live::kHardwareLowerLimit[compact] ||
          sample.q[slot] > live::kHardwareUpperLimit[compact]) {
        Latch(Fault::JointPositionLimit);
        return false;
      }
      if (std::abs(sample.dq[slot]) >
          live::kHardwareVelocityLimit[compact]) {
        Latch(Fault::JointVelocityLimit);
        return false;
      }
      if (std::abs(sample.tau_est[slot]) >
          kHardwareEffortLimitNm[compact]) {
        Latch(Fault::JointEffortLimit);
        return false;
      }
    }
    return true;
  }

  void Latch(Fault fault) {
    if (fault_ == Fault::None) {
      fault_ = fault;
    }
    armed_ = false;
    deadman_held_ = false;
    mutation_surface_allowed_ = false;
  }

  ActiveArtifactBinding artifact_;
  Native124ActiveArtifactBinding native124_artifact_;
  Fault fault_ = Fault::None;
  std::optional<StateSample> state_;
  std::optional<PolicySample> policy_;
  std::array<double, 23> last_target_{};
  std::uint32_t last_tick_ = 0;
  std::int64_t last_advancing_state_ns_ = 0;
  int stable_samples_ = 0;
  bool seen_mode_four_ = false;
  bool have_tick_ = false;
  bool mutation_surface_allowed_ = false;
  bool deadman_held_ = false;
  bool armed_ = false;
  bool have_last_target_ = false;
};

// Separate non-policy first-contact path.  It can only hold joint positions
// sampled at the explicit A+L2 arm edge.  No ONNX output or PICO field enters
// this type, so a failed/no-go policy cannot be smuggled into motor commands.
class HoldSmokeSafetyCore {
 public:
  void ObserveCrcFailure() { Latch(Fault::CrcFailure); }
  void ObserveInternalFailure() { Latch(Fault::InternalInvariant); }

  void ObserveState(const StateSample& sample,
                    std::int64_t now_monotonic_ns) {
    if (fault_ != Fault::None) {
      return;
    }
    if (!sample.crc_valid) {
      Latch(Fault::CrcFailure);
      return;
    }
    if (!Fresh(sample.received_monotonic_ns, now_monotonic_ns)) {
      Latch(Fault::StateStale);
      return;
    }
    if (sample.mode_machine != kRequiredModeMachine) {
      if (seen_mode_four_) {
        Latch(Fault::WrongMode);
      }
      stable_samples_ = 0;
      return;
    }
    seen_mode_four_ = true;
    if (have_tick_) {
      const auto delta = static_cast<std::int32_t>(sample.tick - last_tick_);
      if (delta < 0) {
        Latch(Fault::TickRegression);
        return;
      }
      if (delta == 0) {
        return;
      }
    }
    const auto telemetry_fault = ValidateTelemetry(sample);
    if (telemetry_fault != Fault::None) {
      Latch(telemetry_fault);
      return;
    }
    state_ = sample;
    have_tick_ = true;
    last_tick_ = sample.tick;
    last_advancing_state_ns_ = sample.received_monotonic_ns;
    if (stable_samples_ < kStableModeSamples) {
      ++stable_samples_;
    }
    mutation_surface_allowed_ = stable_samples_ == kStableModeSamples;
  }

  // Runtime calls this only after successful motion release and publisher
  // construction.  It is intentionally one-way.
  void EnableOperatorArming() {
    if (mutation_surface_allowed_ && fault_ == Fault::None) {
      operator_arming_enabled_ = true;
    }
  }

  void ObserveOperator(const OperatorInput& input,
                       std::int64_t now_monotonic_ns) {
    if (input.stop_requested) {
      Latch(Fault::OperatorStop);
      return;
    }
    if (fault_ != Fault::None) {
      return;
    }
    if (armed_ && !input.deadman_held) {
      Latch(Fault::DeadmanReleased);
      return;
    }
    deadman_held_ = input.deadman_held;
    if (!input.arm_edge || !operator_arming_enabled_ ||
        !input.deadman_held || !state_.has_value()) {
      return;
    }
    CheckWatchdog(now_monotonic_ns);
    if (fault_ != Fault::None) {
      return;
    }
    for (std::size_t compact = 0; compact < 23; ++compact) {
      const auto slot =
          static_cast<std::size_t>(kHardwareJointIds[compact]);
      hold_target_[compact] = state_->q[slot];
    }
    armed_ = true;
  }

  void CheckWatchdog(std::int64_t now_monotonic_ns) {
    if (fault_ == Fault::None && have_tick_ &&
        !Fresh(last_advancing_state_ns_, now_monotonic_ns)) {
      Latch(Fault::StateStale);
    }
  }

  [[nodiscard]] MotorCommand BuildCommand(
      std::int64_t now_monotonic_ns) {
    CheckWatchdog(now_monotonic_ns);
    if (fault_ != Fault::None || !armed_ || !deadman_held_ ||
        !state_.has_value()) {
      return BuildDampingCommand();
    }
    MotorCommand command = BuildDampingCommand();
    for (std::size_t compact = 0; compact < 23; ++compact) {
      const auto slot =
          static_cast<std::size_t>(kHardwareJointIds[compact]);
      const double kp = 0.25 * kStageOneKp[compact];
      const double kd = kStageOneKd[compact];
      const double predicted_effort =
          kp * (hold_target_[compact] - state_->q[slot]) -
          kd * state_->dq[slot];
      if (!std::isfinite(predicted_effort) ||
          std::abs(predicted_effort) >
              0.10 * kHardwareEffortLimitNm[compact]) {
        Latch(Fault::PredictedEffortLimit);
        return BuildDampingCommand();
      }
      command[slot] = {
          .mode = 1,
          .q = hold_target_[compact],
          .dq = 0.0,
          .kp = kp,
          .kd = kd,
          .tau = 0.0,
      };
    }
    for (const int slot : kExcludedHardwareJointIds) {
      command[static_cast<std::size_t>(slot)].mode = 0;
    }
    return command;
  }

  [[nodiscard]] MotorCommand BuildDampingCommand() const {
    MotorCommand command{};
    for (std::size_t slot = 0; slot < command.size(); ++slot) {
      command[slot] = {
          .mode = 1,
          .q = state_.has_value() ? state_->q[slot] : 0.0,
          .dq = 0.0,
          .kp = 0.0,
          .kd = kFailSafeKd,
          .tau = 0.0,
      };
    }
    for (const int slot : kExcludedHardwareJointIds) {
      command[static_cast<std::size_t>(slot)].mode = 0;
    }
    return command;
  }

  void Stop() { Latch(Fault::OperatorStop); }
  [[nodiscard]] bool mutation_surface_allowed() const {
    return mutation_surface_allowed_ && fault_ == Fault::None;
  }
  [[nodiscard]] bool operator_arming_enabled() const {
    return operator_arming_enabled_ && fault_ == Fault::None;
  }
  [[nodiscard]] bool armed() const {
    return armed_ && fault_ == Fault::None;
  }
  [[nodiscard]] bool stopped() const { return fault_ != Fault::None; }
  [[nodiscard]] Fault fault() const { return fault_; }
  [[nodiscard]] int stable_samples() const { return stable_samples_; }

 private:
  static bool Fresh(std::int64_t produced_ns, std::int64_t now_ns) {
    if (produced_ns <= 0 || now_ns <= 0) {
      return false;
    }
    const auto age = now_ns - produced_ns;
    return age >= -kFutureClockToleranceNs && age <= kStateFreshnessNs;
  }

  static Fault ValidateTelemetry(const StateSample& sample) {
    const auto finite = [](const auto& values) {
      return std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
      });
    };
    if (!finite(sample.q) || !finite(sample.dq) ||
        !finite(sample.tau_est)) {
      return Fault::StateNonFinite;
    }
    for (std::size_t compact = 0; compact < 23; ++compact) {
      const auto slot =
          static_cast<std::size_t>(kHardwareJointIds[compact]);
      if (sample.q[slot] < live::kHardwareLowerLimit[compact] ||
          sample.q[slot] > live::kHardwareUpperLimit[compact]) {
        return Fault::JointPositionLimit;
      }
      if (std::abs(sample.dq[slot]) >
          live::kHardwareVelocityLimit[compact]) {
        return Fault::JointVelocityLimit;
      }
      if (std::abs(sample.tau_est[slot]) >
          kHardwareEffortLimitNm[compact]) {
        return Fault::JointEffortLimit;
      }
    }
    return Fault::None;
  }

  void Latch(Fault fault) {
    if (fault_ == Fault::None) {
      fault_ = fault;
    }
    armed_ = false;
    deadman_held_ = false;
    mutation_surface_allowed_ = false;
    operator_arming_enabled_ = false;
  }

  Fault fault_ = Fault::None;
  std::optional<StateSample> state_;
  std::array<double, 23> hold_target_{};
  std::uint32_t last_tick_ = 0;
  std::int64_t last_advancing_state_ns_ = 0;
  int stable_samples_ = 0;
  bool seen_mode_four_ = false;
  bool have_tick_ = false;
  bool mutation_surface_allowed_ = false;
  bool operator_arming_enabled_ = false;
  bool deadman_held_ = false;
  bool armed_ = false;
};

}  // namespace gear_sonic::true23::active
