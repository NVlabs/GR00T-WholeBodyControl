#include "true23_shadow_gate.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

namespace {

namespace true23 = gear_sonic::true23;
using nlohmann::json;

constexpr std::string_view kEncoderOnnxSha =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr std::string_view kDecoderOnnxSha =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr std::string_view kCheckpointSha =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr std::string_view kPolicySha =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
constexpr std::string_view kEncoderStateSha =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
constexpr std::string_view kDecoderStateSha =
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
constexpr std::string_view kReportSha =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr std::string_view kReportPayloadSha =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr std::string_view kEncoderConfigSha =
    "3333333333333333333333333333333333333333333333333333333333333333";
constexpr std::string_view kDecoderConfigSha =
    "4444444444444444444444444444444444444444444444444444444444444444";
constexpr std::string_view kPolicyConfigSha =
    "5555555555555555555555555555555555555555555555555555555555555555";
constexpr std::string_view kSimConfigSha =
    "6666666666666666666666666666666666666666666666666666666666666666";
constexpr std::string_view kOtherSha =
    "7777777777777777777777777777777777777777777777777777777777777777";

class TestRunner {
 public:
  void Check(bool condition, std::string message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const { return failures_; }

 private:
  int failures_ = 0;
};

json MakeContract(
    std::string_view reference_profile =
        true23::kNormalReferenceProfile) {
  const auto lower_body_term =
      reference_profile == true23::kCausalHistoryReferenceProfile
          ? "causal_history_lower_body"
          : "command_multi_future_lower_body";
  return {
      {"schema_version", true23::kArtifactSchemaVersion},
      {"robot_model", true23::kRobotModel},
      {"required_mode_machine", true23::kRequiredModeMachine},
      {"observation_layout", true23::kObservationLayout},
      {"history_length", true23::kHistoryLength},
      {"history_order", "oldest_to_newest"},
      {"term_order",
       {"base_ang_vel", "joint_pos_rel", "joint_vel",
        "previous_action", "projected_gravity"}},
      {"token_dim", true23::kEncoderOutputDim},
      {"proprioception_dim", 930},
      {"decoder_input_dim", true23::kDecoderInputDim},
      {"decoder_output_dim", true23::kDecoderOutputDim},
      {"decoder_output_layout", true23::kDecoderOutputLayout},
      {"reference_profile", reference_profile},
      {"reference_contract",
       true23::ReferenceProfileContract(reference_profile)},
      {"source_il29_keep_indices",
       true23::detail::ToJsonArray(true23::kSourceIl29KeepIndices)},
      {"source_il29_excluded_indices",
       true23::detail::ToJsonArray(
           true23::kSourceIl29ExcludedIndices)},
      {"native_il23_to_canonical_il29",
       true23::detail::ToJsonArray(true23::kNativeToCanonicalIl29)},
      {"source_mj29_to_native_il23",
       true23::detail::ToJsonArray(true23::kSourceMj29ToNative)},
      {"missing_fill",
       {
           {"joint_pos_rel", "fixed_default_relative_zero"},
           {"joint_vel", "zero"},
           {"previous_action", "zero_every_history_frame"},
       }},
      {"hardware_joint_ids",
       true23::detail::ToJsonArray(true23::kHardwareJointIds)},
      {"excluded_hardware_joint_ids",
       true23::detail::ToJsonArray(
           true23::kExcludedHardwareJointIds)},
      {"native_il23_joint_names",
       true23::detail::ToJsonArray(true23::kNativeJointNames)},
      {"hardware_joint_names",
       true23::detail::ToJsonArray(true23::kHardwareJointNames)},
      {"hardware_action_scale",
       true23::detail::ToJsonArray(true23::kHardwareActionScale)},
      {"native_il23_action_scale",
       true23::detail::ToJsonArray(true23::kNativeActionScale)},
      {"isaaclab_to_mujoco_dof",
       true23::detail::ToJsonArray(
           true23::kNativeToHardwareCompact)},
      {"mujoco_to_isaaclab_dof",
       true23::detail::ToJsonArray(
           true23::kHardwareCompactToNative)},
      {"teleop_encoder",
       {
           {"input_dim", true23::kEncoderInputDim},
           {"input_term_order",
            {lower_body_term,
             "vr_3point_local_target",
             "vr_3point_local_orn_target",
             "motion_anchor_ori_b"}},
           {"input_term_dims", {240, 9, 12, 6}},
           {"layer_dims", {267, 2048, 1024, 512, 512, 64}},
           {"linear_indices", {0, 2, 4, 6, 8}},
           {"activation", "SiLU"},
           {"token_count", 2},
           {"token_width", 32},
           {"output_dim", true23::kEncoderOutputDim},
           {"fsq_level", 32},
           {"fsq_formula", "fsq_tanh_round_ste_even_levels_v1"},
           {"config_sha256", kEncoderConfigSha},
       }},
      {"decoder",
       {
           {"layer_dims",
            true23::DecoderLayerDimsForReferenceProfile(
                reference_profile)},
           {"linear_indices",
            true23::DecoderLinearIndicesForReferenceProfile(
                reference_profile)},
           {"activation", "SiLU"},
           {"config_sha256", kDecoderConfigSha},
       }},
      {"policy_config_sha256", kPolicyConfigSha},
      {"sim_validation",
       {
           {"config_sha256", kSimConfigSha},
           {"control_hz", 50},
           {"minimum_coverage",
            {
                {"seeds_per_scenario", 3},
                {"episodes_per_scenario", 64},
                {"seconds_per_episode", 5.0},
                {"steps_per_episode", 250},
            }},
       }},
      {"encoder_onnx",
       {
           {"opset", true23::kOnnxOpset},
           {"input_name", true23::kEncoderInputName},
           {"input_shape", {1, true23::kEncoderInputDim}},
           {"input_dtype", "float32"},
           {"output_name", true23::kEncoderOutputName},
           {"output_shape", {1, true23::kEncoderOutputDim}},
           {"output_dtype", "float32"},
           {"dynamic_axes", false},
       }},
      {"decoder_onnx",
       {
           {"opset", true23::kOnnxOpset},
           {"input_name", true23::kDecoderInputName},
           {"input_shape", {1, true23::kDecoderInputDim}},
           {"input_dtype", "float32"},
           {"output_name", true23::kDecoderOutputName},
           {"output_shape", {1, true23::kDecoderOutputDim}},
           {"output_dtype", "float32"},
           {"dynamic_axes", false},
       }},
  };
}

json MakeRuntimeSourceManifest() {
  const json files = json::array(
      {
          {
              {"relpath", "gear_sonic/a.py"},
              {"size_bytes", 10},
              {"sha256", kEncoderConfigSha},
          },
          {
              {"relpath", "gear_sonic/b.py"},
              {"size_bytes", 20},
              {"sha256", kDecoderConfigSha},
          },
      });
  return {
      {"schema_version", 1},
      {"file_count", 2},
      {"total_bytes", 30},
      {"manifest_sha256", true23::Sha256CanonicalJson(files)},
      {"files", files},
  };
}

json MakeRobotAssetManifest() {
  const json files = json::array(
      {
          {
              {"relpath", "gear_sonic/data/robots/g1/model.urdf"},
              {"size_bytes", 40},
              {"sha256", kOtherSha},
          },
      });
  return {
      {"schema_version", 1},
      {"file_count", 1},
      {"total_bytes", 40},
      {"manifest_sha256", true23::Sha256CanonicalJson(files)},
      {"files", files},
  };
}

json MakeTrainingEvidence(std::string_view reference_profile) {
  const bool low_latency =
      reference_profile == true23::kLowLatencyReferenceProfile ||
      reference_profile == true23::kCausalHistoryReferenceProfile;
  const json resolved_config = {
      {"reference_profile", reference_profile},
      {"experiment", "true23-test"},
  };
  const json material_config = {
      {"reference_profile", reference_profile},
      {"reward_contract", "true23-test"},
  };
  return {
      {"schema_version", 3},
      {"kind", "g1_23dof_training_checkpoint"},
      {"producer", "gear_sonic.trl.callbacks.ModelSaveCallback"},
      {"robot_model", true23::kRobotModel},
      {"global_step", 100},
      {"history_length", true23::kHistoryLength},
      {"observation_layout", true23::kObservationLayout},
      {"decoder_input_dim", true23::kDecoderInputDim},
      {"decoder_output_dim", true23::kDecoderOutputDim},
      {"reference_profile", reference_profile},
      {"reference_contract",
       true23::ReferenceProfileContract(reference_profile)},
      {"source_family",
       low_latency ? "sonic_low_latency" : "sonic_release"},
      {"source_revision",
       low_latency
           ? json(true23::kLowLatencyReleaseRevision)
           : json(nullptr)},
      {"source_checkpoint_sha256",
       low_latency
           ? true23::kLowLatencyReleaseSha256
           : true23::kNormalReleaseSha256},
      {"initial_policy_state_sha256",
       low_latency
           ? true23::kLowLatencyInitialPolicySha256
           : true23::kNormalInitialPolicySha256},
      {"training_start_global_step", 0},
      {"training_updates", 100},
      {"minimum_training_updates", true23::kMinimumTrainingUpdates},
      {"policy_state_sha256", kPolicySha},
      {"weights_only_initialization", false},
      {"training_material",
       {
           {"schema_version", 1},
           {"resolved_config", resolved_config},
           {"resolved_config_sha256",
            true23::Sha256CanonicalJson(resolved_config)},
           {"material_config", material_config},
           {"material_config_sha256",
            true23::Sha256CanonicalJson(material_config)},
           {"runtime_source", MakeRuntimeSourceManifest()},
           {"robot_assets", MakeRobotAssetManifest()},
       }},
      {"motion_dataset",
       {
           {"schema_version", 1},
           {"source_archive",
            {
                {"repository", true23::kMotionDatasetRepository},
                {"revision", true23::kMotionDatasetRevision},
                {"relpath", true23::kMotionDatasetArchiveRelpath},
                {"size_bytes",
                 true23::kMotionDatasetArchiveSizeBytes},
                {"sha256", true23::kMotionDatasetArchiveSha256},
            }},
           {"processed",
            {
                {"root_relpath",
                 true23::kProcessedMotionDatasetRootRelpath},
                {"file_count", 2},
                {"total_bytes", 1024},
                {"manifest_sha256", kOtherSha},
            }},
       }},
  };
}

json MakeEmbedded(
    std::string_view role,
    const json& contract,
    const json& training_evidence) {
  return {
      {"artifact_kind", true23::kArtifactKind},
      {"artifact_role", role},
      {"contract", contract},
      {"checkpoint_stage", "trained"},
      {"training_global_step", 100},
      {"training_evidence", training_evidence},
      {"checkpoint_sha256", kCheckpointSha},
      {"policy_state_sha256", kPolicySha},
      {"encoder_state_sha256", kEncoderStateSha},
      {"decoder_state_sha256", kDecoderStateSha},
      {"sim_validation_computed", true},
      {"sim_report_sha256", kReportSha},
      {"sim_report_payload_sha256", kReportPayloadSha},
  };
}

json MakeValidationRecord() {
  return {
      {"onnx_checker_full_check", true},
      {"shape_inference", true},
      {"ort_provider", "CPUExecutionProvider"},
      {"parity_case_count", 3},
  };
}

json MakeMujocoValidationRecord() {
  return {
      {"onnx_checker_full_check", true},
      {"shape_inference", true},
      {"ort_provider", "CPUExecutionProvider"},
      {"parity_case_count", 3},
      {"parity_atol", 1.0e-5},
      {"parity_rtol", 1.0e-4},
      {"parity_max_abs_error", 1.0e-6},
      {"parity_max_rel_error", 1.0e-5},
      {"parity_inputs_sha256", kEncoderConfigSha},
      {"parity_outputs_sha256", kDecoderConfigSha},
  };
}

true23::ModelSignature MakeEncoderSignature() {
  return {
      .input_count = 1,
      .output_count = 1,
      .input_name = std::string(true23::kEncoderInputName),
      .output_name = std::string(true23::kEncoderOutputName),
      .input_shape = {1, true23::kEncoderInputDim},
      .output_shape = {1, true23::kEncoderOutputDim},
      .input_float32 = true,
      .output_float32 = true,
      .default_opsets = {true23::kOnnxOpset},
  };
}

true23::ModelSignature MakeDecoderSignature() {
  return {
      .input_count = 1,
      .output_count = 1,
      .input_name = std::string(true23::kDecoderInputName),
      .output_name = std::string(true23::kDecoderOutputName),
      .input_shape = {1, true23::kDecoderInputDim},
      .output_shape = {1, true23::kDecoderOutputDim},
      .input_float32 = true,
      .output_float32 = true,
      .default_opsets = {true23::kOnnxOpset},
  };
}

struct Fixture {
  json sidecar;
  json encoder_embedded;
  json decoder_embedded;
  true23::ModelSignature encoder_signature = MakeEncoderSignature();
  true23::ModelSignature decoder_signature = MakeDecoderSignature();
  true23::PairBinding binding{
      .encoder_filename = "policy_encoder.onnx",
      .decoder_filename = "policy_decoder.onnx",
      .metadata_filename = "policy.metadata.json",
      .encoder_onnx_sha256 = std::string(kEncoderOnnxSha),
      .decoder_onnx_sha256 = std::string(kDecoderOnnxSha),
      .encoder_path = "/models/policy_encoder.onnx",
      .decoder_path = "/models/policy_decoder.onnx",
      .metadata_path = "/models/policy.metadata.json",
      .metadata_sha256 = std::string(kOtherSha),
  };
};

Fixture MakeFixture(
    std::string_view reference_profile =
        true23::kNormalReferenceProfile) {
  Fixture fixture;
  const auto contract = MakeContract(reference_profile);
  const auto training_evidence =
      MakeTrainingEvidence(reference_profile);
  fixture.encoder_embedded =
      MakeEmbedded(
          true23::kEncoderRole, contract, training_evidence);
  fixture.decoder_embedded =
      MakeEmbedded(
          true23::kDecoderRole, contract, training_evidence);
  const json material_provenance = {
      {"schema_version", 1},
      {"runtime_source",
       training_evidence["training_material"]["runtime_source"]},
      {"motion_dataset", training_evidence["motion_dataset"]},
  };

  fixture.sidecar = {
      {"schema_version", true23::kArtifactSchemaVersion},
      {"artifact_kind", true23::kArtifactKind},
      {"robot_model", true23::kRobotModel},
      {"mode_machine", true23::kRequiredModeMachine},
      {"action_dof", true23::kDecoderOutputDim},
      {"hardware_joint_ids",
       true23::detail::ToJsonArray(true23::kHardwareJointIds)},
      {"excluded_hardware_joint_ids",
       true23::detail::ToJsonArray(
           true23::kExcludedHardwareJointIds)},
      {"decoder_output_layout", true23::kDecoderOutputLayout},
      {"observation_layout", true23::kObservationLayout},
      {"history_length", true23::kHistoryLength},
      {"decoder_input_dim", true23::kDecoderInputDim},
      {"decoder_output_dim", true23::kDecoderOutputDim},
      {"reference_profile", reference_profile},
      {"reference_contract",
       true23::ReferenceProfileContract(reference_profile)},
      {"onnx_opset", true23::kOnnxOpset},
      {"checkpoint_stage", "trained"},
      {"deployment_ready", true},
      {"sim_validation_passed", true},
      {"naive_output_masking", false},
      {"encoder_onnx_filename", fixture.binding.encoder_filename},
      {"decoder_onnx_filename", fixture.binding.decoder_filename},
      {"metadata_filename", fixture.binding.metadata_filename},
      {"training_evidence", training_evidence},
      {"simulation_evidence",
       {
           {"schema_version", true23::kSimulationSchemaVersion},
           {"computed_pass", true},
           {"producer",
            {
                {"kind",
                 "gear_sonic_true23_isaaclab_disturbance_validation"},
                {"version", 1},
                {"runner_sha256", kOtherSha},
            }},
           {"runtime_config",
            {
                {"resolved_config_sha256", kOtherSha},
                {"material_config_sha256", kEncoderConfigSha},
            }},
           {"material_provenance", material_provenance},
           {"trace_manifest_sha256", kOtherSha},
           {"trace_count", 9},
           {"required_scenarios",
            {"nominal", "disturbance_50", "disturbance_100"}},
           {"run_count", 9},
           {"total_episodes", 192},
           {"total_steps", 48000},
           {"scenario_coverage",
            {
                {"nominal",
                 {{"seed_count", 3},
                  {"episodes", 64},
                  {"steps", 16000}}},
                {"disturbance_50",
                 {{"seed_count", 3},
                  {"episodes", 64},
                  {"steps", 16000}}},
                {"disturbance_100",
                 {{"seed_count", 3},
                  {"episodes", 64},
                  {"steps", 16000}}},
            }},
           {"simulator",
            {
                {"name", "IsaacLab"},
                {"version", "1.0-test"},
                {"asset_sha256", kOtherSha},
                {"robot_config_sha256", kOtherSha},
                {"config_sha256", kSimConfigSha},
                {"runtime_config_sha256", kOtherSha},
            }},
           {"max_metrics",
            {
                {"phantom_observation_max_abs", 0.0},
                {"max_recovery_time_s", 1.0},
                {"action_saturation_fraction", 0.05},
                {"mpjpe_m", 0.10},
            }},
           {"checkpoint_sha256", kCheckpointSha},
           {"report_sha256", kReportSha},
           {"report_payload_sha256", kReportPayloadSha},
       }},
      {"observation_contract",
       {
           {"native_il23_to_canonical_il29",
            true23::detail::ToJsonArray(
                true23::kNativeToCanonicalIl29)},
           {"source_il29_keep_indices",
            true23::detail::ToJsonArray(
                true23::kSourceIl29KeepIndices)},
           {"source_il29_excluded_indices",
            true23::detail::ToJsonArray(
                true23::kSourceIl29ExcludedIndices)},
           {"term_order",
            {"base_ang_vel", "joint_pos_rel", "joint_vel",
             "previous_action", "projected_gravity"}},
           {"history_order", "oldest_to_newest"},
           {"missing_fill",
            {
                {"joint_pos_rel", "fixed_default_relative_zero"},
                {"joint_vel", "zero"},
                {"previous_action", "zero_every_history_frame"},
            }},
       }},
      {"action_contract",
       {
           {"native_il23_joint_names",
            true23::detail::ToJsonArray(true23::kNativeJointNames)},
           {"hardware_joint_names",
            true23::detail::ToJsonArray(true23::kHardwareJointNames)},
           {"isaaclab_to_mujoco_dof",
            true23::detail::ToJsonArray(
                true23::kNativeToHardwareCompact)},
           {"mujoco_to_isaaclab_dof",
            true23::detail::ToJsonArray(
                true23::kHardwareCompactToNative)},
           {"hardware_action_scale",
            true23::detail::ToJsonArray(
                true23::kHardwareActionScale)},
           {"native_il23_action_scale",
            true23::detail::ToJsonArray(true23::kNativeActionScale)},
       }},
      {"validation",
       {
           {"teleop_encoder", MakeValidationRecord()},
           {"true23_decoder", MakeValidationRecord()},
           {"pair_dry_run", true},
       }},
      {"hashes",
       {
           {"checkpoint_sha256", kCheckpointSha},
           {"policy_state_sha256", kPolicySha},
           {"encoder_state_sha256", kEncoderStateSha},
           {"decoder_state_sha256", kDecoderStateSha},
           {"encoder_onnx_sha256", kEncoderOnnxSha},
           {"decoder_onnx_sha256", kDecoderOnnxSha},
           {"sim_report_sha256", kReportSha},
           {"sim_report_payload_sha256", kReportPayloadSha},
           {"contract_sha256", true23::Sha256CanonicalJson(contract)},
           {"robot_asset_sha256", kOtherSha},
           {"robot_config_sha256", kOtherSha},
           {"sim_config_sha256", kSimConfigSha},
           {"encoder_config_sha256", kEncoderConfigSha},
           {"decoder_config_sha256", kDecoderConfigSha},
           {"policy_config_sha256", kPolicyConfigSha},
           {"encoder_embedded_metadata_sha256",
            true23::Sha256CanonicalJson(fixture.encoder_embedded)},
           {"decoder_embedded_metadata_sha256",
            true23::Sha256CanonicalJson(fixture.decoder_embedded)},
       }},
  };
  fixture.sidecar["metadata_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.sidecar);
  return fixture;
}

json MakeMujocoEmbedded(
    std::string_view role,
    std::string_view reference_profile,
    std::string_view training_evidence_sha256,
    std::string_view contract_sha256) {
  return {
      {"schema_version", true23::kMujocoCandidateSchemaVersion},
      {"kind", true23::kMujocoCandidateEmbeddedKind},
      {"artifact_role", role},
      {"promotion_stage", true23::kMujocoCandidateStage},
      {"deployment_authorized", false},
      {"robot_model", true23::kRobotModel},
      {"checkpoint_stage", "trained"},
      {"checkpoint_sha256", kCheckpointSha},
      {"policy_state_sha256", kPolicySha},
      {"encoder_state_sha256", kEncoderStateSha},
      {"decoder_state_sha256", kDecoderStateSha},
      {"training_evidence_sha256", training_evidence_sha256},
      {"contract_sha256", contract_sha256},
      {"global_step", 100},
      {"reference_profile", reference_profile},
      {"reference_contract",
       true23::ReferenceProfileContract(reference_profile)},
      {"encoder_input_dim", true23::kEncoderInputDim},
      {"token_dim", true23::kEncoderOutputDim},
      {"decoder_input_dim", true23::kDecoderInputDim},
      {"decoder_output_dim", true23::kDecoderOutputDim},
      {"naive_output_masking", false},
  };
}

struct MujocoFixture {
  json candidate;
  json promotion;
  json encoder_embedded;
  json decoder_embedded;
  true23::ModelSignature encoder_signature = MakeEncoderSignature();
  true23::ModelSignature decoder_signature = MakeDecoderSignature();
  true23::PairBinding binding{
      .encoder_filename = "policy_encoder.onnx",
      .decoder_filename = "policy_decoder.onnx",
      .metadata_filename = "policy.metadata.json",
      .encoder_onnx_sha256 = std::string(kEncoderOnnxSha),
      .decoder_onnx_sha256 = std::string(kDecoderOnnxSha),
      .encoder_path = "/models/policy_encoder.onnx",
      .decoder_path = "/models/policy_decoder.onnx",
      .metadata_path = "/models/policy.metadata.json",
      .metadata_sha256 = std::string(kOtherSha),
  };
};

void RefreshMujocoPromotionPayload(MujocoFixture& fixture) {
  fixture.candidate.erase("metadata_payload_sha256");
  fixture.candidate["metadata_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.candidate);
  const auto candidate_full_sha =
      true23::Sha256CanonicalJson(fixture.candidate);
  for (auto* source :
       {
           &fixture.promotion["source_candidate"],
           &fixture.promotion["mujoco_evidence"]["source_artifact"],
       }) {
    (*source)["candidate_manifest_sha256"] =
        fixture.binding.metadata_sha256;
    (*source)["candidate_manifest_payload_sha256"] =
        candidate_full_sha;
    (*source)["candidate_claimed_payload_sha256"] =
        fixture.candidate["metadata_payload_sha256"];
  }
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
}

MujocoFixture MakeMujocoFixture() {
  MujocoFixture fixture;
  const auto legacy = MakeFixture();
  fixture.candidate = {
      {"schema_version", legacy.sidecar["schema_version"]},
      {"robot_model", legacy.sidecar["robot_model"]},
      {"mode_machine", legacy.sidecar["mode_machine"]},
      {"action_dof", legacy.sidecar["action_dof"]},
      {"hardware_joint_ids", legacy.sidecar["hardware_joint_ids"]},
      {"excluded_hardware_joint_ids",
       legacy.sidecar["excluded_hardware_joint_ids"]},
      {"decoder_output_layout",
       legacy.sidecar["decoder_output_layout"]},
      {"observation_layout", legacy.sidecar["observation_layout"]},
      {"history_length", legacy.sidecar["history_length"]},
      {"decoder_input_dim", legacy.sidecar["decoder_input_dim"]},
      {"decoder_output_dim", legacy.sidecar["decoder_output_dim"]},
      {"reference_profile", legacy.sidecar["reference_profile"]},
      {"reference_contract", legacy.sidecar["reference_contract"]},
      {"checkpoint_stage", "trained"},
      {"deployment_ready", false},
      {"sim_validation_passed", false},
      {"naive_output_masking", false},
      {"observation_contract", legacy.sidecar["observation_contract"]},
      {"action_contract", legacy.sidecar["action_contract"]},
      {"artifact_kind", true23::kMujocoCandidateKind},
      {"promotion_stage", true23::kMujocoCandidateStage},
      {"deployment_authorized", false},
      {"encoder_onnx_filename", fixture.binding.encoder_filename},
      {"decoder_onnx_filename", fixture.binding.decoder_filename},
      {"metadata_filename", fixture.binding.metadata_filename},
      {"onnx_opset", true23::kOnnxOpset},
      {"asset_provenance",
       {
           {"repository", true23::kPinnedUnitreeRepository},
           {"revision", true23::kPinnedUnitreeRevision},
           {"root_relpath", true23::kPinnedUnitreeRootRelpath},
           {"text_normalization",
            true23::kPinnedUnitreeTextNormalization},
           {"file_count", true23::kPinnedUnitreeFileCount},
           {"total_bytes", true23::kPinnedUnitreeTotalBytes},
           {"manifest_sha256",
            true23::kPinnedUnitreeManifestSha256},
           {"urdf_sha256", true23::kPinnedUnitreeUrdfSha256},
           {"mjcf_sha256", true23::kPinnedUnitreeMjcfSha256},
           {"verified", true},
       }},
      {"training_evidence", legacy.sidecar["training_evidence"]},
  };
  fixture.candidate["observation_contract"]["token_dim"] =
      true23::kEncoderOutputDim;
  fixture.candidate["observation_contract"]["proprioception_dim"] =
      true23::kDecoderInputDim - true23::kEncoderOutputDim;

  const auto training_evidence_sha =
      true23::Sha256CanonicalJson(
          fixture.candidate["training_evidence"]);
  const auto contract_sha = true23::Sha256CanonicalJson(
      true23::MujocoCandidateBaseContract(fixture.candidate));
  fixture.encoder_embedded = MakeMujocoEmbedded(
      true23::kEncoderRole, true23::kNormalReferenceProfile,
      training_evidence_sha, contract_sha);
  fixture.decoder_embedded = MakeMujocoEmbedded(
      true23::kDecoderRole, true23::kNormalReferenceProfile,
      training_evidence_sha, contract_sha);
  fixture.candidate["hashes"] = {
      {"checkpoint_sha256", kCheckpointSha},
      {"policy_state_sha256", kPolicySha},
      {"encoder_state_sha256", kEncoderStateSha},
      {"decoder_state_sha256", kDecoderStateSha},
      {"encoder_onnx_sha256", kEncoderOnnxSha},
      {"decoder_onnx_sha256", kDecoderOnnxSha},
      {"training_evidence_sha256", training_evidence_sha},
      {"contract_sha256", contract_sha},
      {"urdf_sha256", true23::kPinnedUnitreeUrdfSha256},
      {"mjcf_sha256", true23::kPinnedUnitreeMjcfSha256},
      {"robot_config_sha256", true23::kPinnedRobotConfigSha256},
      {"asset_manifest_sha256",
       true23::kPinnedUnitreeManifestSha256},
      {"encoder_embedded_metadata_sha256",
       true23::Sha256CanonicalJson(fixture.encoder_embedded)},
      {"decoder_embedded_metadata_sha256",
       true23::Sha256CanonicalJson(fixture.decoder_embedded)},
  };
  fixture.candidate["validation"] = {
      {"teleop_encoder", MakeMujocoValidationRecord()},
      {"true23_decoder", MakeMujocoValidationRecord()},
      {"pair_dry_run",
       {
           {"performed", true},
           {"device", "cpu"},
           {"dtype", "float32"},
           {"token_sha256", kEncoderConfigSha},
           {"action_sha256", kDecoderConfigSha},
       }},
  };

  const json source_artifact = {
      {"artifact_kind", true23::kMujocoCandidateSourceKind},
      {"checkpoint_sha256", kCheckpointSha},
      {"policy_state_sha256", kPolicySha},
      {"encoder_onnx_sha256", kEncoderOnnxSha},
      {"decoder_onnx_sha256", kDecoderOnnxSha},
      {"candidate_manifest_sha256", fixture.binding.metadata_sha256},
      {"candidate_manifest_payload_sha256", kOtherSha},
      {"candidate_claimed_payload_sha256", kOtherSha},
      {"inference_runtime", "onnxruntime_cpu"},
      {"inference_threads", 1},
  };
  json source_candidate = source_artifact;
  source_candidate["checkpoint_filename"] =
      "policy.promotion.pt";
  source_candidate["encoder_onnx_filename"] =
      fixture.binding.encoder_filename;
  source_candidate["decoder_onnx_filename"] =
      fixture.binding.decoder_filename;
  source_candidate["metadata_filename"] =
      fixture.binding.metadata_filename;
  fixture.promotion = {
      {"schema_version", true23::kMujocoPromotionSchemaVersion},
      {"kind", true23::kMujocoPromotionKind},
      {"robot_model", true23::kRobotModel},
      {"promotion_stage", true23::kMujocoPromotedStage},
      {"deployment_authorized", true},
      {"active_motor_control_authorized", false},
      {"checkpoint_stage", "trained"},
      {"source_candidate", source_candidate},
      {"mujoco_evidence",
       {
           {"schema_version",
            true23::kMujocoPromotionSchemaVersion},
           {"computed_pass", true},
           {"report_sha256", kReportSha},
           {"report_payload_sha256", kReportPayloadSha},
           {"source_artifact", source_artifact},
           {"producer",
            {
                {"kind", true23::kMujocoProducerKind},
                {"version", 1},
                {"runner_sha256",
                 true23::kPinnedMujocoRunnerSha256},
                {"runtime_sha256",
                 true23::kPinnedMujocoRuntimeSha256},
            }},
           {"simulator",
            {
                {"name", "MuJoCo"},
                {"version", true23::kMujocoVersion},
                {"mjcf_sha256", true23::kPinnedUnitreeMjcfSha256},
                {"config_sha256",
                 true23::kPinnedMujocoConfigSha256},
                {"physics_contract_sha256",
                 true23::kPinnedMujocoPhysicsContractSha256},
                {"asset_manifest_sha256",
                 true23::kPinnedUnitreeManifestSha256},
            }},
           {"trace_manifest_sha256", kOtherSha},
           {"trace_count", 9},
           {"scenario_count", 3},
           {"run_count", 9},
           {"episodes_per_scenario", 66},
           {"total_episodes", 198},
           {"total_records", 49500},
           {"deterministic_onnx_mujoco_replay_verified", true},
           {"summary_metrics",
            {
                {"run_count", 9},
                {"episode_count", 198},
                {"record_count", 49500},
                {"termination_count", 0},
                {"nonfinite_count", 0},
                {"joint_limit_violation_count", 0},
                {"min_base_height_m", 0.75},
                {"max_tilt_rad", 0.25},
                {"max_tracking_rmse_rad", 0.10},
                {"max_abs_joint_velocity_radps", 10.0},
                {"max_abs_applied_torque_nm", 50.0},
                {"max_abs_native_action", 1.0},
                {"max_abs_native_action_raw", 1.0},
                {"max_action_saturation_fraction", 0.01},
                {"minimum_recovery_fraction", 1.0},
                {"max_recovery_time_s", 1.0},
            }},
       }},
      {"deployment_conditions",
       {
           {"mode_machine", true23::kRequiredModeMachine},
           {"paired_onnx_bytes_must_remain_unchanged", true},
           {"live_shadow_required", true},
           {"gantry_or_rated_support_required_for_first_actuation",
            true},
           {"free_standing_first_actuation_authorized", false},
       }},
  };
  RefreshMujocoPromotionPayload(fixture);
  return fixture;
}

true23::ValidationResult ValidateMujoco(
    const MujocoFixture& fixture,
    true23::RequestedMode mode = true23::RequestedMode::Shadow) {
  return true23::ValidateMujocoShadowPromotion(
      fixture.promotion, fixture.candidate,
      fixture.encoder_embedded, fixture.decoder_embedded,
      fixture.encoder_signature, fixture.decoder_signature,
      fixture.binding, mode);
}

void CheckMujocoRejects(
    TestRunner& runner,
    const MujocoFixture& fixture,
    std::string label) {
  runner.Check(!ValidateMujoco(fixture).ok(), std::move(label));
}

true23::ValidationResult Validate(
    const Fixture& fixture,
    true23::RequestedMode mode = true23::RequestedMode::Shadow) {
  return true23::ValidateShadowArtifact(
      fixture.sidecar, fixture.encoder_embedded,
      fixture.decoder_embedded, fixture.encoder_signature,
      fixture.decoder_signature, fixture.binding, mode);
}

void CheckRejects(
    TestRunner& runner,
    const Fixture& fixture,
    std::string label) {
  runner.Check(!Validate(fixture).ok(), std::move(label));
}

void RefreshTrainingEvidenceBindings(Fixture& fixture) {
  fixture.encoder_embedded["training_evidence"] =
      fixture.sidecar["training_evidence"];
  fixture.decoder_embedded["training_evidence"] =
      fixture.sidecar["training_evidence"];
  fixture.sidecar["hashes"]["encoder_embedded_metadata_sha256"] =
      true23::Sha256CanonicalJson(fixture.encoder_embedded);
  fixture.sidecar["hashes"]["decoder_embedded_metadata_sha256"] =
      true23::Sha256CanonicalJson(fixture.decoder_embedded);
  fixture.sidecar.erase("metadata_payload_sha256");
  fixture.sidecar["metadata_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.sidecar);
}

void TestPairReadinessAndAuthority(TestRunner& runner) {
  const auto fixture = MakeFixture();
  const auto result = Validate(fixture);
  const auto low_latency_fixture =
      MakeFixture(true23::kLowLatencyReferenceProfile);
  const auto causal_fixture =
      MakeFixture(true23::kCausalHistoryReferenceProfile);
  runner.Check(result.ok(),
               "valid trained and simulated encoder/decoder pair accepted");
  runner.Check(
      Validate(low_latency_fixture).ok(),
      "valid low-latency trained encoder/decoder pair accepted");
  runner.Check(
      Validate(causal_fixture).ok(),
      "valid causal-history trained encoder/decoder pair accepted");

  auto causal_namespace_mismatch = causal_fixture;
  causal_namespace_mismatch.encoder_embedded["contract"]
      ["teleop_encoder"]["input_term_order"][0] =
          "command_multi_future_lower_body";
  CheckRejects(
      runner, causal_namespace_mismatch,
      "causal artifact with released future namespace rejected");
  runner.Check(!result.authorization.lowcmd_publisher_allowed,
               "command publisher remains forbidden");
  runner.Check(!result.authorization.motion_mode_release_allowed,
               "motion-mode release remains forbidden");
  runner.Check(!result.authorization.command_writer_allowed,
               "command writer remains forbidden");

  const auto control_result =
      Validate(fixture, true23::RequestedMode::Control);
  runner.Check(!control_result.ok(), "control request rejected");
  runner.Check(!control_result.authorization.lowcmd_publisher_allowed,
               "rejected control cannot gain publisher authority");

  auto decoder_only = fixture;
  decoder_only.encoder_embedded = json::object();
  decoder_only.encoder_signature = {};
  CheckRejects(runner, decoder_only,
               "decoder-only artifact cannot become ready");
}

void TestMetadataFailsClosed(TestRunner& runner) {
  for (const auto& mutation :
       std::array<std::pair<std::string, json>, 8>{
           std::pair<std::string, json>{"mode_machine", 5},
           {"decoder_output_dim", 29},
           {"decoder_output_layout", "canonical_il29"},
           {"checkpoint_stage", "initialization_only"},
           {"deployment_ready", false},
           {"sim_validation_passed", false},
           {"naive_output_masking", true},
           {"onnx_opset", 12},
       }) {
    auto fixture = MakeFixture();
    fixture.sidecar[mutation.first] = mutation.second;
    CheckRejects(runner, fixture,
                 "sidecar mutation rejected: " + mutation.first);
  }

  auto fixture = MakeFixture();
  fixture.sidecar["reference_contract"]["future_frame_offsets_s"][9] =
      0.18;
  CheckRejects(
      runner, fixture,
      "normal artifact with low-latency offset rejected");

  fixture = MakeFixture();
  fixture.decoder_embedded["contract"]["reference_profile"] =
      true23::kLowLatencyReferenceProfile;
  fixture.decoder_embedded["contract"]["reference_contract"] =
      true23::ReferenceProfileContract(
          true23::kLowLatencyReferenceProfile);
  CheckRejects(
      runner, fixture,
      "encoder/decoder reference profile swap rejected");

  fixture = MakeFixture();
  fixture.sidecar["training_evidence"]
                 ["initial_policy_state_sha256"] = kPolicySha;
  CheckRejects(
      runner, fixture,
      "unchanged initialization/final policy rejected");

  fixture = MakeFixture();
  fixture.sidecar["training_evidence"].erase("training_material");
  CheckRejects(
      runner, fixture,
      "missing training material rejected");

  fixture = MakeFixture();
  fixture.sidecar["training_evidence"]["schema_version"] = 2;
  CheckRejects(
      runner, fixture,
      "legacy training evidence schema rejected");

  fixture = MakeFixture();
  fixture.sidecar["training_evidence"]["training_material"]
                 ["resolved_config"]["experiment"] = "tampered";
  CheckRejects(
      runner, fixture,
      "training resolved-config digest mismatch rejected");

  fixture = MakeFixture();
  auto& trained_runtime =
      fixture.sidecar["training_evidence"]["training_material"]
                     ["runtime_source"];
  trained_runtime["files"][0]["sha256"] = kOtherSha;
  trained_runtime["manifest_sha256"] =
      true23::Sha256CanonicalJson(trained_runtime["files"]);
  RefreshTrainingEvidenceBindings(fixture);
  CheckRejects(
      runner, fixture,
      "training/simulation runtime source mismatch rejected");

  fixture = MakeFixture();
  fixture.sidecar["simulation_evidence"].erase(
      "material_provenance");
  CheckRejects(
      runner, fixture,
      "missing simulation material provenance rejected");

  fixture = MakeFixture();
  fixture.sidecar["simulation_evidence"]["material_provenance"]
                 ["runtime_source"]["files"][0]["sha256"] = kOtherSha;
  CheckRejects(
      runner, fixture,
      "tampered runtime source record rejected");

  fixture = MakeFixture();
  fixture.sidecar["simulation_evidence"]["material_provenance"]
                 ["motion_dataset"]["processed"]
                 ["manifest_sha256"] = kEncoderConfigSha;
  CheckRejects(
      runner, fixture,
      "training/simulation motion manifest mismatch rejected");

  fixture = MakeFixture();
  fixture.sidecar["validation"]["pair_dry_run"] = false;
  CheckRejects(runner, fixture, "failed pair dry-run rejected");

  fixture = MakeFixture();
  fixture.sidecar["simulation_evidence"]["max_metrics"]["mpjpe_m"] =
      0.151;
  CheckRejects(runner, fixture,
               "simulation metric above threshold rejected");

  fixture = MakeFixture();
  fixture.sidecar["hardware_joint_ids"][13] = 13;
  CheckRejects(runner, fixture, "wrong hardware ID rejected");

  fixture = MakeFixture();
  std::swap(
      fixture.sidecar["action_contract"]["isaaclab_to_mujoco_dof"][0],
      fixture.sidecar["action_contract"]["isaaclab_to_mujoco_dof"][1]);
  CheckRejects(runner, fixture, "wrong action mapping rejected");

  fixture = MakeFixture();
  fixture.encoder_embedded["artifact_role"] = true23::kDecoderRole;
  CheckRejects(runner, fixture, "duplicate decoder role rejected");

  fixture = MakeFixture();
  fixture.encoder_embedded["sim_validation_computed"] = false;
  CheckRejects(runner, fixture,
               "unvalidated encoder metadata rejected");

  fixture = MakeFixture();
  fixture.decoder_embedded["training_global_step"] = 0;
  CheckRejects(runner, fixture, "zero training step rejected");

  fixture = MakeFixture();
  fixture.decoder_embedded["contract"]["decoder_output_layout"] =
      "masked_il29";
  CheckRejects(runner, fixture,
               "wrong embedded output layout rejected");

  fixture = MakeFixture();
  fixture.encoder_signature.input_shape =
      {-1, true23::kEncoderInputDim};
  CheckRejects(runner, fixture, "dynamic encoder input rejected");

  fixture = MakeFixture();
  fixture.decoder_signature.output_shape = {1, 29};
  CheckRejects(runner, fixture, "29-output decoder rejected");

  fixture = MakeFixture();
  fixture.encoder_signature.default_opsets = {12};
  CheckRejects(runner, fixture, "wrong actual encoder opset rejected");

  fixture = MakeFixture();
  fixture.binding.encoder_filename = "other_encoder.onnx";
  CheckRejects(runner, fixture, "encoder filename mismatch rejected");

  fixture = MakeFixture();
  fixture.binding.decoder_path = fixture.binding.encoder_path;
  CheckRejects(runner, fixture, "aliased ONNX paths rejected");

  fixture = MakeFixture();
  fixture.binding.metadata_filename = fixture.binding.encoder_filename;
  CheckRejects(runner, fixture, "aliased sidecar filename rejected");

  fixture = MakeFixture();
  fixture.binding.decoder_onnx_sha256 = std::string(kOtherSha);
  CheckRejects(runner, fixture, "decoder file hash mismatch rejected");

  fixture = MakeFixture();
  fixture.sidecar["hashes"]["contract_sha256"] = kOtherSha;
  fixture.sidecar["metadata_payload_sha256"] =
      true23::Sha256CanonicalJson(
          [&fixture] {
            auto value = fixture.sidecar;
            value.erase("metadata_payload_sha256");
            return value;
          }());
  CheckRejects(runner, fixture, "wrong exact contract hash rejected");
}

void TestMujocoPromotionFailsClosed(TestRunner& runner) {
  auto fixture = MakeMujocoFixture();
  const auto result = ValidateMujoco(fixture);
  runner.Check(
      result.ok(),
      "valid MuJoCo candidate and promotion accepted for shadow");
  runner.Check(
      !result.authorization.lowcmd_publisher_allowed &&
          !result.authorization.motion_mode_release_allowed &&
          !result.authorization.command_writer_allowed,
      "MuJoCo promotion grants no motor-control authority");
  runner.Check(
      !ValidateMujoco(fixture, true23::RequestedMode::Control).ok(),
      "MuJoCo promotion rejected for control request");

  fixture = MakeMujocoFixture();
  fixture.promotion["active_motor_control_authorized"] = true;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "promotion claiming active motor control rejected");

  fixture = MakeMujocoFixture();
  fixture.candidate["deployment_authorized"] = true;
  RefreshMujocoPromotionPayload(fixture);
  CheckMujocoRejects(
      runner, fixture,
      "candidate claiming deployment authorization rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["deployment_conditions"]["mode_machine"] = 5;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "promotion for wrong mode_machine rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["deployment_conditions"]
                   ["paired_onnx_bytes_must_remain_unchanged"] = false;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "promotion permitting changed ONNX bytes rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["deployment_conditions"]["live_shadow_required"] =
      false;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "promotion without live-shadow requirement rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["deployment_conditions"]
                   ["gantry_or_rated_support_required_for_first_actuation"] =
      false;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "promotion without first-actuation support rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["deployment_conditions"]
                   ["free_standing_first_actuation_authorized"] = true;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "free-standing first actuation authorization rejected");

  fixture = MakeMujocoFixture();
  fixture.candidate["hashes"]["encoder_onnx_sha256"] = kOtherSha;
  RefreshMujocoPromotionPayload(fixture);
  CheckMujocoRejects(
      runner, fixture,
      "candidate ONNX hash differing from loaded bytes rejected");

  fixture = MakeMujocoFixture();
  fixture.candidate["hashes"]["policy_state_sha256"] = kOtherSha;
  fixture.encoder_embedded["policy_state_sha256"] = kOtherSha;
  fixture.decoder_embedded["policy_state_sha256"] = kOtherSha;
  fixture.candidate["hashes"]
                   ["encoder_embedded_metadata_sha256"] =
      true23::Sha256CanonicalJson(fixture.encoder_embedded);
  fixture.candidate["hashes"]
                   ["decoder_embedded_metadata_sha256"] =
      true23::Sha256CanonicalJson(fixture.decoder_embedded);
  fixture.promotion["source_candidate"]["policy_state_sha256"] =
      kOtherSha;
  fixture.promotion["mujoco_evidence"]["source_artifact"]
                   ["policy_state_sha256"] = kOtherSha;
  RefreshMujocoPromotionPayload(fixture);
  CheckMujocoRejects(
      runner, fixture,
      "candidate policy identity differing from training evidence rejected");

  fixture = MakeMujocoFixture();
  fixture.binding.metadata_sha256 = std::string(kReportSha);
  CheckMujocoRejects(
      runner, fixture,
      "promotion candidate-manifest raw-byte hash mismatch rejected");

  fixture = MakeMujocoFixture();
  fixture.encoder_embedded["deployment_authorized"] = true;
  fixture.candidate["hashes"]
                   ["encoder_embedded_metadata_sha256"] =
      true23::Sha256CanonicalJson(fixture.encoder_embedded);
  RefreshMujocoPromotionPayload(fixture);
  CheckMujocoRejects(
      runner, fixture,
      "embedded ONNX deployment authorization rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["mujoco_evidence"]["source_artifact"]
                   ["policy_state_sha256"] = kOtherSha;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "MuJoCo evidence source identity mismatch rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["mujoco_evidence"]["summary_metrics"]
                   ["termination_count"] = 1;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "MuJoCo evidence with termination rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["mujoco_evidence"]["simulator"]["mjcf_sha256"] =
      kOtherSha;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "MuJoCo evidence using different MJCF rejected");

  fixture = MakeMujocoFixture();
  fixture.candidate["hashes"]["robot_config_sha256"] = kOtherSha;
  RefreshMujocoPromotionPayload(fixture);
  CheckMujocoRejects(
      runner, fixture,
      "candidate using unapproved robot config rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["mujoco_evidence"]["producer"]["runner_sha256"] =
      kOtherSha;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "MuJoCo evidence using unapproved runner rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["mujoco_evidence"]["producer"]["runtime_sha256"] =
      kOtherSha;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "MuJoCo evidence using unapproved runtime rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["mujoco_evidence"]["simulator"]["config_sha256"] =
      kOtherSha;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "MuJoCo evidence using unapproved sim config rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["mujoco_evidence"]["simulator"]
                   ["physics_contract_sha256"] = kOtherSha;
  fixture.promotion.erase("promotion_payload_sha256");
  fixture.promotion["promotion_payload_sha256"] =
      true23::Sha256CanonicalJson(fixture.promotion);
  CheckMujocoRejects(
      runner, fixture,
      "MuJoCo evidence using unapproved physics contract rejected");

  fixture = MakeMujocoFixture();
  fixture.promotion["promotion_payload_sha256"] = kOtherSha;
  CheckMujocoRejects(
      runner, fixture,
      "tampered MuJoCo promotion self-hash rejected");
}

void TestMappings(TestRunner& runner) {
  std::array<int, 23> native{};
  for (std::size_t index = 0; index < native.size(); ++index) {
    native[index] = 100 + static_cast<int>(index);
  }
  const auto hardware = true23::NativeToHardwareCompact(native);
  runner.Check(true23::HardwareCompactToNative(hardware) == native,
               "native/hardware permutations round-trip");

  std::array<int, 29> motor_slots{};
  motor_slots.fill(-1);
  for (std::size_t index = 0;
       index < true23::kHardwareJointIds.size(); ++index) {
    motor_slots[static_cast<std::size_t>(
        true23::kHardwareJointIds[index])] = hardware[index];
  }
  runner.Check(true23::HardwareSlotsToNative(motor_slots) == native,
               "exact hardware motor IDs map to native order");

  const auto baseline = true23::HardwareSlotsToNative(motor_slots);
  for (const int excluded_id : true23::kExcludedHardwareJointIds) {
    motor_slots[static_cast<std::size_t>(excluded_id)] = 99999;
  }
  runner.Check(true23::HardwareSlotsToNative(motor_slots) == baseline,
               "excluded slots never enter true23 observation");

  runner.Check(
      true23::kNativeToHardwareMotorIds ==
          std::array<int, 23>{
              0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3,
              9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26,
          },
      "native outputs bind to exact hardware motor IDs");
}

void TestModeMachineGate(TestRunner& runner) {
  true23::ModeMachineShadowGate gate;
  runner.Check(
      gate.Observe(1, 5) == true23::ModeObservation::WrongMode,
      "wrong initial mode rejected");
  runner.Check(gate.stable_samples() == 0,
               "wrong mode resets stability");
  runner.Check(
      gate.Observe(2, 4) == true23::ModeObservation::Waiting,
      "first advancing mode-4 sample waits");
  runner.Check(
      gate.Observe(2, 4) == true23::ModeObservation::DuplicateTick,
      "duplicate LowState tick ignored");
  for (std::uint32_t tick = 3; tick <= 5; ++tick) {
    runner.Check(
        gate.Observe(tick, 4) == true23::ModeObservation::Waiting,
        "gate waits for five advancing LowState ticks");
  }
  runner.Check(
      gate.Observe(6, 4) == true23::ModeObservation::Ready &&
          gate.ready(),
      "fifth advancing mode-4 LowState opens shadow gate");
  runner.Check(
      gate.Observe(6, 5) == true23::ModeObservation::LatchedFailure,
      "same-tick post-ready mode change latches failure");
  runner.Check(
      gate.Observe(8, 4) == true23::ModeObservation::LatchedFailure &&
          gate.latched_failure() && !gate.ready(),
      "latched failure cannot reopen");

  true23::ModeMachineShadowGate contradictory_gate;
  runner.Check(
      contradictory_gate.Observe(10, 4) ==
          true23::ModeObservation::Waiting,
      "contradictory duplicate test starts stability");
  runner.Check(
      contradictory_gate.Observe(10, 5) ==
              true23::ModeObservation::WrongMode &&
          contradictory_gate.stable_samples() == 0,
      "same-tick wrong mode resets pre-ready stability");

  true23::ModeMachineShadowGate regressing_gate;
  runner.Check(
      regressing_gate.Observe(100, 4) ==
          true23::ModeObservation::Waiting,
      "regression test starts with an advancing sample");
  runner.Check(
      regressing_gate.Observe(99, 4) ==
              true23::ModeObservation::TickRegression &&
          regressing_gate.latched_failure(),
      "regressing LowState tick latches failure");
  runner.Check(
      regressing_gate.Observe(100, 4) ==
          true23::ModeObservation::LatchedFailure,
      "alternating regressed ticks cannot reach ready");
}

void TestSha256(TestRunner& runner) {
  true23::Sha256 digest;
  constexpr std::array<std::uint8_t, 3> input = {'a', 'b', 'c'};
  digest.Update(input.data(), input.size());
  runner.Check(
      digest.FinalHex() ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb"
          "410ff61f20015ad",
      "SHA-256 matches the abc test vector");

  const json python_canonical_vector = {
      {"b", {true, "x"}},
      {"a", 1},
  };
  runner.Check(
      true23::Sha256CanonicalJson(python_canonical_vector) ==
          "ab12ff164ee3d53aa5791e4981dead1905cc4c45ff6af92d0"
          "fd0b2f79844f96d",
      "canonical JSON hash matches Python sort_keys+newline fixture");

  constexpr std::array<std::uint8_t, 4> onnx_opset_13 = {
      0x42, 0x02, 0x10, 0x0d,
  };
  runner.Check(
      true23::InspectDefaultOnnxOpsets(onnx_opset_13) ==
          std::vector<std::int64_t>{13},
      "ONNX protobuf inspector reads actual default opset 13");
}

}  // namespace

int main() {
  TestRunner runner;
  TestPairReadinessAndAuthority(runner);
  TestMetadataFailsClosed(runner);
  TestMujocoPromotionFailsClosed(runner);
  TestMappings(runner);
  TestModeMachineGate(runner);
  TestSha256(runner);

  if (runner.failures() != 0) {
    std::cerr << runner.failures()
              << " true23 shadow-gate check(s) failed\n";
    return 1;
  }
  std::cout << "true23 paired shadow-gate harness: all checks passed\n";
  return 0;
}
